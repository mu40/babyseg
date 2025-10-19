"""Evaluation utilities."""

import babyseg as bs
import json
import katy
import logging
import pathlib
import time
import torch
import voxel as vx


logger = logging.getLogger(__name__)


def select_dtype(x, /):
    """Select the narrowest integer type for data.

    Limited to integer types supported by FreeSurfer.

    Parameters
    ----------
    x : torch.Tensor
        Input data.

    Returns
    -------
    {torch.uint8, torch.int16, torch.uint16, torch.int32}
        Integer type.

    Raises
    ------
    TypeError
        For floating-point or complex input data.
    ValueError
        If no integer type matches.

    """
    types = (torch.uint8, torch.int16, torch.uint16, torch.int32)
    if x.dtype.is_floating_point or x.dtype.is_complex:
        raise TypeError(f'{x.dtype} is not an integer type')

    max_val = x.max()
    min_val = x.min()
    logger.debug('input data spans %s to %s range', max_val, min_val)

    for t in types:
        logger.debug('examining type candidate %s', t)
        info = torch.iinfo(t)
        if info.min <= min_val and max_val <= info.max:
            logger.debug('found type %s compatible with data', t)
            return t

    raise ValueError(f'no type in {types} spans {min_val} to {max_val}')


def segment(
    config,
    images,
    checkpoint='latest',
    out_seg=None,
    out_prob=None,
    out_lead=None,
    device=None,
    threads=None,
):
    """Segment an image using a trained model.

    Parameters
    ----------
    config : os.PathLike or dict
        Model configuration
    images : os.PathLike or sequence of os.PathLike
        Input images.
    checkpoint : 'latest' or int or os.PathLike, optional
        Model checkpoint.
    out_seg : os.PathLike, optional
        Output label map.
    out_prob : os.PathLike, optional
        Output probability maps.
    out_lead : os.PathLike, optional
        Conformed lead image output.
    device : torch.device, optional
        Device used for inference.
    threads : int, optional
        Number of intraop threads on CPU.

    """
    # Inputs.
    if not isinstance(config, dict):
        config = bs.config.load(config)
    if isinstance(images, (str, pathlib.Path)):
        images = [images]
    logger.info('received inputs %s', images)

    # Outputs.
    if not any((out_seg, out_prob, out_lead)):
        logger.error('received no output file paths')
        exit(1)

    # Model.
    device = torch.device(device)
    if threads is not None:
        torch.set_num_threads(threads)

    model = bs.config.load_model(config, init=checkpoint, device=device)
    model.eval()

    # Lead image.
    ori = config['eval']['orientation']
    spacing = config['eval']['spacing']
    lead = vx.load_volume(images[0]).to(device).float()
    conf = lead.reorient(ori).resample(spacing).crop_to_nonzero()
    logger.info('reoriented lead image to "%s"', ori)
    logger.info('resampled lead image to voxel spacing %s mm', spacing)
    logger.info('cropped lead image to bounding box %s', conf.baseshape)

    # Shape.
    div = config['eval']['divisor']
    shape = torch.tensor(conf.baseshape)
    shape = shape.div(div).ceil().long().mul(div).clamp(128, 320)
    conf = conf.reshape(shape)
    logger.info('reshaped lead image to %s', conf.baseshape)
    if out_lead:
        conf.save(out_lead)
        logger.info('saved conformed lead image to "%s"', out_lead)

    # Resampling. Create multi-channel batch.
    logger.info('conforming remaining images to lead image')
    images = map(vx.load_volume, images[1:])
    images = (i.to(device).float().resample_like(conf) for i in images)
    images = vx.volume.stack(conf, *images).tensor.unsqueeze(0)

    # Inference.
    logger.info('running model on tensor of %s', images.shape)
    with torch.no_grad():
        start = time.time()
        out = model(images).squeeze()
        logger.info('inference took %.2f seconds', time.time() - start)
        logger.debug('received model output %s', out.shape)
        out = conf.new(out).resample_like(lead)
        logger.debug('resampled model output %s', out.shape)

    # Probability maps.
    if out_prob:
        out.save(out_prob)
        logger.info('saved probability maps to "%s"', out_prob)

    # Label map.
    if out_seg:
        labels = config['synthesis']['kwargs']['optimize']
        logger.debug('remapping one-hot to labels using "%s"', labels)
        lut = list(map(int, katy.io.load(labels)))
        lut = torch.tensor(lut, dtype=torch.uint8, device=device)
        out = lut[out.tensor.argmax(0)]

        lead.new(out).save(out_seg)
        logger.info('saved label map to "%s"', out_seg)


def remap_labels(input, mapping, output, colors=None):
    """Remap the labels of a label map.

    Parameters
    ----------
    input : os.PathLike
        Input label map.
    mapping : os.PathLike
        Dictionary mapping from input to output labels, in a JSON file.
    output : os.PathLike
        Output path.
    colors : os.PathLike, optional
        Embed a color table. Requires Surfa.

    """
    # Mapping.
    logger.info('loading label mapping from "%s"', mapping)
    with open(mapping) as f:
        mapping = json.load(f)
        mapping = {int(k): int(v) for k, v in mapping.items()}
        logger.debug('using mapping %s', mapping)

    # Lookup table. Keep missing labels as is.
    logger.info('reading input label map "%s"', input)
    out = vx.load_volume(input).int()
    size = max(max(mapping), out.max()) + 1
    lut = torch.arange(size)
    for old, new in mapping.items():
        lut[old] = new

    # Recoding.
    logger.debug('constructed LUT %s', lut)
    logger.info('recoding label map')
    out = out.new(lut[out.tensor])

    # Output.
    dtype = select_dtype(out)
    logger.info('selected output data type %s', dtype)
    out.type(dtype).save(output)
    logger.info('saved output to "%s"', output)

    # Color table.
    if colors:
        import surfa
        out = surfa.load_volume(output)
        out.labels = surfa.load_label_lookup(colors)
        out.save(output)
        logger.info('embedded color table "%s"', colors)


def save_metrics(config, /, split, labels, metrics, test=False):
    """Save metrics dictionary structure.

    Parameters
    ----------
    config : dict
        Model configuration
    split : str or os.PathLike
        Split JSON file.
    labels : str
        Name of evaluation labels.
    metrics : dict
        Metric information.
    test : bool, optional
        Do not record Git hashes, to prevent failure on uncommitted change.

    """
    # Variables.
    split = pathlib.Path(split).stem
    method = config['cache']['name']
    checkpoint = config['eval'].get('checkpoint')
    epoch = None
    if checkpoint:
        checkpoint = str(checkpoint)
        epoch = bs.state.epoch(config, checkpoint)

    # Data.
    out = {
        'split': split,
        'labels': labels,
        'method': method,
        'checkpoint': checkpoint,
        'epoch': epoch,
        'metrics': metrics,
        **bs.config.env(config, test=test),
    }

    # Output.
    path = config['eval']['save_name']
    path = path.format(split=split, labels=labels, method=method)
    pathlib.Path(path).parent.mkdir(exist_ok=True, parents=True)
    katy.io.save(out, path)
    logger.info('saved metrics to "%s"', path)


def validate_sample(sample, config):
    """Ensure sample images match method inputs and vice versa.

    Parameters
    ----------
    sample : babyseg.data.Sample
        Data sample.
    config : dict
        Model configuration.

    Raises
    ------
    ValueError
        If a sample image is not a method input or vice versa.

    """
    logger.info('validating sample "%s"', sample)
    inputs = config['eval'].get('inputs')
    if not inputs:
        return

    if any(t not in inputs for t in sample.image_types):
        raise ValueError(f'image not among inputs {inputs}')

    if any(t not in sample.image_types for t in inputs):
        raise ValueError(f'input not among images {sample.image_types}')


def remap_sample(sample, config, force=False):
    """Remap the labels of a sample.

    Parameters
    ----------
    sample : babyseg.data.Sample
        Data sample.
    config : dict
        Model configuration.
    force : bool, optional
        Override existing outputs.

    Raises
    ------
    ValueError
        If a sample image is not a method input or vice versa.

    """
    method = config['cache']['name']
    native = config['eval']['native']
    labels = config['eval']['labels']

    # Predicted labels.
    logger.info('mapping "%s" prediction to labels %s', sample, list(labels))
    for label_set in labels:
        inp = sample.output(method, labels=native)
        out = sample.output(method, labels=label_set)
        if out.exists() and not force:
            logger.info('skipping existing sample "%s"', sample)
            continue

        mapping = config['eval']['mapping_pred'][label_set]
        remap_labels(inp, mapping, out)

    # Ground-truth labels.
    logger.info('mapping "%s" ground truth to labels %s', sample, list(labels))
    for label_set in labels:
        inp = sample.label_map()
        out = sample.label_map(labels=label_set)
        if inp == out or (out.exists() and not force):
            logger.info('skipping existing sample "%s"', sample)
            continue

        mapping = config['eval']['mapping_true'][label_set]
        remap_labels(inp, mapping, out)


def score_sample(sample, config, label_set, device=None, decimals=5):
    """Compute metrics for a sample.

    Parameters
    ----------
    sample : babyseg.data.Sample
        Data sample.
    config : dict
        Model configuration.
    label_set : str
        Label set name used when segmenting.
    device : torch.device, optional
        Device to use for computation.
    decimals : int, optional
        Number of decimal places to keep.

    Returns
    -------
    dict
        Metric information.

    """
    # Setup.
    method = config['cache']['name']
    labels = config['eval']['labels'][label_set]
    labels = katy.io.load(labels).items()
    labels = {int(k): v for k, v in labels}

    # Input. Resampling is a no-op if already in the same space.
    true = sample.label_map(label_set)
    pred = sample.output(method, label_set)
    true = vx.load_volume(true).to(device)
    pred = vx.load_volume(pred).to(device).resample_like(true, mode='nearest')

    # Dice.
    true = true.tensor[None, None]
    pred = pred.tensor[None, None]
    dice = katy.metrics.dice(true, pred, labels=list(labels))
    dice = [round(d.item(), decimals) for d in dice.squeeze().cpu()]

    return {
        'sample': sample.name,
        'age': sample.age,
        **{f'Dice-{k}': v for k, v in zip(labels.values(), dice, strict=True)},
    }
