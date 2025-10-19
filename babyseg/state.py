"""Checkpointing utilities."""

import babyseg as bs
import logging
import pathlib
import re
import torch


logger = logging.getLogger(__name__)


def load(conf, /, **kwargs):
    """Initialize objects from a general checkpoint for training.

    The function initializes objects based on the configuration:
    - new experiment (no checkpoints) -> `init` if defined, return 0
    - existing experiment, `resume` False -> `init` if defined, return 0
    - existing experiment, `resume` True -> latest checkpoint, return epoch
    - existing experiment, `resume` int -> specific checkpoint, return epoch

    Parameters
    ----------
    conf : os.PathLike or dict
        Model configuration.
    **kwargs : dict
        Key-value pairs of keys used for saving, objects to initialize.

    Returns
    -------
    int
        Epoch.

    """
    if not isinstance(conf, dict):
        conf = bs.config.load(conf)

    init = conf['training'].get('init')
    resume = conf['training']['resume']
    try:
        checkpoints = bs.state.list(conf)
    except FileNotFoundError:
        checkpoints = []

    # New experiment or existing experiment with no-resume. Exit early.
    if not checkpoints or (isinstance(resume, bool) and not resume):
        epoch = 0
        if not init:
            return epoch

    # Existing experiment at latest epoch. If bool, it will be true.
    elif isinstance(resume, bool):
        init = checkpoints[-1]
        epoch = bs.state.epoch(conf, init)

    # Existing experiment at specific epoch.
    elif isinstance(resume, int):
        init = bs.state.path(conf, resume)
        epoch = resume

    else:
        raise ValueError(f'resume value "{resume}" not of type bool or int')

    state = torch.load(init, weights_only=True)
    for k, v in kwargs.items():
        v.load_state_dict(state[k])

    logger.info('restored %s from %s', tuple(kwargs), init)
    return epoch


def save(conf, /, epoch, force=False, **kwargs):
    """Save a general checkpoint for inference or resuming training.

    Saves the state of objects with a `state_dict` method passed as key-value
    pairs, to a location determined by the configuration. Unless you force it,
    it will only create checkpoints for epochs matching the save frequency.

    Parameters
    ----------
    conf : os.PathLike or dict
        Model configuration.
    epoch : int
        Epoch number.
    force : bool, optional
        Force saving, even if `epoch` does not match the save period.
    **kwargs : dict
        Additional key-value objects to save, with keys for loading.

    """
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f'epoch "{epoch}" is not a non-negative int')

    if not isinstance(conf, dict):
        conf = bs.config.load(conf)

    if epoch % conf['checkpoint']['period'] != 0 and not force:
        return

    # Output path.
    path = bs.state.path(conf, epoch)
    path.parent.mkdir(exist_ok=True)

    state = {k: v.state_dict() for k, v in kwargs.items()}
    state['env'] = bs.config.env(conf)
    torch.save(state, path)
    logger.info('saved %s to "%s"', state.keys(), path)


def path(conf, /, epoch):
    """Construct the checkpoint path from an epoch number.

    Parameters
    ----------
    conf : os.PathLike or dict
        Model configuration.
    epoch : int
        Epoch number.

    Returns
    -------
    pathlib.Path
        Checkpoint path.

    """
    if not isinstance(conf, dict):
        conf = bs.config.load(conf)

    folder = pathlib.Path(conf['checkpoint']['folder'])
    path = conf['checkpoint']['path']
    name = conf['cache']['name']
    return folder / path.format(name=name, epoch=epoch)


def epoch(conf, /, path):
    """Extract the epoch number from a checkpoint path.

    Parameters
    ----------
    conf : os.PathLike or dict
        Model configuration.
    path : os.PathLike
        Checkpoint path.

    Returns
    -------
    int
        Epoch number.

    """
    if not isinstance(conf, dict):
        conf = bs.config.load(conf)

    pattern = conf['checkpoint']['regex']
    return int(re.search(pattern, str(path))['epoch'])


def list(conf, /, epoch=None):
    """List checkpoint paths in ascending order.

    Parameters
    ----------
    conf : os.PathLike or dict
        Model configuration.
    epoch : int or tuple of int, optional
        Find a specific epoch, epochs within an inclusive `(min, max)` range,
        or pass `(min, max, step)` for a minimum step between consecutive
        epochs. None disables min, max, or step filtering.

    Returns
    -------
    list
        Checkpoint paths.

    Raises
    ------
    FileNotFoundError
        If there are no checkpoints within the epoch range.

    """
    if not isinstance(conf, dict):
        conf = bs.config.load(conf)

    name = conf['cache']['name']
    folder = pathlib.Path(conf['checkpoint']['folder'])
    pattern = conf['checkpoint']['glob'].format(name=name)
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f'no checkpoints for "{name}"')

    # Range.
    if epoch is None or isinstance(epoch, int):
        epoch = [epoch]
    if len(epoch) == 1:
        epoch = (*epoch, *epoch)
    if len(epoch) == 2:
        epoch = (*epoch, None)
    start, stop, step = epoch

    # Filtering.
    epochs = [bs.state.epoch(conf, f) for f in files]
    indices = []
    for i, e in enumerate(epochs):
        if start is not None and e < start:
            continue
        if stop is not None and e > stop:
            break
        if step is not None and indices and e - epochs[i - 1] < step:
            continue
        indices.append(i)

    if not indices:
        raise FileNotFoundError(f'no checkpoints in epoch range {epoch}')

    return [files[i] for i in indices]
