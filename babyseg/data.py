"""Data abstraction utilities. Importable without PyTorch installed."""

import concurrent.futures
import glob
import json
import logging
import os
import pathlib
import time


logger = logging.getLogger(__name__)


def slurm_map(work, samples, *args, **kwargs):
    """Shard data and map callable for a current worker.

    Reads the current worker ID and total number of workers as environment
    variables SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT, respectively.
    If unset, the function maps the callable to all data.

    Parameters
    ----------
    work : callable
        Callable taking a `Sample` as its first argument.
    samples : sequence of Sample
        Samples to process.
    *args : tuple, optional
        Passed to `work`.
    **kwargs : dict, optional
        Passed to `work`.

    Returns
    -------
    list
        Values returned by the callable for the shard.

    Raises
    ------
    ValueError
        If SLURM_ARRAY_TASK_MIN or SLURM_ARRAY_TASK_STEP is set but not 1.

    """
    # Environment.
    for f in ('SLURM_ARRAY_TASK_MIN', 'SLURM_ARRAY_TASK_STEP'):
        if 1 != int(os.getenv(f, 1)):
            raise ValueError(f'environment variable "{f}" is not 1')

    worker_id = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))
    worker_num = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))
    logger.info('identifying as worker %d of %d', worker_id, worker_num)

    # Shard.
    def keep(ind):
        return ind % worker_num == worker_id - 1

    samples = list(samples)
    samples = [d for i, d in enumerate(samples) if keep(i)]
    logger.info('retaining %d samples for worker %d', len(samples), worker_id)

    # Mapping.
    results = []
    for i, sample in enumerate(samples, start=1):
        start = time.time()
        results.append(work(sample, *args, **kwargs))
        t = time.time() - start
        logger.info('finished sample %d of %d in %.1f sec', i, len(samples), t)

    return results


def read_samples(files, which=None):
    """Read data samples from dataset split files.

    Parameters
    ----------
    files : os.PathLike
        JSON files.
    which : str or sequence of str
        Only return samples that have specific image types.

    Returns
    -------
    list
        List of `Sample` objects knowing data paths.

    """
    if isinstance(files, (str, os.PathLike)):
        files = [files]

    if isinstance(which, str):
        which = [which]

    def read(path):
        with open(path) as f:
            return json.load(f)

    samples = [Sample(d) for f in files for d in read(f)]
    logger.info('found %d samples in files %s', len(samples), files)

    if which is not None:
        samples = [f for f in samples if all(f.has(i) for i in which)]
        logger.info('kept %d samples that have %s', len(samples), which)

    return samples


class Sample:
    """A sample from a dataset.

    This class should be the only piece of code knowing the dataset structure.
    Ideally, we would pass around generic data structures like a `dict` instead
    of a class. However, the class allows us to add baselines, label sets, or
    image contrasts without changing code.

    """

    def __init__(self, data, /):
        """Initialize the sample.

        Parameters
        ----------
        data : dict
            Key-value pairs from JSON file.

        """
        self.data = data
        self.folder = pathlib.Path(data['folder'])

    def has(self, name):
        """Check if the sample has a specific image.

        Parameters
        ----------
        name : str
            Image to check.

        Returns
        -------
        bool
            Whether the sample has the image.

        """
        return name in self.data

    def image(self, name):
        """Image.

        Parameters
        ----------
        name : str
            Image type. Must match baseline inputs in JSON files and image
            types passed to to `torch.utils.data.Dataset` for validation.

        Returns
        -------
        pathlib.Path
            Image path.

        """
        return self.folder / self.data[name]

    @property
    def image_types(self):
        """Image types.

        Returns
        -------
        tuple of str
            Image types whose paths are accessible via `image`.

        """
        return tuple(self.data['images'])

    def label_map(self, labels='baby'):
        """Label map.

        Parameters
        ----------
        labels : str, optional
            Label set name. Must match keys of validation labels and baseline
            mappings in JSON files.

        Returns
        -------
        pathlib.Path
            Label map path.

        """
        if labels == 'baby':
            return self.folder / self.data['labels']

        return self.folder / f'labels.{labels}.nii.gz'

    @property
    def name(self):
        """Sample name.

        Returns
        -------
        str
            Sample identifier.

        """
        try:
            return self.data['name']
        except KeyError:
            return self.folder.name

    @property
    def age(self):
        """Age.

        Returns
        -------
        float
            Age of the individual.

        """
        return self.data['age']

    @property
    def unit(self):
        """Unit.

        Returns
        -------
        str
            Unit of the age.

        """
        return self.data['unit']

    def output(self, method, labels='baby'):
        """Prediction.

        Parameters
        ----------
        method : str
            Method name.
        labels : str, optional
            Label set name. Must match keys of validation labels and baseline
            mappings in JSON files.

        Returns
        -------
        pathlib.Path
            Path to save method output.

        """
        d = self.folder / method
        d.mkdir(exist_ok=True)
        return d / f'{labels}.nii.gz'

    def purge(self, method):
        """Remove method artifacts.

        Parameters
        ----------
        method : str
            Method name.

        """
        if not method:
            raise ValueError('method name is empty')

        d = self.folder / method
        files = d.glob('*')
        for f in files:
            f.unlink()

        try:
            d.rmdir()
        except (OSError, FileNotFoundError):
            pass

        logger.info('purged %s artifacts for sample "%s"', method, self)

    def __repr__(self):
        """Return string representation of the sample."""
        return str(self.name)


def load_label_maps(*pattern, threads=None):
    """Load label maps and determine unique labels.

    Returns a list of unique labels and the label maps. The label maps have to
    be of integer type and identical shape.

    Parameters
    ----------
    *pattern : os.PathLike
        Glob pattern matching label maps.
    threads : int, optional
        Maximum number of workers.

    Returns
    -------
    torch.Tensor
        Unique labels.
    tuple of torch.Tensor
        Label maps.

    Raises
    ------
    FileNotFoundError
        If the pattern yields no label maps.

    """
    import nibabel as nib
    import torch

    if isinstance(pattern, (str, os.PathLike)):
        pattern = [pattern]

    # Processing.
    def load(path):
        x = nib.load(path).get_fdata()
        x = torch.as_tensor(x, dtype=torch.float32)
        return x.to(torch.int64).unique(), x.squeeze()

    pattern = tuple(map(str, pattern))
    with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        files = pool.map(glob.glob, pattern)
        files = sorted(f for sub in files for f in sub)
        uniq, maps = zip(*pool.map(load, files), strict=True)

    # Validation.
    if not files:
        raise FileNotFoundError(f'pattern "{pattern}" yields no label maps')

    return torch.cat(uniq).unique(), maps


def save_split(df, /, root, contrast, labels, path, complete=True):
    """Save rows of a Polars data frame as a JSON file.

    Parameters
    ----------
    df : polars.DataFrame
        Data frame. Columns: sample, contrast, out_image, `labels`, age, unit.
    root : os.PathLike
        Root data directory.
    contrast : str or sequence of str
        Image type.
    labels : str or None
        Filename of label map. None means data-frame column 'out_labels'.
    path : os.PathLike
        Output JSON file.
    complete : bool, optional
        Only save samples that have all requested contrasts.

    Raises
    ------
    ValueError
        If there are no rows with matching contrast.
    FileNotFoundError
        If specified images or label maps do not exist.

    """
    import katy
    import polars as pl
    if isinstance(contrast, str):
        contrast = [contrast]

    # Retain only the samples that have each requested contrast.
    if complete:
        keep = [df.filter(pl.col('contrast') == c)['sample'] for c in contrast]
        keep = set.intersection(*map(set, keep))
        df = df.filter(pl.col('sample').is_in(keep))

    df = df.filter(pl.col('contrast').is_in(contrast))
    if df.is_empty():
        ValueError(f'no rows with contrast in {contrast}')

    out = {}
    for row in df.rows(named=True):
        # Paths.
        sample =  row['sample']
        folder = pathlib.Path(root) / sample
        image = folder / row['out_image']
        annot = folder / (labels if labels else row['out_labels'])
        for f in (image, annot):
            if not f.exists:
                raise FileNotFoundError(f'file {f} not found')

        # Output dictionary.
        if sample not in out:
            out[sample] = {
                'folder': str(folder),
                'labels': annot.name,
                'images': [row['contrast']],
                'age': round(row['age'], ndigits=2),
                'unit': row['unit'],
            }

        else:
            out[sample]['images'].append(row['contrast'])

        out[sample][row['contrast']] = image.name

    # Save as a list of dictionaries.
    out = list(out.values())
    katy.io.save(out, path)
