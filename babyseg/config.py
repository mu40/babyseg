"""Configuration utilities. Importable without PyTorch installed."""

import babyseg
import functools
import importlib
import json
import logging
import os
import pathlib
import socket
import subprocess
import sys


logger = logging.getLogger(__name__)
_cache = {}
DEFAULTS = 'config/defaults.json'


def load(*files):
    """Successively load settings from JSON files.

    The function loads files successively, overriding prior entries unless they
    are dictionaries, which it will merge. However, if the dictionary in the
    current file includes the key `clear`, and its value is True, then the new
    dictionary will replace the existing entry. The function always parses
    defaults from `DEFAULTS` first, if truthy.

    Parameters
    ----------
    *files : os.PathLike, optional
        JSON files, each containing a `dict`. Will run through `qualify_path`.

    Returns
    -------
    dict
        Settings.

    """
    def read(path):
        with open(path) as f:
            d = json.load(f)
        if isinstance(d, dict):
            return d
        raise ValueError(f'file "{path}" does not specify a dict')

    def merge(old, new):
        for k, v in new.items():
            if any(not isinstance(x, dict) for x in (old.get(k), v)):
                old[k] = v
            elif v.pop('clear', False):
                old[k] = v
                logger.info('cleared configuration entry "%s"', k)
            else:
                old[k] = merge(old[k], v)

        return old

    out = {}
    if DEFAULTS:
        logger.debug('loading default configuration from "%s"', DEFAULTS)
        out = read(qualify_path(DEFAULTS))

    names = []
    for f in map(qualify_path, files):
        names.append(f.stem)
        out = merge(out, read(f))

    # Track files read.
    if names:
        out.setdefault('cache', {})
        out['cache'].setdefault('name', names[0])
        out['cache']['names'] = names
        out['cache']['files'] = tuple(map(str, files))
        logger.info('loaded configuration from %s', files)

    return out


def build(f, /, *args, instance=True, **kwargs):
    """Build a class instance or partial function.

    Builds a class instance or partial function from a callable, a qualifying
    string like 'module.submodule.entity', or a dictionary containing that
    string under the 'name' key and, optionally, arguments 'args' and options
    'kwargs'. Any `args` and `kwargs` arguments will override these options.

    Parameters
    ----------
    f : str or dict or callable
        Class or function.
    *args : tuple, optional
        Positional arguments to forward to the callable.
    instance : bool, optional
        Return a class instance rather than a partial constructor.
    **kwargs : dict, optional
        Key-value settings to forward, overriding 'kwargs' in `f`.

    Returns
    -------
    object
        Class instance or function.

    Examples
    --------
    If `f` is the string 'torch.arange', expect `functools.partial(f, *args,
    **kwargs)`. If the string points to a class 'torch.nn.ReLU', expect
    instance `torch.nn.ReLU(*args, **kwargs)`, unless `instance` is False.

    For arguments `(f, d, eps=1)`, if f is `{name='torch.optim.Adam',
    kwargs={'lr': 2}}`, expect instance: `torch.optim.Adam(d, eps=1, lr=2)`.

    """
    ar = []
    kw = {}
    if isinstance(f, str):
        f = dict(name=f)

    if isinstance(f, dict):
        # Entity and module name.
        name = f['name']
        *module, name = name.split('.')
        module = '.'.join(module)
        logger.debug('parsed package "%s", module "%s"', name, module)
        if not module:
            raise ValueError(f'entity name "{name}" does not specify a module')

        # Import.
        ar = f.get('args', [])
        kw = f.get('kwargs', {})
        module = importlib.import_module(module)
        f = getattr(module, name)
        logger.debug('extracted %s from %s', f, module)
        if not callable(f):
            raise TypeError(f'entity {f} at "{module}.{name}" is not callable')

    # Arguments.
    logger.debug('received configuration arguments %s and %s', ar, kw)
    logger.debug('received function arguments %s and %s', args, kwargs)
    ar.extend(args)
    kw.update(kwargs)

    # Class instance.
    logger.debug('building callable with merged arguments %s and %s', ar, kw)
    if type(f) is type and instance:
        return f(*ar, **kw)

    # No need for `partial` if no arguments.
    return functools.partial(f, *ar, **kw) if ar or kw else f


def load_model(
    config,
    /,
    init='latest',
    *,
    key='model',
    device=None,
    **kwargs,
):
    """Conveniently build and initialize a model from a configuration.

    Parameters
    ----------
    config : os.PathLike or dict
        Model configuration.
    init : 'latest' or int or os.PathLike or None, optional
        Load the latest, a specific epoch, a path, or no checkpoint.
    key : str, optional
        Key pointing to the correct configuration and checkpoint entry.
    device : torch.device, optional
        Configure the model on a specific device.
    **kwargs : dict, optional
        Model options overriding the configuration.

    Returns
    -------
    torch.nn.Module
        Model.

    """
    if not isinstance(config, dict):
        config = load(config)

    # Weights. Try the whole returned object if the key does not exist.
    model = build(config[key], **kwargs).to(device)
    if init is not None:
        import torch

        # Path.
        try:
            path = babyseg.state.path(config, int(init))
        except (ValueError, TypeError):
            path = babyseg.state.list(config)[-1] if init == 'latest' else init

        # Avoid error if CUDA unavailable.
        device = next(model.parameters()).device
        state = torch.load(path, weights_only=True, map_location=device)
        model.load_state_dict(state.get(key, state))
        logger.info('loaded checkpoint "%s"', path)

    return model


def argparse(config, /, *options):
    """Parse options from the command line and add them into a configuration.

    Parses strings of format 'key[:sub[:...]]=value', adding them into a
    configuration. The value will have the first type it matches or casts to,
    in order: 'true' -> `True`, 'false' -> `False`, `int`, `float`, `str`.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    *options : sequence of str
        Option strings to add.

    """
    def cast(x):
        if x == 'true':
            return True
        if x == 'false':
            return False

        for t in (int, float, str):
            try:
                return t(x)
            except ValueError:
                pass

    for option in options:
        logger.debug('parsing option %s', option)
        try:
            keys, value = option.split('=')
            *keys, last = keys.split(':')
        except ValueError as e:
            raise ValueError(f'option "{option}" has invalid format') from e

        # Descend or error out.
        sub = config
        for k in keys:
            sub = sub[k]
        sub[last] = cast(value)
        logger.info('added configuration option "%s"', option)


def device(device):
    """Select a device and fail if it is unavailable.

    Parameters
    ----------
    device : str or torch.device
        Device.

    Returns
    -------
    torch.device
        Selected device.

    """
    import torch

    logger.info('selecting device "%s"', device)
    device = torch.device(device)

    logger.info('testing device availability')
    torch.tensor(1, device=device)

    logger.info('confirmed availability of %s', device)
    return device


def qualify_path(path):
    """Interpret paths relative to `BABYSEG_HOME` if needed for portability.

    Prepends the directory set in environment variable `BABYSEG_HOME` to
    relative paths that do not exist. Other paths will be returned unchanged.

    Parameters
    ----------
    path : os.PathLike
        Path to interpret.

    Returns
    -------
    pathlib.Path
        Path, possibly prefixed with `BABYSEG_HOME`.

    """
    p = pathlib.Path(path)
    if p.is_absolute():
        logger.debug('not qualifying absolute path "%s"', p)
        return p

    if p.exists():
        logger.debug('not qualifying existing relative path "%s"', p)
        return p

    logger.info('encountered nonexistent relative path "%s"', p)
    home = os.getenv('BABYSEG_HOME')
    if home:
        logger.info('found environment variable BABYSEG_HOME="%s"', home)
        logger.info('qualifying path by prepending BABYSEG_HOME to "%s"', p)
        return home / p

    logger.warning('not qualifying path because BABYSEG_HOME unset')
    return p


def git_status(path, *pathspec, timeout=10):
    """Fetch the working tree status of a Git repository.

    Parameters
    ----------
    path : os.PathLike
        Git repository path.
    *pathspec : str, optional
        Limit the scope of `git status`.
    timeout : int, optional,
        Timeout in seconds.

    Returns
    -------
    str
        Output of `git status --porcelain`.

    """
    p = subprocess.run(
        ('git', 'status', '--porcelain', '--', *pathspec),
        capture_output=True,
        cwd=pathlib.Path(path).expanduser(),
        timeout=timeout,
        text=True,
    )

    # Print error, which `CalledProcessError` from `check=True` does not.
    if p.returncode:
        raise ValueError(p.stderr)

    return p.stdout.strip()


def git_hash(path, *pathspec, short=False, timeout=10):
    """Fetch the currently checked-out commit hash in a Git repository.

    Parameters
    ----------
    path : os.PathLike
        Git repository path.
    *pathspec : str, optional
        Limit the scope `git status`.
    short : bool, optional
        Shorten the commit hash.
    timeout : int, optional,
        Timeout in seconds.

    Returns
    -------
    str
        Commit hash.

    Raises
    ------
    ValueError
        If `git status` reports uncommitted change.

    """
    # Working tree status.
    path = pathlib.Path(path).expanduser()
    if git_status(path, *pathspec, timeout=timeout):
        raise ValueError(f'uncommitted change in repository "{path}"')

    # Checked-out commit.
    hash = ('git', 'rev-parse', 'HEAD')
    if short:
        hash = (*hash[:-1], '--short', *hash[-1:])

    return subprocess.run(
        hash,
        capture_output=True,
        cwd=path,
        timeout=timeout,
        text=True,
        check=True,
    ).stdout.strip()


def env(config, /, test=False):
    """Return cached information about execution environment.

    Parameters
    ----------
    config : os.PathLike or dict
        Model configuration.
    test : bool, optional
        Do not fail on uncommitted change.

    Returns
    -------
    dict
        Information about environment.

    """
    if 'env' in _cache:
        return _cache['env']

    # Date, hostname, uptime.
    out = {
        'sys.version': sys.version,
        'sys.argv': sys.argv,
        'sys.path': sys.path,
        'hostname': socket.gethostname(),
    }
    for c in ('date', 'uptime'):
        out[c] = subprocess.check_output(c, text=True).strip()

    # Git commit hashes.
    out['git-rev-parse'] = {}
    if not isinstance(config, dict):
        config = load(config)

    try:
        for f, stat in config['repositories'].items():
            out['git-rev-parse'][f] = git_hash(f, *stat)
    except ValueError:
        if not test:
            raise

    # Environment variables.
    out['os.environ'] = dict(os.environ)

    # Packages.
    dist = importlib.metadata.distributions()
    out['packages'] = {}
    for d in sorted(dist, key=lambda d: d.metadata['name']):
        out['packages'][d.metadata['name']] = d.version

    _cache['env'] = out
    return out
