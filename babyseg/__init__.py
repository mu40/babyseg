"""BabySeg: brain segmentation across the first years of life."""

import importlib


__all__ = (
    'config',
    'data',
    'eval',
    'nn',
    'state',
)


def __getattr__(name):
    """Lazily import submodules on first access.

    Import lazily, to be able to use submodules that do not depend on PyTorch
    in baseline containers that do not have PyTorch installed and to not waste
    time when just parsing arguments or printing help text in scripts.

    Parameters
    ----------
    name : str
        Submodule name.

    Returns
    -------
    module
        Imported submodule.

    """
    if name in __all__:
        module = importlib.import_module(f'{__name__}.{name}')
        globals()[name] = module
        return module

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
