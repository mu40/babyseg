"""Test Docker units."""

import argparse
import docker.entrypoint
import pathlib
import pytest


@pytest.mark.parametrize('path', ('in.nii', 'in.nii.gz'))
def test_nifti_valid(path):
    """Test if valid NIfTI paths return successfully."""
    assert docker.entrypoint.nifti(path) == pathlib.Path(path)


@pytest.mark.parametrize('path', ('in.mgz', 'in.txt'))
def test_nifti_error(path):
    """Test if non-NIfTI paths raise an ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError):
        docker.entrypoint.nifti(path)


def test_nifti_pathlike():
    """Test if path-like NIfTI paths remain unchanged."""
    f = pathlib.Path('/root/x.nii')
    assert docker.entrypoint.nifti(f) == f
