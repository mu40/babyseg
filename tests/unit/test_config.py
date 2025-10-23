"""Test configuration module."""

import babyseg
import json
import katy
import pathlib
import pytest


def test_load_default():
    """Test loading configuration defaults."""
    y = katy.io.load(babyseg.config.DEFAULTS)
    assert babyseg.config.load() == y


def test_load_default_change(monkeypatch, tmp_path):
    """Test reading defaults from a changed path."""
    path = tmp_path / 'new.json'
    data = {'a': 'b'}
    katy.io.save(data, path)

    # Expect defaults to be data above.
    monkeypatch.setattr('babyseg.config.DEFAULTS', path)
    assert babyseg.config.load() == data


def test_load_files_merge(tmp_path):
    """Test loading settings from several files."""
    f_1 = tmp_path / '1.json'
    f_2 = tmp_path / '2.json'

    data_1 = {'a': 1}
    data_2 = {'b': 2}
    katy.io.save(data_1, f_1)
    katy.io.save(data_2, f_2)

    c = babyseg.config.load(f_1, f_2)
    assert c['a'] == data_1['a']
    assert c['b'] == data_2['b']


def test_load_files_replace(tmp_path):
    """Test replacement of overlapping settings."""
    f_1 = tmp_path / '1.json'
    f_2 = tmp_path / '2.json'

    data_1 = {'a': 1}
    data_2 = {'a': 2}
    katy.io.save(data_1, f_1)
    katy.io.save(data_2, f_2)

    # Expect second file to override first.
    c = babyseg.config.load(f_1, f_2)
    assert c['a'] == data_2['a']


def test_qualify_path_type():
    """Test the return type when qualifying a path."""
    for path in (pathlib.Path('x'), pathlib.Path('/usr'), 'a/b', '/a/b'):
        out = babyseg.config.qualify_path(path)
        assert isinstance(out, pathlib.Path)


def test_qualify_path_absolute():
    """Test if qualifying an absolute path doesn't change it."""
    f = pathlib.Path('/one/two/three')
    assert babyseg.config.qualify_path(f) == f


def test_qualify_path_relative_existing(monkeypatch, tmp_path):
    """Test if qualifying an existing relative path doesn't change it."""
    monkeypatch.chdir(tmp_path.parent)
    f = pathlib.Path(tmp_path.name)
    assert babyseg.config.qualify_path(f) == f


def test_qualify_path_relative_missing(monkeypatch, tmp_path):
    """Test if qualifying a nonexistent relative path."""
    f = pathlib.Path('some_file')
    monkeypatch.chdir(tmp_path)

    # Expect no change if `BABYSEG_HOME` unset.
    monkeypatch.delenv('BABYSEG_HOME', raising=False)
    assert babyseg.config.qualify_path(f) == f

    # Expect prefix `BABYSEG_HOME` if set.
    home = '/some/home'
    monkeypatch.setenv('BABYSEG_HOME', home)
    assert babyseg.config.qualify_path(f) == home / f


def test_load_invalid(tmp_path):
    """Test if loading non-dictionary settings raise an error."""
    f = tmp_path / 'string.json'
    katy.io.save('this is a string', f)
    with pytest.raises(ValueError):
        babyseg.config.load(f)


def test_build_args():
    """Test building a partial function with positional arguments."""
    f = babyseg.config.build(lambda *args: args, 1, 2)
    assert f(3) == (1, 2, 3)


def test_build_kwargs():
    """Test building a partial function with keyword arguments."""
    f = babyseg.config.build(lambda **kwargs: kwargs, one=1, two=2)
    assert f() == dict(one=1, two=2)


def test_build_function():
    """Test building a function from a string."""
    f = babyseg.config.build('json.loads')
    assert f is json.loads
    assert callable(f)


def test_build_class():
    """Test building a class from a string."""
    f = babyseg.config.build('json.JSONDecodeError', instance=False)
    assert isinstance(f, type)
    assert f is json.JSONDecodeError


def test_build_object():
    """Test building a object from a string."""
    f = babyseg.config.build('json.JSONDecodeError', 'hi', doc='doc', pos=0)
    assert isinstance(f, json.JSONDecodeError)
