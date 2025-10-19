"""Test configuration module."""

import babyseg
import json
import katy
import pytest


def test_load_default():
    """Test loading configuration defaults."""
    y = katy.io.load(babyseg.config.DEFAULT)
    assert babyseg.config.load() == y


def test_load_default_change(tmp_path):
    """Test changing the default configuration path."""
    # Paths.
    old = babyseg.config.DEFAULT
    new = tmp_path / 'new.json'

    # Data.
    data = katy.io.load(old)
    katy.io.save(data, new)

    # Read from function argument.
    assert babyseg.config.load(defaults=new) == data

    # Read from changed module attribute.
    try:
        babyseg.config.DEFAULT = new
        assert babyseg.config.load() == data

    finally:
        babyseg.config.DEFAULT = old


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
    """Test building a partial function with keyword arguments.."""
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
