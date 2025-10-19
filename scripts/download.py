#!/usr/bin/env python3
"""Download BabySeg model checkpoints from the internet."""

import argparse
import logging
import pathlib
import re
import urllib.parse
import urllib.request


# Keep the trailing slash for `urljoin`.
REMOTE = 'https://surfer.nmr.mgh.harvard.edu/docs/babyseg/'


# Logging.
logger = logging.getLogger(__name__)


def main(argv=None):
    """Entry point for command-line execution.

    Parameters
    ----------
    argv : tuple of str, optional
        Command-line arguments. If None, defaults to `sys.argv[1:]`.

    """
    # Arguments.
    # ruff: noqa: E501
    f = argparse.ArgumentDefaultsHelpFormatter
    p = argparse.ArgumentParser(formatter_class=f, description=__doc__)
    p.add_argument('-o', dest='out_dir', default='checkpoints', help='output directory')
    p.add_argument('-f', dest='force', action='store_true', help='override')
    arg = p.parse_args()
    # ruff: enable: E501

    # File list.
    logging.info('retrieving checkpoint list "%s"', REMOTE)
    with urllib.request.urlopen(REMOTE) as f:
        charset = f.headers.get_content_charset() or 'utf-8'
        website = f.read().decode(charset)

    # Checkpoint files.
    logging.debug('parsing checkpoint list')
    pattern = r'href\s*=\s*["\']?([^"\' >]+\.pt)(?=["\' >])'
    pattern = re.compile(pattern, re.IGNORECASE)
    files = set()
    for f in pattern.findall(website):
        f = urllib.parse.urljoin(REMOTE, f)
        if f not in files:
            files.add(f)
            logging.info('found checkpoint "%s"', f)

    if not files:
        logging.error('no checkpoints founds')
        exit(1)

    # Download.
    out_dir = pathlib.Path(arg.out_dir)
    out_dir.mkdir(exist_ok=True)
    for i, f in enumerate(files, start=1):
        logging.info('processing checkpoint %d of %d', i, len(files))
        out = out_dir / pathlib.Path(urllib.parse.urlparse(f).path).name

        if out.exists():
            do = 'overriding' if arg.force else 'skipping'
            logging.info('%s checkpoint existing at "%s"', do, out)
            if not arg.force:
                continue

        logging.info('downloading "%s"', f)
        urllib.request.urlretrieve(f, out)
        logging.info('saved checkpoint to "%s"', out)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level='INFO')
    main()
