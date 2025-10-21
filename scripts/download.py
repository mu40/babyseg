#!/usr/bin/env python3
"""Download BabySeg checkpoints and test data."""

import logging
import pathlib
import re
import urllib.parse
import urllib.request


# Keep trailing slash for `urljoin`.
REMOTE = 'https://surfer.nmr.mgh.harvard.edu/docs/babyseg/'
output = {'.pt': 'checkpoints', '.nii.gz': 'data'}
logger = logging.getLogger(__name__)


def main():
    """Download BabySeg files."""
    logging.info('retrieving file list from "%s"', REMOTE)
    with urllib.request.urlopen(REMOTE) as r:
        site = r.read().decode(encoding='utf-8')

    regex = r'href\s*=\s*["\']?([^"\' >]+\.(?:pt|nii.gz))(?=["\' >])'
    files = re.findall(regex, site, flags=re.IGNORECASE)
    files = sorted({urllib.parse.urljoin(REMOTE, f) for f in files})
    if not files:
        logging.error('cannot find any files to download')
        exit(1)

    # Downloads.
    for i, f in enumerate(files, start=1):
        logging.info('processing file %d of %d at "%s"', i, len(files), f)

        name = pathlib.Path(urllib.parse.urlparse(f).path).name
        for ext, d in output.items():
            if name.endswith(ext):
                out = pathlib.Path(d) / name
                out.parent.mkdir(exist_ok=True)
                break

        else:
            logging.error('unexpected file suffix for "%s"', f)
            exit(1)

        if out.exists():
            logging.info('skipping file existing at "%s"', out)
            continue

        urllib.request.urlretrieve(f, out)
        logging.info('saved file to "%s"', out)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level='INFO')
    main()
