#!/usr/bin/env -S python3 -u
"""BabySeg: brain segmentation across the first years of life."""

import argparse
import babyseg
import logging
import os
import pathlib
import sys


logger = logging.getLogger(__name__)


def main(argv=None):
    """Entry point for command-line execution.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. If None, defaults to `sys.argv[1:]`.

    """
    # Environment.
    home = os.getenv('BABYSEG_HOME')
    if not home:
        print('ERROR: environment variable BABYSEG_HOME is unset')
        exit(1)

    # Defaults.
    home = pathlib.Path(home)
    config = home / 'config'
    babyseg.config.DEFAULT = config / 'defaults.json'
    d = dict(
        c=sorted(config.glob('babyseg.*.json'))[-1],
        k=sorted((home / 'checkpoints').glob('babyseg.*.pt'))[-1],
        g=argparse.SUPPRESS,
        j=argparse.SUPPRESS,
        v=argparse.SUPPRESS,
    )

    # Arguments.
    p = argparse.ArgumentParser(
        prog='babyseg',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='Website: https://w3id.org/babyseg',
        description='''
            BabySeg is a tool for brain segmentation across the first years of
            life, without preprocessing. It can integrate information from
            multiple image volumes of variable size, resolution, and contrast
            in any order, provided that (1) their header geometries are
            correct, and (2) they are properly aligned in world space.
        ''',
    )

    # ruff: noqa: E501
    p.add_argument('images', metavar='image', nargs='+', help='input images')
    p.add_argument('-c', dest='config', default=d['c'], help='model JSON file')
    p.add_argument('-k', dest='checkpoint', default=d['k'], help='model weights')
    p.add_argument('-g', dest='gpu', default=d['g'], action='store_true', help='enable GPU acceleration')
    p.add_argument('-o', dest='out_seg', help='output label map')
    p.add_argument('-p', dest='out_prob', help='output probability maps')
    p.add_argument('-i', dest='out_lead', help='output conformed lead image')
    p.add_argument('-j', dest='threads', default=d['j'], type=int, help='CPU threads (default: 1/core)')
    p.add_argument('-v', dest='verbose', default=d['v'], action='count', help='repeat to increase verbosity')
    # ruff: enable: E501

    if len(sys.argv) == 1:
        p.print_usage()
        exit(0)

    # Device.
    arg = vars(p.parse_args(argv))
    if arg.pop('gpu', False):
        arg['device'] = 'cuda'

    # Verbosity.
    v = {0: 'WARNING', 1: 'INFO'}.get(arg.pop('verbose', 0), 'DEBUG')
    logging.basicConfig(format='%(levelname)s: %(message)s', level=v)
    babyseg.eval.segment(**arg)


if __name__ == '__main__':
    main()
