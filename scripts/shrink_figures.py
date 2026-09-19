#!/usr/bin/env python3
# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
"""Quantise the report figures to 256-colour palette PNGs (lossless for line plots, ~3x smaller).

Usage: python3 scripts/shrink_figures.py [report/figures]
Called by make_figures.py / make_results_figures.py after every savefig; can also be run standalone.
"""
import os, sys, glob
from PIL import Image


def shrink(path):
    im = Image.open(path)
    if im.mode == 'P':
        return
    q = im.convert('RGB').quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q.save(path, format='PNG', optimize=True)


if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else 'report/figures'
    files = sorted(glob.glob(os.path.join(d, '*.png')))
    before = sum(os.path.getsize(f) for f in files)
    for f in files:
        shrink(f)
    after = sum(os.path.getsize(f) for f in files)
    print(f'{len(files)} figures: {before/1e6:.1f} MB -> {after/1e6:.1f} MB')
