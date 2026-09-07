#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flexible command-line test launcher.

Usage:
    python run_test.py --model fipl_da --dataset fl_pacs --communication_epoch 50 \
        --local_epoch 1 --online_ratio 1.0 --eps 0.0 --device_id 0

Any unspecified arguments use the defaults in best_args.py.
"""

import sys
from main import parse_args, main

if __name__ == '__main__':
    if len(sys.argv) > 1:
        cmd = ' '.join(sys.argv[1:])
        args = parse_args(cmd)
    else:
        args = parse_args()
    main(args)
