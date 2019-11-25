# encoding: utf8

import os
import sys
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.basename(__file__)))

sys.path.insert(0, BASE_DIR)

logging.basicConfig(format=logging.BASIC_FORMAT, level=logging.INFO)
