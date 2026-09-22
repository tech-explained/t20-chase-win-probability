#!/usr/bin/env python3
"""Train the cricket v1 engines, evaluate, and run the graduation gate."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rater.engines.cricket.train import main

if __name__ == "__main__":
    main()
