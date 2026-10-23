#!/usr/bin/env python3
"""Unified 5-Benchmark Runner CLI Script.

Supports:
    python benchmarks/run_benchmarks.py --benchmark {1,2,3,4,5,all} [--quick] [--plot] [--output-dir reports/]

Delegates to `mcmc_robust.bench.__main__`.
"""

from __future__ import annotations

import sys

from mcmc_robust.bench.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
