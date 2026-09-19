"""Importing this module pins every numeric backend to one thread, and must happen
before numpy and mediapipe load, because each reads these variables once at
import. Otherwise every worker sizes its own thread pool to the whole machine
and N workers x N threads oversubscribes the cores.
"""

import os
import sys

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

for var in THREAD_ENV_VARS:
    os.environ.setdefault(var, "1")


def available_cpus():

    if sys.platform == "linux":
        return len(os.sched_getaffinity(0))

    return os.cpu_count() or 4


def default_max_workers(to_reserve=2):
    return max(1, available_cpus() - to_reserve)
