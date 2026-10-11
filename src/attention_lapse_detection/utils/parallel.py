import os
import sys

# Set environment variables to limit the number of threads used by numeric backends.
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

    # On Linux, we count only the CPUs available to this process.
    if sys.platform == "linux":
        return len(os.sched_getaffinity(0))

    return os.cpu_count() or 4


def default_max_workers(to_reserve=2):
    return max(1, available_cpus() - to_reserve)
