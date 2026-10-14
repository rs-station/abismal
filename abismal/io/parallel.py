"""
Parsing in parallel with a pool of processes from the standard library.

Each worker process parses whole files (or groups of CrystFEL chunks) and sends
back numpy arrays. Results come back in task order, each as soon as it and the
tasks before it are done, so the first images reach training while later files
are still being parsed. The pool is sized by the CPUs this process may run on
(`available_cpus`), so a run that Slurm binds to N cores uses N workers.
"""
import multiprocessing
import os
import sys
import types
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager


def available_cpus():
    """
    How many CPUs this process may run on.

    This is the size of the process's CPU affinity mask, the set of cores the
    kernel will schedule it on. Slurm sets the mask when it binds a task (e.g.
    `srun -c 16 --cpu-bind=cores`), as do `taskset` and container CPU limits, so
    it is what the process can actually use rather than the machine's size.
    """
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    n = os.environ.get('SLURM_CPUS_PER_TASK')
    if n:
        return int(n)
    return os.cpu_count() or 1


@contextmanager
def _main_module_hidden():
    """
    Start worker processes without the caller's main script.

    A 'spawn' worker normally re-imports the parent's `__main__` so that
    functions defined there can be unpickled. The workers here only run
    functions from abismal, and re-importing the caller's script is at best
    wasted work: it fails for a script read from stdin, and re-runs an
    unguarded script (no `if __name__ == "__main__":`) in every worker. While
    this is active, `__main__` is a bare module that multiprocessing does not
    try to import.
    """
    main = sys.modules.get('__main__')
    sys.modules['__main__'] = types.ModuleType('__main__')
    try:
        yield
    finally:
        if main is None:
            sys.modules.pop('__main__', None)
        else:
            sys.modules['__main__'] = main


def _init_worker():
    # Parsing needs no GPU and no threads beyond its own: hide the GPU, and keep
    # numpy's BLAS to one thread per worker so a pool of N workers uses N cores.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = '1'


def ordered_map(fn, tasks, num_workers=None, window=None):
    """
    Yield `fn(*task)` for every task, in task order.

    With more than one worker and more than one task the calls run in a pool
    of `num_workers` processes (`available_cpus()` by default, and never more
    than there are tasks); otherwise they run here, one after another. At most
    `window` calls (twice the number of workers by default) are outstanding at
    a time, so a fast pool does not run far ahead of a slow consumer and fill
    memory with parsed data nobody has asked for yet.

    `fn` and the tasks must be picklable, and `fn` importable from a module
    (not defined in the caller's script). Workers are started with 'spawn', not
    'fork': the caller may already hold CUDA and threads (parsing runs alongside
    training), and forking such a process is unsafe.
    """
    tasks = list(tasks)
    if num_workers is None:
        num_workers = available_cpus()
    num_workers = max(1, min(int(num_workers), len(tasks)))
    if num_workers == 1:
        for task in tasks:
            yield fn(*task)
        return

    window = max(num_workers, window or 2 * num_workers)
    pool = ProcessPoolExecutor(
        num_workers, mp_context=multiprocessing.get_context('spawn'), initializer=_init_worker,
    )
    try:
        pending = deque()
        remaining = iter(tasks)
        # The pool starts a worker per submission until it has `num_workers`;
        # the first `window >= num_workers` submissions start them all.
        with _main_module_hidden():
            for task in remaining:
                pending.append(pool.submit(fn, *task))
                if len(pending) >= window:
                    break
        while pending:
            result = pending.popleft().result()
            for task in remaining:
                pending.append(pool.submit(fn, *task))
                break
            yield result
    finally:
        # Also reached when the consumer stops early: drop what has not started.
        pool.shutdown(wait=True, cancel_futures=True)
