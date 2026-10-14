"""Parallel parsing with a standard-library process pool (abismal.io.parallel)."""
import os

import numpy as np
import pytest

from abismal.io import parallel
from abismal.io.parallel import available_cpus, ordered_map


pytestmark = pytest.mark.timeout(300)


def square(x):
    return x * x


def fail_on(x, bad):
    if x == bad:
        raise ValueError(f"bad task {x}")
    return x


def worker_environment(_):
    return os.environ.get("CUDA_VISIBLE_DEVICES"), os.environ.get("OMP_NUM_THREADS")


def test_results_come_back_in_task_order():
    tasks = [(i,) for i in range(40)]
    assert list(ordered_map(square, tasks, num_workers=4, window=5)) == [i * i for i in range(40)]


def test_one_worker_or_one_task_runs_in_this_process(monkeypatch):
    def no_pool(*args, **kwargs):
        raise AssertionError("a pool was started")
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", no_pool)
    assert list(ordered_map(square, [(3,), (4,)], num_workers=1)) == [9, 16]
    assert list(ordered_map(square, [(5,)], num_workers=8)) == [25]
    assert list(ordered_map(square, [], num_workers=8)) == []


def test_pool_size_follows_the_affinity_mask_and_the_task_count(monkeypatch):
    sizes = []

    class Recorder:
        def __init__(self, max_workers, **kwargs):
            sizes.append(max_workers)
        def submit(self, fn, *args):
            from concurrent.futures import Future
            f = Future()
            f.set_result(fn(*args))
            return f
        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", Recorder)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    assert available_cpus() == 3
    list(ordered_map(square, [(i,) for i in range(10)]))          # default: the mask
    list(ordered_map(square, [(i,) for i in range(2)]))           # never more than tasks
    list(ordered_map(square, [(i,) for i in range(10)], num_workers=5))
    assert sizes == [3, 2, 5]


def test_workers_have_no_gpu_and_one_thread():
    env = list(ordered_map(worker_environment, [(i,) for i in range(4)], num_workers=2))
    assert all(e == ("", "1") for e in env)


def test_an_error_in_a_worker_reaches_the_consumer():
    with pytest.raises(ValueError, match="bad task 3"):
        list(ordered_map(fail_on, [(i, 3) for i in range(8)], num_workers=2))


def test_stopping_early_shuts_the_pool_down():
    it = ordered_map(square, [(i,) for i in range(100)], num_workers=2, window=4)
    assert next(it) == 0
    it.close()  # must return promptly, cancelling what has not started


def _records_equal(a, b):
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert len(ra) == len(rb)
        for x, y in zip(ra, rb):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_dials_files_parse_the_same_in_parallel():
    pytest.importorskip("dxtbx")
    from abismal.io import StillsLoader
    here = os.path.join(os.path.dirname(__file__), "..", "data")
    expt, refl = os.path.join(here, "stills.expt"), os.path.join(here, "stills.refl")
    loader = StillsLoader([expt, expt], [refl, refl], None, None, 2.0)
    serial = list(loader.get_dataset(num_cpus=1))
    pooled = list(loader.get_dataset(num_cpus=2))
    _records_equal(serial, pooled)
    assert len(serial) > 0


def test_stream_chunks_parse_the_same_in_parallel():
    from abismal.io import StreamLoader
    stream = os.path.join(os.path.dirname(__file__), "..", "data", "crystfel.stream")
    serial = list(StreamLoader(stream, dmin=2.0).get_dataset(num_cpus=1))
    pooled = list(StreamLoader(stream, dmin=2.0).get_dataset(num_cpus=3))
    _records_equal(serial, pooled)
    assert len(serial) > 0


SCRIPT = """
import operator, os, sys
sys.path.insert(0, {repo!r})
from abismal.io.parallel import ordered_map
print("top level", flush=True)
print(list(ordered_map(operator.mul, [(i, i) for i in range(6)], num_workers=3)), flush=True)
"""


@pytest.mark.parametrize("how", ["file", "stdin"])
def test_workers_do_not_rerun_the_callers_script(tmp_path, how):
    """ An unguarded script, or one read from stdin, runs once, in the parent only """
    import subprocess, sys
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    code = SCRIPT.format(repo=repo)
    if how == "file":
        path = tmp_path / "unguarded.py"
        path.write_text(code)
        out = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=240)
    else:
        out = subprocess.run([sys.executable, "-"], input=code, capture_output=True, text=True, timeout=240)
    assert out.returncode == 0, out.stderr
    assert out.stdout.count("top level") == 1
    assert "[0, 1, 4, 9, 16, 25]" in out.stdout
