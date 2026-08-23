import logging
import warnings
from contextlib import contextmanager
from importlib.util import find_spec


def set_ray_loglevel(level):
    logger = logging.getLogger("ray")
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def check_for_ray():
    has_ray = True
    if find_spec("ray") is None:
        has_ray = False

        message = (
            "ray (https://www.ray.io/) is not available..." "Falling back to serial."
        )
        warnings.warn(message, ImportWarning)
    return has_ray


def check_for_mpi():
    try:
        from mpi4py import MPI

        return True
    except Exception as err:
        message = (
            f"Failed `from mpi4py import MPI` with {err}. Falling back to serial mode."
        )
        warnings.warn(message, ImportWarning)
        return False


@contextmanager
def ray_context(log_level="DEBUG", **ray_kwargs):
    import ray

    set_ray_loglevel(log_level)

    # Hide the GPU from the workers. These tasks only parse reflection files, but a
    # worker that can see the GPU initializes its own TensorFlow GPU context, and
    # that context takes essentially all the free memory -- set_memory_growth runs
    # in the training process (abismal.io.set_gpu), not here. The workers run during
    # data loading, so they get there first and training then OOMs on its first
    # matmul with only a GB left.
    #
    # Ray used to do this for us: through 2.53 a task requesting no GPUs got
    # CUDA_VISIBLE_DEVICES="", but 2.57 leaves the variable unset. Set it here so
    # the behaviour does not depend on the ray version.
    runtime_env = dict(ray_kwargs.pop("runtime_env", None) or {})
    env_vars = dict(runtime_env.get("env_vars") or {})
    env_vars.setdefault("CUDA_VISIBLE_DEVICES", "")
    runtime_env["env_vars"] = env_vars

    ray.init(runtime_env=runtime_env, **ray_kwargs)
    try:
        yield ray
    finally:
        ray.shutdown()
