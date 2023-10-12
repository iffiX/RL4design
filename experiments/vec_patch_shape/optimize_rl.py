import ray
import numpy as np
from ray import tune
from ray.tune.logger import TBXLoggerCallback
from launch.snapshot import init_or_load_launch_config
from experiments.vec_patch_shape.model import CustomPPO
from experiments.vec_patch_shape.utils import DataLoggerCallback
from experiments.vec_patch_shape.config import (
    config,
    iters,
)

if __name__ == "__main__":
    # 1GB heap memory, 1GB object store
    ray.init(_memory=1 * (10**9), object_store_memory=iters * (5 * 10**6) * 1.1)

    tune.run(
        CustomPPO,
        config=config,
        name="",
        checkpoint_freq=1,
        log_to_file=True,
        stop={
            "timesteps_total": config["train_batch_size"] * iters,
            "episodes_total": np.infty,
        },
        # Order is important!
        callbacks=[
            DataLoggerCallback(),
            TBXLoggerCallback(),
        ],
        restore="",
        local_dir=init_or_load_launch_config()["save_location"],
    )
    ray.shutdown()
