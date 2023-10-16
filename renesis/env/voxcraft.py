import os
import numpy as np
from time import time
from typing import Dict, Any, Optional
from ray.rllib import VectorEnv
from ray.rllib.utils import override
from renesis.sim import Voxcraft
from renesis.utils.voxcraft import (
    vxd_creator,
    get_voxel_positions,
    get_center_of_mass,
)
from renesis.utils.metrics import (
    distance_traveled,
    distance_traveled_of_com,
)
from renesis.utils.debug import enable_debugger
from renesis.env.utils import normalize
from renesis.env_model.base import BaseVectorizedModel
from renesis.env_model.vec_patch import VectorizedPatchModel, VectorizedPatchSphereModel


class VoxcraftSingleRewardBaseEnvironmentForVecEnvModel(VectorEnv):
    """
    Voxcraft environment where we simulate how robot bodies behave and measure
    various performance metrics such as travel distance, travel height, etc.
    """

    metadata = {"render.modes": ["ansi"]}

    def __init__(self, config: Dict[str, Any], vec_env_model: BaseVectorizedModel):
        self.config = config
        self.vec_env_model = vec_env_model
        self.max_steps = config["max_steps"]
        self.action_space = vec_env_model.action_space
        self.observation_space = vec_env_model.observation_space
        self.reward_range = (0, float("inf"))
        with open(config["base_config_path"], "r") as file:
            self.base_config = file.read()
        self.reward_type = config["reward_type"]
        self.voxel_size = config["voxel_size"]
        self.fallen_threshold = config["fallen_threshold"]

        self.end_rewards = [0 for _ in range(config["num_envs"])]
        self.end_robots = ["\n" for _ in range(config["num_envs"])]
        self.end_records = ["" for _ in range(config["num_envs"])]

        self.simulator = Voxcraft(batch_size_per_device=config["num_envs"])
        self.reset_envs = set()
        self.time = {
            "env_model_step_time": [],
            "env_model_observe_time": [],
            "sim_time": 0.0,
            "compute_reward_time": 0.0,
        }
        super().__init__(self.observation_space, self.action_space, config["num_envs"])

    def print_and_reset_time(self):
        if len(self.time["env_model_step_time"]) > 0:
            print(
                f"Avg Env model step ms: {np.mean(self.time['env_model_step_time']) * 1000:.2f}\n"
                f"Avg Env model observe ms: {np.mean(self.time['env_model_observe_time']) * 1000:.2f}\n"
                f"Sim ms: {self.time['sim_time'] * 1000:.2f}\n"
                f"Compute reward ms: {self.time['compute_reward_time'] * 1000:.2f}"
            )
        self.time = {
            "env_model_step_time": [],
            "env_model_observe_time": [],
            "sim_time": 0.0,
            "compute_reward_time": 0.0,
        }

    @override(VectorEnv)
    def reset_at(self, index: Optional[int] = None, *args, **kwargs):
        # Will reset all models at end
        self.end_rewards[index] = 0
        self.end_robots[index] = "\n"
        self.end_records[index] = ""
        if index in self.reset_envs:
            raise RuntimeError("Environment already reset")
        else:
            self.reset_envs.add(index)
            if len(self.reset_envs) == self.num_envs:
                self.vec_env_model.reset()
                self.reset_envs.clear()
                self.print_and_reset_time()
        return self.vec_env_model.initial_observation_after_reset_single_env, None

    @override(VectorEnv)
    def restart_at(self, index: Optional[int] = None):
        self.reset_at(index)

    @override(VectorEnv)
    def vector_reset(self):
        self.vec_env_model.reset()
        self.end_rewards = [0 for _ in range(self.num_envs)]
        self.end_robots = ["\n" for _ in range(self.num_envs)]
        self.end_records = ["" for _ in range(self.num_envs)]
        self.print_and_reset_time()
        return self.vec_env_model.observe(), [{} for _ in range(self.num_envs)]

    @override(VectorEnv)
    def vector_step(self, actions):
        before_finished = self.vec_env_model.is_finished() or (
            self.vec_env_model.steps == self.max_steps
        )
        begin = time()
        if not before_finished:
            self.vec_env_model.step(actions)
        after_finished = self.vec_env_model.is_finished() or (
            self.vec_env_model.steps == self.max_steps
        )
        env_model_step_time = time() - begin
        if not before_finished and after_finished:
            self.update_rewards()

        begin = time()
        obs = self.vec_env_model.observe()
        env_model_observe_time = time() - begin
        self.time["env_model_step_time"].append(env_model_step_time)
        self.time["env_model_observe_time"].append(env_model_observe_time)

        # print(actions)
        # print([f"{r:.3f}" for r in reward_signals])
        return (
            obs,
            self.end_rewards
            if not before_finished and after_finished
            else [0] * self.num_envs,
            [after_finished] * self.num_envs,
            [False for _ in range(self.num_envs)],
            [{} for _ in range(self.num_envs)],
        )

    def update_rewards(self):
        """
        Update rewards of all sub environments.
        """
        begin = time()
        robots, (results, records) = self.run_simulations()
        sim_time = time() - begin

        begin = time()
        all_start_pos, all_end_pos = [], []
        all_start_com, all_end_com = [], []
        for result in results:
            if result is not None:
                try:
                    start_pos, end_pos = get_voxel_positions(
                        result, voxel_size=self.voxel_size
                    )
                    start_com, end_com = get_center_of_mass(
                        result, voxel_size=self.voxel_size
                    )
                    all_start_pos.append(start_pos)
                    all_end_pos.append(end_pos)
                    all_start_com.append(start_com)
                    all_end_com.append(end_com)
                except:
                    all_start_pos.append(None)
                    all_end_pos.append(None)
                    all_start_com.append(None)
                    all_end_com.append(None)
            else:
                all_start_pos.append(None)
                all_end_pos.append(None)
                all_start_com.append(None)
                all_end_com.append(None)
        self.end_rewards = self.compute_reward_from_sim_result(
            all_start_pos,
            all_end_pos,
            all_start_com,
            all_end_com,
            self.vec_env_model.get_robots_voxels(),
        )
        compute_reward_time = time() - begin

        self.end_robots = robots
        self.end_records = records
        self.time["sim_time"] = sim_time
        self.time["compute_reward_time"] = compute_reward_time

    def run_simulations(self):
        """
        Run a simulation using current representation.

        Returns:
            A list of robots (each robot is a VXD file in string).
            A tuple, first member is a list of simulation summary
            result string, the second member is a list of simulation
            recording string.
        """
        robots = []
        valid_indices = []
        for idx, (sizes, representation) in enumerate(self.vec_env_model.get_robots()):
            if sizes[0] == 0:
                # which means robot is empty since all three sizes are 0
                continue
            valid_indices.append(idx)
            robots.append(vxd_creator(sizes, representation, record_history=True))

        all_robots = [None] * self.vec_env_model.env_num
        all_out = [
            [None] * self.vec_env_model.env_num,
            [None] * self.vec_env_model.env_num,
        ]

        if len(valid_indices) > 0:
            begin = time()
            for attempt in range(3):
                try:
                    out = self.simulator.run_sims(
                        [self.base_config] * len(robots), robots, save_record=False
                    )
                    end = time()
                except Exception as e:
                    print(f"Failed attempt {attempt + 1}")
                    if attempt == 2:
                        print(f"Final attempt failed")
                        dump_dir = os.path.expanduser(f"~/renesis_sim_dump/{begin}")
                        os.makedirs(dump_dir, exist_ok=True)
                        print(f"Debug info saved to {dump_dir}")
                        with open(os.path.join(dump_dir, "base.vxa"), "w") as file:
                            file.write(self.base_config)
                        for i, robot in enumerate(robots):
                            with open(os.path.join(dump_dir, f"{i}.vxd"), "w") as file:
                                file.write(robot)
                        raise e
                else:
                    print(
                        f"{len(robots)} simulations total {end - begin:.3f}s, "
                        f"average {(end - begin) / len(robots):.3f}s"
                    )
                    for idx, robot, result, record in zip(
                        valid_indices, robots, out[0], out[1]
                    ):
                        all_robots[idx] = robot
                        all_out[0][idx] = result
                        all_out[1][idx] = record
                    break
        else:
            print("Warning: no valid robot in vectorized batch")
        return all_robots, all_out

    def compute_reward_from_sim_result(
        self, all_start_pos, all_end_pos, all_start_com, all_end_com, all_voxels
    ):
        """
        Note: reward should always have an initial value of 0 for empty robots.

        distance_traveled: optimize for maximize the max distance of all voxels
            from their start position to end position
        distance_traveled_com: use center of mass to measure distance instead of
            using all voxels
        distance_traveled_com_restricted_axis: restrict to only measure distance
            in the x-axis.
        """
        rewards = []
        if self.reward_type == "distance_traveled":
            for start_pos, end_pos in zip(all_start_pos, all_end_pos):
                if start_pos is None:
                    reward = 0
                else:
                    reward = distance_traveled(start_pos, end_pos)
                    if reward < 1e-3:
                        reward = 0
                rewards.append(reward)
        elif self.reward_type == "distance_traveled_com":
            for start_com, end_com in zip(all_start_com, all_end_com):
                if start_com is None:
                    reward = 0
                else:
                    reward = distance_traveled_of_com(start_com, end_com)
                    if reward < 1e-3:
                        reward = 0
                rewards.append(reward)
        elif self.reward_type == "distance_traveled_com_restricted_axis":
            for start_com, end_com in zip(all_start_com, all_end_com):
                if start_com is None:
                    reward = 0
                else:
                    diff = np.abs(end_com - start_com)
                    reward = diff[0] - diff[1] - diff[2]
                    if reward < 1e-3:
                        reward = 0
                rewards.append(reward)
        else:
            raise Exception("Unknown reward type: {self.reward_type}")
        rewards = [reward if not np.isnan(reward) else 0 for reward in rewards]
        return rewards

    def render(self, mode="ansi"):
        if mode == "ansi":
            return self.robot[0] + "\n"


class VoxcraftSingleRewardVectorizedPatchEnvironment(
    VoxcraftSingleRewardBaseEnvironmentForVecEnvModel
):
    def __init__(self, config):
        if config.get("debug", False):
            enable_debugger(
                config.get("debug_ip", "localhost"), config.get("debug_port", 8223)
            )
        super().__init__(
            config,
            VectorizedPatchModel(
                materials=config["materials"],
                dimension_size=config["dimension_size"],
                patch_size=config["patch_size"],
                max_patch_num=config["max_patch_num"],
                env_num=config["num_envs"],
            ),
        )

    def vector_step(self, actions):
        normalize_mode = self.config.get("normalize_mode", "clip")
        return super().vector_step(
            [normalize(action, mode=normalize_mode) for action in actions]
        )


class VoxcraftSingleRewardVectorizedPatchSphereEnvironment(
    VoxcraftSingleRewardBaseEnvironmentForVecEnvModel
):
    def __init__(self, config):
        if config.get("debug", False):
            enable_debugger(
                config.get("debug_ip", "localhost"), config.get("debug_port", 8223)
            )
        super().__init__(
            config,
            VectorizedPatchSphereModel(
                materials=config["materials"],
                dimension_size=config["dimension_size"],
                patch_size=config["patch_size"],
                max_patch_num=config["max_patch_num"],
                env_num=config["num_envs"],
            ),
        )

    def vector_step(self, actions):
        normalize_mode = self.config.get("normalize_mode", "clip")
        return super().vector_step(
            [normalize(action, mode=normalize_mode) for action in actions]
        )
