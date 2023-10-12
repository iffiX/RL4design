import torch as t
import numpy as np
from typing import List
from gymnasium.spaces import Box
from multiprocessing.pool import Pool
from .base import BaseVectorizedModel
from renesis.utils.robot import get_robot_voxels_from_voxels


class VectorizedPatchModel(BaseVectorizedModel):
    def __init__(
        self,
        materials=(0, 1, 2),
        dimension_size=(20, 20, 20),
        patch_size=1,
        max_patch_num=100,
        env_num=100,
        device=None,
    ):
        """
        Vectorized version of PatchModel
        """
        if device is None:
            device = t.cuda.current_device()
        assert len(dimension_size) == 3
        dimension_size = list(dimension_size)
        super().__init__()
        self.materials = materials
        self.dimension_size = dimension_size
        self.voxel_num = dimension_size[0] * dimension_size[1] * dimension_size[2]
        self.center_voxel_offset = [size // 2 for size in dimension_size]
        self.patch_size = patch_size
        self.max_patch_num = max_patch_num
        self.env_num = env_num
        self.device = device

        # A list of arrays of shape [env_num, 3 + len(self.materials)],
        # For the last dimension, first 3 elements are mean (x, y, z)
        # Remaining elements are material weights
        self.vec_patches = []  # type: List[np.ndarray]
        self.vec_prev_voxels = np.zeros([env_num] + dimension_size, dtype=np.float32)
        self.vec_voxels = np.zeros([env_num] + dimension_size, dtype=np.float32)
        self.vec_occupied = np.zeros([env_num] + dimension_size, dtype=bool)
        self.vec_robot_voxels = np.zeros([env_num] + dimension_size, dtype=np.float32)
        self.vec_robot_occupied = np.zeros([env_num] + dimension_size, dtype=bool)
        self.last_biggest_value = t.zeros(
            [self.env_num, np.prod(self.dimension_size)], device=self.device
        )
        self.pool = Pool()

    @property
    def action_space(self):
        return Box(low=0, high=1, shape=(3 + len(self.materials),))

    @property
    def observation_space(self):
        return Box(
            low=np.array(
                (min(min(self.materials), 0),) * self.voxel_num,
                dtype=np.float32,
            ),
            high=np.array(
                (max(max(self.materials), 0),) * self.voxel_num,
                dtype=np.float32,
            ),
        )

    @property
    def initial_observation_after_reset_single_env(self):
        return np.zeros(self.voxel_num, dtype=np.float32)

    def reset(self):
        self.steps = 0
        self.vec_patches = []
        self.vec_prev_voxels = np.zeros(
            [self.env_num] + self.dimension_size, dtype=np.float32
        )
        self.vec_voxels = np.zeros(
            [self.env_num] + self.dimension_size, dtype=np.float32
        )
        self.vec_occupied = np.zeros([self.env_num] + self.dimension_size, dtype=bool)
        self.vec_robot_voxels = np.zeros(
            [self.env_num] + self.dimension_size, dtype=np.float32
        )
        self.vec_robot_occupied = np.zeros(
            [self.env_num] + self.dimension_size, dtype=bool
        )
        self.last_biggest_value = t.zeros(
            [self.env_num, np.prod(self.dimension_size)], device=self.device
        )

    def is_finished(self):
        return self.steps >= self.max_patch_num

    def step(self, actions: np.ndarray):
        self.vec_prev_voxels = self.vec_voxels
        self.vec_patches.append(np.array(actions))
        self.update_voxels()
        self.steps += 1

    def observe(self):
        return [obs for obs in self.vec_voxels.reshape(self.env_num, -1)]

    def get_robots(self):
        result = self.pool.map(
            self.get_robots_worker,
            list(zip(self.vec_robot_voxels, self.vec_robot_occupied)),
        )
        return result

    @staticmethod
    def get_robots_worker(args):
        robot_voxels, robot_occupied = args
        if not np.any(robot_occupied):
            return (0, 0, 0), []
        x_occupied = [
            x for x in range(robot_occupied.shape[0]) if np.any(robot_occupied[x])
        ]
        y_occupied = [
            y for y in range(robot_occupied.shape[1]) if np.any(robot_occupied[:, y])
        ]
        z_occupied = [
            z for z in range(robot_occupied.shape[2]) if np.any(robot_occupied[:, :, z])
        ]
        min_x = min(x_occupied)
        max_x = max(x_occupied) + 1
        min_y = min(y_occupied)
        max_y = max(y_occupied) + 1
        min_z = min(z_occupied)
        max_z = max(z_occupied) + 1
        representation = []

        for z in range(min_z, max_z):
            layer_representation = (
                robot_voxels[min_x:max_x, min_y:max_y, z]
                .astype(int)
                .flatten(order="F")
                .tolist(),
                None,
                None,
                None,
            )
            representation.append(layer_representation)
        return (max_x - min_x, max_y - min_y, max_z - min_z), representation

    def get_robots_voxels(self):
        return [rv for rv in self.vec_robot_voxels]

    def get_voxels(self):
        return [v for v in self.vec_voxels]

    def scale(self, action):
        min_value = [-offset for offset in self.center_voxel_offset]
        return np.array([min_value + [0] * len(self.materials)]) + action * np.array(
            [self.dimension_size + [1] * len(self.materials)]
        )

    def update_voxels(self):
        # generate coordinates
        # Eg: if dimension size is 20, indices are [-10, ..., 9]
        # if dimension size if 21, indices are [-10, ..., 10]
        indices = [
            t.arange(-offset, size - offset, dtype=t.long, device=self.device)
            for size, offset in zip(self.dimension_size, self.center_voxel_offset)
        ]
        # coords shape [1, coord_num, 3]
        coords = (
            t.stack(t.meshgrid(*indices, indexing="ij"))
            .view(3, -1)
            .transpose(1, 0)
            .unsqueeze(0)
        )

        patch_radius = self.patch_size / 2

        idx = len(self.vec_patches) - 1

        # patch shape [env_num, action_dim]
        patch = t.from_numpy(self.scale(self.vec_patches[-1])).to(
            device=self.device, dtype=t.float32
        )
        # covered shape [env_num, coord_num]
        covered = t.all(
            t.stack(
                [
                    coords[:, :, 0] >= patch[:, 0:1] - patch_radius,
                    coords[:, :, 0] < patch[:, 0:1] + patch_radius,
                    coords[:, :, 1] >= patch[:, 1:2] - patch_radius,
                    coords[:, :, 1] < patch[:, 1:2] + patch_radius,
                    coords[:, :, 2] >= patch[:, 2:3] - patch_radius,
                    coords[:, :, 2] < patch[:, 2:3] + patch_radius,
                ]
            ),
            dim=0,
        )
        # later added patches has a higher weight,
        # so previous patches will be overwritten
        # Add 1 so that the first patch is not zero
        # because idx starts from 0
        patch_value = (covered * (idx + 1)).to(dtype=t.float32)
        overwrite_idx = patch_value > self.last_biggest_value
        self.last_biggest_value = t.where(
            overwrite_idx, patch_value, self.last_biggest_value
        )

        modify_voxels = t.full(
            [self.env_num] + self.dimension_size,
            -1,
            dtype=t.float32,
            device=self.device,
        )

        # material map shape [env_num, 1]
        material_map = t.tensor(
            [self.materials[int(mat)] for mat in t.argmax(patch[:, 3:], dim=1)],
            device=self.device,
        ).unsqueeze(1)
        # material shape [env_num, coord_num]
        modify_material = t.where(
            overwrite_idx,
            material_map,
            -1,
        )

        modify_voxels[
            t.arange(0, self.env_num, dtype=t.long).unsqueeze(1),
            (coords[:, :, 0] + self.center_voxel_offset[0]),
            (coords[:, :, 1] + self.center_voxel_offset[1]),
            (coords[:, :, 2] + self.center_voxel_offset[2]),
        ] = modify_material.to(dtype=t.float32)

        old_vec_voxels = t.from_numpy(self.vec_voxels).to(device=self.device)
        new_vec_voxels = t.where(modify_voxels != -1, modify_voxels, old_vec_voxels)
        self.vec_voxels = new_vec_voxels.cpu().numpy()
        self.vec_occupied = (new_vec_voxels != 0).cpu().numpy()
        result = self.pool.starmap(
            get_robot_voxels_from_voxels,
            list(zip(self.vec_voxels, self.vec_occupied)),
        )
        self.vec_robot_voxels = np.array([res[0] for res in result])
        self.vec_robot_occupied = np.array([res[1] for res in result])


class VectorizedPatchSphereModel(VectorizedPatchModel):
    """
    Vectorized version of PatchSphereModel
    """

    def update_voxels(self):
        # generate coordinates
        # Eg: if dimension size is 20, indices are [-10, ..., 9]
        # if dimension size if 21, indices are [-10, ..., 10]
        indices = [
            t.arange(-offset, size - offset, dtype=t.long, device=self.device)
            for size, offset in zip(self.dimension_size, self.center_voxel_offset)
        ]
        # coords shape [env_num, coord_num, 3]
        coords = (
            t.stack(t.meshgrid(*indices, indexing="ij"))
            .view(3, -1)
            .transpose(1, 0)
            .unsqueeze(0)
            .repeat(self.env_num, 1, 1)
        )
        patch_radius = (self.patch_size - 1) / 2 + 1e-3

        idx = len(self.vec_patches) - 1

        # patch shape [env_num, action_dim]
        patch = t.from_numpy(self.scale(self.vec_patches[-1])).to(
            device=self.device, dtype=t.float32
        )
        # patch_center shape [env_num, 1, 3]
        patch_center = t.round(patch[:, None, :3])
        # covered shape [env_num, coord_num]
        covered = t.norm(coords - patch_center, dim=-1) <= patch_radius
        # later added patches has a higher weight,
        # so previous patches will be overwritten
        # Add 1 so that the first patch is not zero
        # because idx starts from 0
        patch_value = (covered * (idx + 1)).to(dtype=t.float32)
        overwrite_idx = patch_value > self.last_biggest_value
        self.last_biggest_value = t.where(
            overwrite_idx, patch_value, self.last_biggest_value
        )

        modify_voxels = t.full(
            [self.env_num] + self.dimension_size,
            -1,
            dtype=t.float32,
            device=self.device,
        )

        # material map shape [env_num, 1]
        material_map = t.tensor(
            [self.materials[int(mat)] for mat in t.argmax(patch[:, 3:], dim=1)],
            device=self.device,
        ).unsqueeze(1)
        # material shape [env_num, coord_num]
        modify_material = t.where(
            overwrite_idx,
            material_map,
            -1,
        )

        modify_voxels[
            t.arange(0, self.env_num, dtype=t.long).unsqueeze(1),
            (coords[:, :, 0] + self.center_voxel_offset[0]),
            (coords[:, :, 1] + self.center_voxel_offset[1]),
            (coords[:, :, 2] + self.center_voxel_offset[2]),
        ] = modify_material.to(dtype=t.float32)

        old_vec_voxels = t.from_numpy(self.vec_voxels).to(device=self.device)
        new_vec_voxels = t.where(modify_voxels != -1, modify_voxels, old_vec_voxels)
        self.vec_voxels = new_vec_voxels.cpu().numpy()
        self.vec_occupied = (new_vec_voxels != 0).cpu().numpy()
        result = self.pool.starmap(
            get_robot_voxels_from_voxels,
            list(zip(self.vec_voxels, self.vec_occupied)),
        )
        self.vec_robot_voxels = np.array([res[0] for res in result])
        self.vec_robot_occupied = np.array([res[1] for res in result])
