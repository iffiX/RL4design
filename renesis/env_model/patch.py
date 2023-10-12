import numpy as np
from typing import List
from gymnasium.spaces import Box
from .base import BaseModel
from renesis.utils.robot import get_robot_voxels_from_voxels


class PatchModel(BaseModel):
    def __init__(
        self,
        materials=(0, 1, 2),
        dimension_size=(20, 20, 20),
        patch_size=1,
        max_patch_num=100,
    ):
        """
        The patch model.

        In each step the policy can place a bundle of voxels, called a patch,
        to a specific position in space. The patch is of fixed shape and same
        material. This model placed cubes, patch_size=1 means a 1x1x1 cube is
        placed in each step.

        The configuration of voxels added by the policy is conditioned on the
        distribution of all placed voxels.
        """
        assert len(dimension_size) == 3
        dimension_size = list(dimension_size)
        super().__init__()
        self.materials = materials
        self.dimension_size = dimension_size
        self.voxel_num = dimension_size[0] * dimension_size[1] * dimension_size[2]
        self.center_voxel_offset = [size // 2 for size in dimension_size]
        self.patch_size = patch_size
        self.max_patch_num = max_patch_num

        # A list of arrays of shape [3 + len(self.materials)],
        # first 3 elements are mean (x, y, z)
        # Remaining elements are material weights
        self.patches = []  # type: List[np.ndarray]
        self.prev_voxels = np.zeros(dimension_size, dtype=np.float32)
        self.voxels = np.zeros(dimension_size, dtype=np.float32)
        self.occupied = np.zeros(dimension_size, dtype=np.bool)
        self.robot_voxels = np.zeros(dimension_size, dtype=np.float32)
        self.robot_occupied = np.zeros(dimension_size, dtype=np.bool)
        self.invalid_count = 0
        self.is_robot_valid = False
        self.update_voxels()

    @property
    def action_space(self):
        """
        Each action is [x, y, z, mat_1, ..., mat_n]
        x, y, z is the center coordinate where patch is placed,
        mat_1, ..., mat_n is the material weight, material with the largest
        weight is chosen.
        """
        return Box(low=0, high=1, shape=(3 + len(self.materials),))

    @property
    def observation_space(self):
        """
        Each observation is a flattened array of shape
        [dimension_size_x * dimension_size_y * dimension_size_z] of all
        placed voxels, the values are discrete material index.
        """
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

    def reset(self):
        self.steps = 0
        self.patches = []
        self.update_voxels()
        self.prev_voxels = self.voxels

    def is_finished(self):
        return self.steps >= self.max_patch_num

    def is_robot_invalid(self):
        return not self.is_robot_valid

    def step(self, action: np.ndarray):
        self.prev_voxels = self.voxels
        self.patches.append(action)
        self.update_voxels()
        self.steps += 1

    def observe(self):
        return self.voxels.reshape(-1)

    def get_robot(self):
        x_occupied = [
            x
            for x in range(self.robot_occupied.shape[0])
            if np.any(self.robot_occupied[x])
        ]
        y_occupied = [
            y
            for y in range(self.robot_occupied.shape[1])
            if np.any(self.robot_occupied[:, y])
        ]
        z_occupied = [
            z
            for z in range(self.robot_occupied.shape[2])
            if np.any(self.robot_occupied[:, :, z])
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
                self.robot_voxels[min_x:max_x, min_y:max_y, z]
                .astype(int)
                .flatten(order="F")
                .tolist(),
                None,
                None,
                None,
            )
            representation.append(layer_representation)
        return (max_x - min_x, max_y - min_y, max_z - min_z), representation

    def get_robot_voxels(self):
        return self.robot_voxels

    def get_voxels(self):
        return self.voxels

    def get_state_data(self):
        return np.stack(self.patches), self.voxels

    def scale(self, action):
        min_value = [-offset for offset in self.center_voxel_offset]
        return np.array(min_value + [0] * len(self.materials)) + action * np.array(
            self.dimension_size + [1] * len(self.materials)
        )

    def update_voxels(self):
        # generate coordinates
        # Eg: if dimension size is 20, indices are [-10, ..., 9]
        # if dimension size if 21, indices are [-10, ..., 10]
        indices = [
            list(
                range(
                    -offset,
                    size - offset,
                )
            )
            for size, offset in zip(self.dimension_size, self.center_voxel_offset)
        ]
        coords = np.stack(np.meshgrid(*indices, indexing="ij"))
        # coords shape [coord_num, 3]
        coords = np.transpose(coords.reshape([coords.shape[0], -1]))
        all_values = []
        patch_radius = self.patch_size / 2
        for idx, patch in enumerate(self.patches):
            patch = self.scale(patch)
            covered = np.all(
                np.array(
                    [
                        coords[:, 0] >= patch[0] - patch_radius,
                        coords[:, 0] < patch[0] + patch_radius,
                        coords[:, 1] >= patch[1] - patch_radius,
                        coords[:, 1] < patch[1] + patch_radius,
                        coords[:, 2] >= patch[2] - patch_radius,
                        coords[:, 2] < patch[2] + patch_radius,
                    ]
                ),
                axis=0,
            )
            # later added patches has a higher weight,
            # so previous patches will be overwritten
            # Add 1 so that the first patch is not zero
            # because idx starts from 0
            all_values.append(covered * (idx + 1))

        self.voxels = np.zeros(
            self.dimension_size,
            dtype=np.float32,
        )

        if self.patches:
            # all_values shape [coord_num, patch_num]
            all_values = np.stack(all_values, axis=1)
            material_map = np.array(
                [self.materials[int(np.argmax(patch[3:]))] for patch in self.patches]
            )
            material = np.where(
                np.any(all_values > 0, axis=1),
                material_map[np.argmax(all_values, axis=1)],
                0,
            )

            self.voxels[
                coords[:, 0] + self.center_voxel_offset[0],
                coords[:, 1] + self.center_voxel_offset[1],
                coords[:, 2] + self.center_voxel_offset[2],
            ] = material

        self.occupied = self.voxels != 0
        self.robot_voxels, self.robot_occupied = get_robot_voxels_from_voxels(
            self.voxels, self.occupied
        )
        self.is_robot_valid = np.any(self.robot_occupied)


class PatchSphereModel(PatchModel):
    """
    The spherical patch model.

    In each step the policy can place a bundle of voxels, called a patch,
    to a specific position in space. The patch is of fixed shape and same
    material. This model placed cubes, patch_size=3 means a ball of diameter
    3 is placed in each step. In low resolution, it is just placing 7 voxels.

    The configuration of voxels added by the policy is conditioned on the
    distribution of all placed voxels.
    """

    def update_voxels(self):
        # generate coordinates
        # Eg: if dimension size is 20, indices are [-10, ..., 9]
        # if dimension size if 21, indices are [-10, ..., 10]
        indices = [
            list(
                range(
                    -offset,
                    size - offset,
                )
            )
            for size, offset in zip(self.dimension_size, self.center_voxel_offset)
        ]
        coords = np.stack(np.meshgrid(*indices, indexing="ij"))
        # coords shape [coord_num, 3]
        coords = np.transpose(coords.reshape([coords.shape[0], -1]))
        all_values = []
        patch_radius = (self.patch_size - 1) / 2 + 1e-3
        for idx, patch in enumerate(self.patches):
            patch = self.scale(patch)
            patch_center = np.round(patch[:3])
            covered = np.linalg.norm(coords - patch_center, axis=-1) <= patch_radius
            # later added patches has a higher weight,
            # so previous patches will be overwritten
            # Add 1 so that the first patch is not zero
            # because idx starts from 0
            all_values.append(covered * (idx + 1))

        self.voxels = np.zeros(
            self.dimension_size,
            dtype=np.float32,
        )

        if self.patches:
            # all_values shape [coord_num, patch_num]
            all_values = np.stack(all_values, axis=1)
            material_map = np.array(
                [self.materials[int(np.argmax(patch[3:]))] for patch in self.patches]
            )
            material = np.where(
                np.any(all_values > 0, axis=1),
                material_map[np.argmax(all_values, axis=1)],
                0,
            )

            self.voxels[
                coords[:, 0] + self.center_voxel_offset[0],
                coords[:, 1] + self.center_voxel_offset[1],
                coords[:, 2] + self.center_voxel_offset[2],
            ] = material

        self.occupied = self.voxels != 0
        self.robot_voxels, self.robot_occupied = get_robot_voxels_from_voxels(
            self.voxels, self.occupied
        )
        self.is_robot_valid = np.any(self.robot_occupied)
