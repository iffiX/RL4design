import numpy as np


def scale(ratio, min, max):
    return min + (max - min) * ratio


def generate_sphere(dimension_size, radius_ratio=0.25, material=1):
    voxels = np.zeros([dimension_size, dimension_size, dimension_size], dtype=np.int)
    center_voxel_offset = dimension_size // 2
    indices = list(
        range(
            -center_voxel_offset,
            dimension_size - center_voxel_offset,
        )
    )
    coords = np.stack(np.meshgrid(indices, indices, indices, indexing="ij"))
    # coords shape [coord_num, 3]
    coords = np.transpose(coords.reshape([coords.shape[0], -1]))
    distance = np.linalg.norm(coords, axis=1)
    all_material = np.where(distance < dimension_size * radius_ratio, material, 0)
    voxels[
        coords[:, 0] + center_voxel_offset,
        coords[:, 1] + center_voxel_offset,
        coords[:, 2] + center_voxel_offset,
    ] = all_material
    return voxels


def generate_random_ellipsoids(dimension_size, num=5, materials=(1, 2, 3), seed=42):
    voxels = np.zeros([dimension_size, dimension_size, dimension_size], dtype=np.int)
    center_voxel_offset = dimension_size // 2
    indices = list(
        range(
            -center_voxel_offset,
            dimension_size - center_voxel_offset,
        )
    )
    coords = np.stack(np.meshgrid(indices, indices, indices, indexing="ij"))
    # coords shape [coord_num, 3]
    coords = np.transpose(coords.reshape([coords.shape[0], -1]))
    rand = np.random.RandomState(seed)
    ellipsoids = [
        (
            scale(rand.rand(), -dimension_size / 3, dimension_size / 3),
            scale(rand.rand(), -dimension_size / 3, dimension_size / 3),
            scale(rand.rand(), -dimension_size / 3, dimension_size / 3),
            scale(rand.rand(), min(1, dimension_size / 10), dimension_size / 6),
            scale(rand.rand(), min(1, dimension_size / 10), dimension_size / 6),
            scale(rand.rand(), min(1, dimension_size / 10), dimension_size / 6),
            rand.choice(materials),
        )
        for _ in range(num)
    ]
    all_material = np.zeros((dimension_size**3,), dtype=np.int)
    for ellipsoid in ellipsoids:
        x_diff = coords[:, 0] - ellipsoid[0]
        y_diff = coords[:, 1] - ellipsoid[1]
        z_diff = coords[:, 2] - ellipsoid[2]
        all_material = np.where(
            x_diff**2 / ellipsoid[3] ** 2
            + y_diff**2 / ellipsoid[4] ** 2
            + z_diff**2 / ellipsoid[5] ** 2
            <= 1,
            ellipsoid[6],
            all_material,
        )
    voxels[
        coords[:, 0] + center_voxel_offset,
        coords[:, 1] + center_voxel_offset,
        coords[:, 2] + center_voxel_offset,
    ] = all_material
    return voxels


def generate_inversed_random_ellipsoids(dimension_size, num=5, material=1, seed=42):
    voxels = np.full(
        [dimension_size, dimension_size, dimension_size], material, dtype=np.int
    )
    center_voxel_offset = dimension_size // 2
    indices = list(
        range(
            -center_voxel_offset,
            dimension_size - center_voxel_offset,
        )
    )
    coords = np.stack(np.meshgrid(indices, indices, indices, indexing="ij"))
    # coords shape [coord_num, 3]
    coords = np.transpose(coords.reshape([coords.shape[0], -1]))
    rand = np.random.RandomState(seed)
    ellipsoids = [
        (
            scale(rand.rand(), -dimension_size / 3, dimension_size / 3),
            scale(rand.rand(), -dimension_size / 3, dimension_size / 3),
            scale(rand.rand(), -dimension_size / 3, dimension_size / 3),
            scale(rand.rand(), min(1, dimension_size / 10), dimension_size / 6),
            scale(rand.rand(), min(1, dimension_size / 10), dimension_size / 6),
            scale(rand.rand(), min(1, dimension_size / 10), dimension_size / 6),
        )
        for _ in range(num)
    ]
    all_material = np.full((dimension_size**3,), material, dtype=np.int)
    for ellipsoid in ellipsoids:
        x_diff = coords[:, 0] - ellipsoid[0]
        y_diff = coords[:, 1] - ellipsoid[1]
        z_diff = coords[:, 2] - ellipsoid[2]
        all_material = np.where(
            x_diff**2 / ellipsoid[3] ** 2
            + y_diff**2 / ellipsoid[4] ** 2
            + z_diff**2 / ellipsoid[5] ** 2
            <= 1,
            0,
            all_material,
        )
    voxels[
        coords[:, 0] + center_voxel_offset,
        coords[:, 1] + center_voxel_offset,
        coords[:, 2] + center_voxel_offset,
    ] = all_material
    return voxels


def generate_cross(dimension_size, length_ratio=0.5, thickness_ratio=0.1, material=1):
    voxels = np.zeros([dimension_size, dimension_size, dimension_size], dtype=np.int)
    center_voxel_offset = dimension_size // 2
    indices = list(
        range(
            -center_voxel_offset,
            dimension_size - center_voxel_offset,
        )
    )
    coords = np.stack(np.meshgrid(indices, indices, indices, indexing="ij"))
    # coords shape [coord_num, 3]
    coords = np.transpose(coords.reshape([coords.shape[0], -1]))
    distance_1 = np.sqrt(coords[:, 1] ** 2 + ((coords[:, 0] - coords[:, 2]) ** 2) / 2)
    distance_2 = np.sqrt(coords[:, 1] ** 2 + ((coords[:, 0] + coords[:, 2]) ** 2) / 2)
    bar_1 = np.logical_and(
        distance_1 < max(dimension_size * thickness_ratio, 1),
        np.abs(coords[:, 0] + coords[:, 2]) < max(dimension_size * length_ratio, 1),
    )
    bar_2 = np.logical_and(
        distance_2 < max(dimension_size * thickness_ratio, 1),
        np.abs(-coords[:, 0] + coords[:, 2]) < max(dimension_size * length_ratio, 1),
    )
    bound = dimension_size * length_ratio * 0.5
    ends = (
        np.linalg.norm(coords - np.array([bound, 0, bound]), axis=1)
        < max(dimension_size * thickness_ratio, 1),
        np.linalg.norm(coords - np.array([bound, 0, -bound]), axis=1)
        < max(dimension_size * thickness_ratio, 1),
        np.linalg.norm(coords - np.array([-bound, 0, bound]), axis=1)
        < max(dimension_size * thickness_ratio, 1),
        np.linalg.norm(coords - np.array([-bound, 0, -bound]), axis=1)
        < max(dimension_size * thickness_ratio, 1),
    )
    all_material = np.where(
        np.any(np.stack((bar_1, bar_2) + ends, axis=0), axis=0),
        material,
        0,
    )
    voxels[
        coords[:, 0] + center_voxel_offset,
        coords[:, 1] + center_voxel_offset,
        coords[:, 2] + center_voxel_offset,
    ] = all_material
    return voxels
