import zlib
import cc3d
import numpy as np


def distance_traveled_of_com(start_com: np.ndarray, end_com: np.ndarray):
    # start_com and end_com shape: [voxel_num, 3]
    return np.linalg.norm(end_com[:2] - start_com[:2]).mean()


def distance_traveled(start_pos: np.ndarray, end_pos: np.ndarray):
    # start_pos and end_pos shape: [voxel_num, 3]
    return np.linalg.norm(
        np.array(start_pos)[:, :2] - np.array(end_pos)[:, :2], axis=1
    ).max()


def get_bounding_box_sizes(X: np.ndarray):
    occupied = X != 0
    x_occupied = [x for x in range(occupied.shape[0]) if np.any(occupied[x])]
    y_occupied = [y for y in range(occupied.shape[1]) if np.any(occupied[:, y])]
    z_occupied = [z for z in range(occupied.shape[2]) if np.any(occupied[:, :, z])]
    min_x = min(x_occupied)
    max_x = max(x_occupied) + 1
    min_y = min(y_occupied)
    max_y = max(y_occupied) + 1
    min_z = min(z_occupied)
    max_z = max(z_occupied) + 1
    return max_x - min_x, max_y - min_y, max_z - min_z


def get_volume(X: np.ndarray):
    return np.sum(X != 0)


def pad_voxels(X: np.ndarray):
    new_voxels = np.zeros(
        (X.shape[0] + 2, X.shape[1] + 2, X.shape[2] + 2), dtype=X.dtype
    )
    new_voxels[1:-1, 1:-1, 1:-1] = X
    return new_voxels


def get_surface_voxels(X: np.ndarray):
    if X.shape[0] > 2 and X.shape[1] > 2 and X.shape[2] > 2:
        padded_X = pad_voxels(X)
        coords = np.stack(
            np.meshgrid(
                list(range(1, padded_X.shape[0] - 1)),
                list(range(1, padded_X.shape[1] - 1)),
                list(range(1, padded_X.shape[2] - 1)),
                indexing="ij",
            )
        ).reshape(3, 1, -1)
        offsets = (
            np.array(
                [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
            )
            .transpose(1, 0)
            .reshape(3, -1, 1)
        )
        # shape [3, 6, X_space_size]
        neighbor_coords = coords + offsets
        # Note: includes voxels of a internal hole surface
        # shape [X_space_size]
        neighbor_is_empty = np.any(
            padded_X[neighbor_coords[0], neighbor_coords[1], neighbor_coords[2]] == 0,
            axis=0,
        )
        # shape [X_space_size]
        self_is_not_empty = padded_X[coords[0, 0], coords[1, 0], coords[2, 0]] != 0
        return np.sum(np.logical_and(self_is_not_empty, neighbor_is_empty))
    else:
        return get_volume(X)


def get_surface_area(X: np.ndarray):
    padded_X = pad_voxels(X)

    coords = np.stack(
        np.meshgrid(
            list(range(1, padded_X.shape[0] - 1)),
            list(range(1, padded_X.shape[1] - 1)),
            list(range(1, padded_X.shape[2] - 1)),
            indexing="ij",
        )
    ).reshape(3, 1, -1)
    offsets = (
        np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]])
        .transpose(1, 0)
        .reshape(3, -1, 1)
    )
    # shape [3, 6, X_space_size]
    neighbor_coords = coords + offsets
    # shape [X_space_size]
    neighbor_is_empty = (
        padded_X[neighbor_coords[0], neighbor_coords[1], neighbor_coords[2]] == 0,
    )
    # shape [1, X_space_size]
    self_is_not_empty = padded_X[coords[0], coords[1], coords[2]] != 0
    surfaces = np.sum(np.logical_and(self_is_not_empty, neighbor_is_empty))
    return surfaces


def get_section_num(X: np.ndarray):
    X = np.copy(X)
    X[X == 0] = X.max() + 1
    _labels, label_num = cc3d.connected_components(
        X, connectivity=6, return_N=True, out_dtype=np.uint64
    )
    return label_num


def get_reflection_symmetry(X: np.ndarray):
    # Note: this function is based on the assumption that
    # voxels X are symmetric with respect to the mid
    # cross section plane in each direction. Mid plane is
    # defined as the plane overlapping the mid point of
    # the representation space in each direction.

    # Eg: if we use GMM model, since the action space is
    # symmetric with respect to the mid cross section plane
    # we can use this metric to get the reflection symmetry.

    # just check non-zero voxel difference ratio in each direction
    # harmonic_mean(symmetric_voxels / left_voxels, symmetric_voxels / right_voxels)
    symmetric_scores = []
    for axis in range(3):
        axis_length = X.shape[axis]
        left_index = tuple(
            slice(0, X.shape[ax]) if ax != axis else slice(0, axis_length // 2)
            for ax in range(3)
        )
        left_section = X[left_index]
        right_index = tuple(
            slice(0, X.shape[ax])
            if ax != axis
            else slice(axis_length - axis_length // 2, X.shape[ax])
            for ax in range(3)
        )
        right_section = X[right_index]
        flipped_right_section = np.flip(right_section, axis=axis)
        left_voxels = np.sum(left_section != 0)
        right_voxels = np.sum(right_section != 0)
        symmetric_voxels = np.sum(
            np.logical_and(left_section == flipped_right_section, left_section != 0)
        )
        # equal to (2 * symmetric_voxels (in half a body)) / (total_voxels)
        symmetric_scores.append(
            2 / (left_voxels / symmetric_voxels + right_voxels / symmetric_voxels)
        )
    return np.mean(symmetric_scores)


def get_gzip_compressed_ratio(X: np.ndarray):
    voxel_bytes = X.astype(np.ubyte).tobytes()
    return len(zlib.compress(voxel_bytes)) / np.prod(X.shape)


def get_passive_material_ratio(X: np.ndarray):
    return np.sum(X == 1) / get_volume(X)
