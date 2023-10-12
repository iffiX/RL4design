import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clip(x):
    return (np.clip(x, -2, 2) + 2) / 4


def clip1(x):
    return (np.clip(x, -1, 1) + 1) / 2


def normalize(x, mode="clip"):
    if mode == "clip":
        return clip(x)
    elif mode == "clip1":
        return clip1(x)
    elif mode == "sigmoid":
        return sigmoid(x)
    elif mode == "none":
        return x
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")
