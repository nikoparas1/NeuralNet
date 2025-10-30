import numpy as np
from enum import Enum
from typing import Callable


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(z.dtype)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return z * (1 - z)


def he_std(fan_in, fan_out):
    return np.sqrt(2.0 / fan_in)


def glorot_std(fan_in, fan_out):
    return np.sqrt(2.0 / (fan_in + fan_out))


class Activation(Enum):
    relu = ("relu", relu, relu_derivative, he_std)
    sigmoid = ("sigmoid", sigmoid, sigmoid_derivative, glorot_std)

    def __init__(
        self, label: str, act_fn: Callable, act_dfn: Callable, init_scale: Callable
    ):
        self.label = label
        self.act_fn = act_fn
        self.act_dfn = act_dfn
        self.init_scale = init_scale
