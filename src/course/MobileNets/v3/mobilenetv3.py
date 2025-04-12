from enum import Enum
from typing import List

from torch import nn


def _make_divisible(channels: int, width_mult: float):
    """"""


class ConvNormActivation(nn.Sequential):
    """"""


class Conv2dNormActivation(ConvNormActivation):
    """"""


class SqueezeExcitation(nn.Module):
    """"""


class InterpolationMode(Enum):
    """"""
    

class ImageClassification(nn.Module):
    """"""


class InvertedResidualConfig:
    """"""


class InvertedResidual(nn.Module):
    """"""


class MobileNetV3(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: List[InvertedResidualConfig],
            num_classes=1000
    ) -> None:
        super().__init__()



if __name__ == '__main__':
    pass