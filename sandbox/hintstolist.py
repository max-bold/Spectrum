from enum import Enum
from turtle import shape
from typing import Literal, get_args, TypeAlias

shapes = ["flat", "cosine", "gaussian", "triangular"]


class Shapes(Enum):
    flat = 1
    cosine = 2
    gaussian = 3
    triangular = 4


B = Enum("B",["flat", "cosine", "gaussian", "triangular"])


class A:

    def __init__(self) -> None:
        self.b: B = B.cosine


print(B.value)
