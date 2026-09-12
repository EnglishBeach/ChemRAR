from collections.abc import Collection
import copy
from pathlib import Path
import typing

from rdkit import Chem as rd

T = typing.TypeVar("T")


def copy_dataclass(obj: T) -> T:
    new_obj = copy.copy(obj)

    if hasattr(new_obj, "__dict__"):
        for attr, value in list(new_obj.__dict__.items()):
            if isinstance(value, Collection):
                new_obj.__dict__[attr] = copy.copy(value)

    return new_obj
