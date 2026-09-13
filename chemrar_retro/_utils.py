import collections.abc as collections
import copy
import typing

from aizynthfinder.context.config import Configuration  # noqa: F401
from aizynthfinder.utils.type_utils import StrDict  # noqa: F401

T = typing.TypeVar("T")


def copy_dataclass(obj: T) -> T:
    new_obj = copy.copy(obj)

    if hasattr(new_obj, "__dict__"):
        for attr, value in list(new_obj.__dict__.items()):
            if isinstance(value, collections.Collection):
                new_obj.__dict__[attr] = copy.copy(value)

    return new_obj
