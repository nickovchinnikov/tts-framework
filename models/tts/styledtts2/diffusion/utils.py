from functools import reduce
from math import ceil, floor, log2
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import torch
from typing_extensions import TypeGuard

T = TypeVar("T")


def exists(val: Optional[T]) -> TypeGuard[T]:
    r"""Check if a value is not None.

    Args:
        val: The value to check.

    Returns:
        True if the value is not None, False otherwise.
    """
    return val is not None


def iff(condition: bool, value: T) -> Optional[T]:
    r"""Return the value if the condition is True, None otherwise.

    Args:
        condition: The condition to check.
        value: The value to return if the condition is True.

    Returns:
        The value if the condition is True, None otherwise.
    """
    return value if condition else None


def is_sequence(obj: T) -> TypeGuard[Union[list, tuple]]:
    r"""Check if an object is a list or a tuple.

    Args:
        obj: The object to check.

    Returns:
        True if the object is a list or a tuple, False otherwise.
    """
    return isinstance(obj, (list, tuple))


def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    r"""Return the value if it exists, otherwise return the default value.

    Args:
        val: The value to check.
        d: The default value to return if the value does not exist.

    Returns:
        The value if it exists, otherwise the default value.
    """
    if exists(val):
        return val
    if callable(d):
        return d()
    else:
        return d


def to_list(val: Union[T, Sequence[T]]) -> List[T]:
    r"""Convert a value or a sequence of values to a list.

    Args:
        val: The value or sequence of values to convert.

    Returns:
        The value or sequence of values as a list.
    """
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, list):
        return val
    if isinstance(val, Sequence):
        return list(val)
    return [val]


def prod(vals: Sequence[int]) -> int:
    r"""Calculate the product of a sequence of integers.

    Args:
        vals: The sequence of integers.

    Returns:
        The product of the sequence of integers.
    """
    return reduce(lambda x, y: x * y, vals)


def closest_power_2(x: float) -> int:
    r"""Find the closest power of 2 to a given number.

    Args:
        x: The number to find the closest power of 2 to.

    Returns:
        The closest power of 2 to the given number.
    """
    exponent = log2(x)
    distance_fn = lambda z: abs(x - 2 ** z)
    exponent_closest = min((floor(exponent), ceil(exponent)), key=distance_fn)
    return 2 ** int(exponent_closest)

def rand_bool(shape: Tuple[int, ...], proba: float):
    r"""Generate a tensor of random booleans.

    Args:
        shape: The shape of the tensor.
        proba: The probability of a True value.
        device: The device to create the tensor on.

    Returns:
        A tensor of random booleans.
    """
    if proba == 1:
        return torch.ones(shape, dtype=torch.bool)
    elif proba == 0:
        return torch.zeros(shape, dtype=torch.bool)
    else:
        return torch.bernoulli(torch.full(shape, proba)).to(dtype=torch.bool)


"""
Kwargs Utils
"""


def group_dict_by_prefix(prefix: str, d: Dict) -> Tuple[Dict, Dict]:
    r"""Group a dictionary by keys that start with a given prefix.

    Args:
        prefix: The prefix to group by.
        d: The dictionary to group.

    Returns:
        A tuple of two dictionaries: one with keys that start with the prefix, and one with keys that do not.
    """
    return_dicts: Tuple[Dict, Dict] = ({}, {})
    for key in d:
        no_prefix = int(not key.startswith(prefix))
        return_dicts[no_prefix][key] = d[key]
    return return_dicts


def groupby(prefix: str, d: Dict, keep_prefix: bool = False) -> Tuple[Dict, Dict]:
    r"""Group a dictionary by keys that start with a given prefix and optionally remove the prefix from the keys.

    Args:
        prefix: The prefix to group by.
        d: The dictionary to group.
        keep_prefix: Whether to keep the prefix in the keys.

    Returns:
        A tuple of two dictionaries: one with keys that start with the prefix, and one with keys that do not.
    """
    kwargs_with_prefix, kwargs = group_dict_by_prefix(prefix, d)
    if keep_prefix:
        return kwargs_with_prefix, kwargs
    kwargs_no_prefix = {k[len(prefix) :]: v for k, v in kwargs_with_prefix.items()}
    return kwargs_no_prefix, kwargs


def prefix_dict(prefix: str, d: Dict) -> Dict:
    r"""Add a prefix to all keys in a dictionary.

    Args:
        prefix: The prefix to add.
        d: The dictionary to modify.

    Returns:
        The modified dictionary with the prefix added to all keys.
    """
    return {prefix + str(k): v for k, v in d.items()}
