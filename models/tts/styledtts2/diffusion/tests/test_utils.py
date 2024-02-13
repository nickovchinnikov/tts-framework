import unittest

import torch

from models.tts.styledtts2.diffusion.utils import (
    closest_power_2,
    default,
    exists,
    group_dict_by_prefix,
    groupby,
    iff,
    is_sequence,
    prefix_dict,
    prod,
    rand_bool,
    to_list,
)


class TestUtils(unittest.TestCase):
    def test_exists(self):
        self.assertTrue(exists(1))
        self.assertFalse(exists(None))

    def test_iff(self):
        self.assertEqual(iff(True, "value"), "value")
        self.assertEqual(iff(False, "value"), None)

    def test_is_sequence(self):
        self.assertTrue(is_sequence([1, 2, 3]))
        self.assertTrue(is_sequence((1, 2, 3)))
        self.assertFalse(is_sequence(123))

    def test_default(self):
        self.assertEqual(default(None, "default"), "default")
        self.assertEqual(default("value", "default"), "value")
        self.assertEqual(default(None, lambda: "default"), "default")

    def test_to_list(self):
        self.assertEqual(to_list((1, 2, 3)), [1, 2, 3])
        self.assertEqual(to_list([1, 2, 3]), [1, 2, 3])
        self.assertEqual(to_list(1), [1])

    def test_prod(self):
        self.assertEqual(prod([1, 2, 3, 4]), 24)

    def test_closest_power_2(self):
        self.assertEqual(closest_power_2(6), 4)
        self.assertEqual(closest_power_2(9), 8)

    def test_rand_bool(self):
        shape = (3, 3)
        tensor = rand_bool(shape, 0.5)
        self.assertEqual(tensor.shape, shape)
        self.assertTrue(tensor.dtype == torch.bool)

    def test_group_dict_by_prefix(self):
        d = {"prefix_key1": 1, "prefix_key2": 2, "key3": 3}
        with_prefix, without_prefix = group_dict_by_prefix("prefix_", d)
        self.assertEqual(with_prefix, {"prefix_key1": 1, "prefix_key2": 2})
        self.assertEqual(without_prefix, {"key3": 3})

    def test_groupby(self):
        d = {"prefix_key1": 1, "prefix_key2": 2, "key3": 3}
        with_prefix, without_prefix = groupby("prefix_", d)
        self.assertEqual(with_prefix, {"key1": 1, "key2": 2})
        self.assertEqual(without_prefix, {"key3": 3})

    def test_prefix_dict(self):
        d = {"key1": 1, "key2": 2, "key3": 3}
        prefixed = prefix_dict("prefix_", d)
        self.assertEqual(prefixed, {"prefix_key1": 1, "prefix_key2": 2, "prefix_key3": 3})

if __name__ == "__main__":
    unittest.main()
