import unittest

from model.attention.conformer import Conformer


# @todo: it's one of the most important component test
# But it's too complicated to cover it from the first glance.
# You need to come back here when acoustic model is ready!
class TestConformer(unittest.TestCase):
    def test_initialization(self):
        """
        Test that a Conformer instance is correctly initialized.
        """
        model = Conformer(
            dim=100,
            n_layers=10,
            n_heads=5,
            embedding_dim=100,
            p_dropout=0.1,
            kernel_size_conv_mod=5,
            with_ff=True,
        )
        self.assertIsInstance(model, Conformer)


if __name__ == "__main__":
    unittest.main()
