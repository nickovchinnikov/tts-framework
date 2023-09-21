import unittest
import torch


from model.attention.conformer_block import ConformerBlock


# @todo: it's one of the most important component test
# But it's too complicated to cover it from the first glance.
# You need to come back here when acoustic model is ready!
class TestConformerBlock(unittest.TestCase):
    def test_initialization(self):
        """
        Test for successful creation of ConformerBlock instance.
        """
        model = ConformerBlock(
            d_model=12,
            n_head=6,
            d_k=12,
            d_v=12,
            kernel_size_conv_mod=3,
            embedding_dim=12,
            dropout=0.1,
            with_ff=False,
        )
        self.assertIsInstance(model, ConformerBlock)

    def test_with_ff_flag(self):
        """
        Test for correct response based on `with_ff` flag during initialization.
        """
        model = ConformerBlock(
            d_model=12,
            n_head=6,
            d_k=12,
            d_v=12,
            kernel_size_conv_mod=3,
            embedding_dim=12,
            dropout=0.1,
            with_ff=True,
        )
        self.assertTrue(hasattr(model, "ff"))

        model = ConformerBlock(
            d_model=20,
            n_head=5,
            d_k=16,
            d_v=16,
            kernel_size_conv_mod=5,
            embedding_dim=20,
            dropout=0.4,
            with_ff=False,
        )
        self.assertFalse(hasattr(model, "ff"))


if __name__ == "__main__":
    unittest.main()
