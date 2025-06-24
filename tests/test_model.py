import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cbam import CBAM


class TestModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model = CBAM(in_channels=3)
        self.batch_size = 64
        self.input_shape = (3, 224, 224)  # Adjust based on your model

    def test_model_forward_pass(self):
        """Test if model can perform forward pass."""
        # Create dummy input
        dummy_input = torch.randn(self.batch_size, *self.input_shape)

        # Forward pass
        output = self.model(dummy_input)

        # Check output shape
        expected_shape = (self.batch_size, 29)  # num_classes
        self.assertEqual(output.shape, expected_shape)

    def test_model_backward_pass(self):
        """Test if model can perform backward pass."""
        dummy_input = torch.randn(self.batch_size, *self.input_shape)
        dummy_target = torch.randint(0, 10, (self.batch_size,))

        # Forward pass
        output = self.model(dummy_input)

        # Loss calculation
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, dummy_target)

        # Backward pass
        loss.backward()

        # Check if gradients are computed
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_model_parameters_count(self):
        """Test if model has expected number of parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 0)

    def test_model_output_range(self):
        """Test if model output is in expected range."""
        dummy_input = torch.randn(self.batch_size, *self.input_shape)
        output = self.model(dummy_input)

        # For classification, check if output can be converted to probabilities
        probabilities = torch.softmax(output, dim=1)
        self.assertTrue(
            torch.allclose(probabilities.sum(dim=1), torch.ones(self.batch_size))
        )


if __name__ == "__main__":
    unittest.main()
