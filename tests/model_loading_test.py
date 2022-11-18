import unittest

from ikflow.model_loading import model_filename

import torch

torch.manual_seed(0)


class ModelLoadingTest(unittest.TestCase):
    def test_model_filename(self):
        url = "https://storage.googleapis.com/ikflow_models/atlas_desert-sweep-6.pkl"
        filename = model_filename(url)
        self.assertEqual(filename, "atlas_desert-sweep-6.pkl")


if __name__ == "__main__":
    unittest.main()
