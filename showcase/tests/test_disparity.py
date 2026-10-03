import unittest
import numpy as np
from cvportfolio.geometry import compute_disparity_ncc

class DisparityTest(unittest.TestCase):
    def test_identical_textured_views_have_zero_disparity(self):
        image=np.random.default_rng(42).random((16,20))
        disparity=compute_disparity_ncc(image,image,patch_size=4,max_disp=5)
        np.testing.assert_array_equal(disparity,0)
