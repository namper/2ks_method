import unittest
from main import compute_initial_nodal_points


class TestNodal(unittest.TestCase):

    def test_initial_nodal_points(self):
        self.assertTrue(
            (compute_initial_nodal_points(1) == [0, 0.5, 1]).all()
        )

    def test_initial_nodal_points_2(self):
        self.assertTrue(
            (
                compute_initial_nodal_points(2) == [0, 1/4, 2/4, 3/4, 1]
            ).all()
        )


if __name__ == '__main__':
    unittest.main()
