import unittest
from kl_tools import utils

class TestUtils(unittest.TestCase):
    def test_read_yaml(self):
        # TODO: Write test cases for read_yaml function
        pass

    def test_build_map_grid(self):
        '''
        Check that our grid indexing is correct / as expected. We use rectangular grids to be sure we are getting the axes right

        As the origin is always at the center of the grid, pixel centers will land on integers for odd grid sizes, and half-integers for even grid sizes. We will test both cases
        '''

        # even-sized grid
        Nx, Ny = (10, 20)
        X, Y = utils.build_map_grid(Nx, Ny)

        self.assertEqual((X[5, 7], Y[5,7]), (0.5, -2.5))

        # odd-sized grid
        Nx, Ny = (15, 7)
        X, Y = utils.build_map_grid(Nx, Ny)

        self.assertEqual((X[9, 2], Y[9,2]), (2, -1))

        # mixed grid
        Nx, Ny = (10, 13)
        X, Y = utils.build_map_grid(Nx, Ny)

        self.assertEqual((X[4, 8], Y[4,8]), (-0.5, 2))

        return

if __name__ == '__main__':
    unittest.main()
