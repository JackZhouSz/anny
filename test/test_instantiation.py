import unittest

import anny


class TestInstantiation(unittest.TestCase):

    def test_instantiation(self):
        for rig in ["default", "soma"]:
            for topology in ["default", "soma", "smplx"]:
                with self.subTest(rig=rig, topology=topology):
                    model = anny.Anny(rig=rig, topology=topology)
                    self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
