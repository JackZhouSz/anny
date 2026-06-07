import unittest

import anny


class TestSomaRigVertexCount(unittest.TestCase):

    def test_soma_rig_soma_topology_vertex_count_matches_default_rig(self):
        default_model = anny.Anny(rig="default", topology="soma")
        soma_model = anny.Anny(rig="soma", topology="soma")
        self.assertEqual(soma_model.template_vertices.shape,
                         default_model.template_vertices.shape)

    def test_soma_rig_default_topology_vertex_count_matches_default_rig(self):
        default_model = anny.Anny(rig="default", topology="default")
        soma_model = anny.Anny(rig="soma", topology="default")
        self.assertEqual(soma_model.template_vertices.shape,
                         default_model.template_vertices.shape)

if __name__ == "__main__":
    unittest.main()
