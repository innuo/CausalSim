from unittest import TestCase, main
import os
from structure import CausalStructure
from datahandler import RawData

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestCausalStructure(TestCase):
    def test_learn_structure(self):
        r = RawData(os.path.join(THIS_DIR, '../data/5d.csv'))
        cs = CausalStructure(r.variables)
        cs.learn_structure(r)
        self.assertTrue(cs.dag.has_edge("Price", "VolumeBought"))
        self.assertTrue(cs.dag.has_edge("Product", "VolumeBought"))
        self.assertTrue(cs.dag.has_edge("Channel", "VolumeBought"))
        self.assertTrue(cs.dag.has_edge("PersonType", "VolumeBought"))
        self.assertTrue(cs.dag.has_edge("Product", "Price"))
        self.assertTrue(cs.dag.has_edge("Channel", "Price"))


    def test_update_structure(self):
        self.assertTrue(True)

    def test_merge(self):
        self.assertTrue(True)

if __name__ == '__main__':
    main()