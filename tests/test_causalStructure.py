from unittest import TestCase, main
import os
import pandas as pd
from structure import CausalStructure
from datahandler import DataUtils

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestCausalStructure(TestCase):
    def test_learn_structure(self):
        df = pd.read_csv(os.path.join(THIS_DIR, '../data/5d.csv'))
        r = DataUtils([df])
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