from unittest import TestCase, main
import os
import pandas as pd
from structure import CausalStructure
from datahandler import DataSet

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestCausalStructure(TestCase):
    def test_learn_structure(self):
        df = pd.read_csv(os.path.join(THIS_DIR, '../data/5d.csv'))
        r = DataSet([df])
        cs = CausalStructure(r.variable_names)
        cs.learn_structure(r)
        self.assertTrue(cs.dag.has_edge("Price", "VolumeBought"))
        self.assertTrue(cs.dag.has_edge("Product", "VolumeBought"))
        self.assertTrue(cs.dag.has_edge("Channel", "VolumeBought"))
        self.assertTrue(cs.dag.has_edge("PersonType", "VolumeBought"))
        self.assertTrue(cs.dag.has_edge("Product", "Price"))
        self.assertTrue(cs.dag.has_edge("Channel", "Price"))


if __name__ == '__main__':
    main()