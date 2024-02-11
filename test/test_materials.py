import unittest
import numpy as np

from secan.material import Concrete, SteelIdeal, SteelHardening

class TestMaterials(unittest.TestCase):

    def test_concrete(self):
        c = Concrete(40e6)
        self.assertEqual(c.get_stiff(-1.5e-3), 10.0e9) 
        self.assertEqual(c.get_stress(-1.5e-3), -37.5e6) 

    def test_steel_ideal(self):
        s = SteelIdeal(young=200e9, fy=400e6)
        self.assertEqual(s.get_stiff(1e-3), 200e9)
        self.assertEqual(s.get_stress(1e-3), 200e6)
        self.assertEqual(s.get_stress(10e-3), 400e6)
        self.assertEqual(s.get_stress(-10e-3), -400e6)
        
    def test_steel_hardening(self):
        s = SteelHardening(young=200e9, fy=400e6, ft=1200e6, ultimate_strain=35e-3)
        
        self.assertEqual(s.get_stiff(1e-3), 200e9)
        self.assertEqual(s.get_stiff(5e-3), 24242424242.42424)
        self.assertEqual(s.get_stiff(36e-3), 0)
        self.assertEqual(s.get_stiff(-1e-3), 200e9)
        self.assertEqual(s.get_stiff(-5e-3), 24242424242.42424)
        self.assertEqual(s.get_stiff(-36e-3), 0)
        
        self.assertEqual(s.get_stress(1e-3), 200e6)
        self.assertEqual(s.get_stress(10e-3), 593939393.939394)
        self.assertEqual(s.get_stress(25e-3), 957575757.5757575)
        self.assertEqual(s.get_stress(36e-3), 0)
        self.assertEqual(s.get_stress(-1e-3), -200e6)
        self.assertEqual(s.get_stress(-10e-3), -593939393.939394)
        self.assertEqual(s.get_stress(-25e-3), -957575757.5757575)
        self.assertEqual(s.get_stress(-36e-3), 0)

if __name__ == '__main__':
    unittest.main()
