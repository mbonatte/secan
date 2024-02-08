import unittest
import numpy as np

from secan.material import Concrete, SteelIdeal, SteelHardening
from secan.geometry import RectSection, Rebar, Tendon

class TestGeometries(unittest.TestCase):

    def test_rectangle(self):
        concrete = Concrete(40e6)
        rect = RectSection(width     = 0.2,
                           height    = 0.6,
                           material  = concrete,
                           center    = (0, 0.3),
                           rotation  = 0,
                           n_discret = 200
                           )
        
        self.assertEqual(rect.area, 0.12) 
        self.assertEqual(rect.boundary, [(-0.1, 0.0), (0.1, 0.6)])
        
        self.assertAlmostEqual(rect.get_stiffness(0, 0, 0)[0,0], 4.8e9, delta=1e-6)
        self.assertAlmostEqual(rect.get_stiffness(0, 0, 0)[0,1], -1.44e9, delta=1e-6)
        self.assertAlmostEqual(rect.get_stiffness(0, 0, 0)[1,0], -1.44e9, delta=1e-6)
        self.assertAlmostEqual(rect.get_stiffness(0, 0, 0)[1,1], 5.759964e8, delta=1e-6)
        
        self.assertAlmostEqual(rect.get_stiffness(-0.1, 0.5, 0.2)[0,0], 1.5e7, delta=1e-6)
        self.assertAlmostEqual(rect.get_stiffness(-0.1, 0.5, 0.2)[0,1], 2977500)
        self.assertAlmostEqual(rect.get_stiffness(-0.1, 0.5, 0.2)[1,0], 2977500)
        self.assertAlmostEqual(rect.get_stiffness(-0.1, 0.5, 0.2)[1,1], 591033.75)
        
        self.assertEqual(rect.get_normal_resistance(e0=0.001, k=0.02, center=0.3), -1133312.9999999998)
        self.assertEqual(rect.get_normal_resistance(e0=0.001, k=0.02, center=0), -1133313.0)
        
        self.assertEqual(rect.get_moment_resistance(e0=0.001, k=0.02, center=0.3), 172496.04929999996)
        self.assertEqual(rect.get_moment_resistance(e0=0.001, k=0.02, center=0), 172496.04929999996)

    def test_rebar(self):
        steel = SteelIdeal(200e9, 400e6)
        rebar = Rebar(diameter=0.012, material=steel, center=(0.1, -0.2))
        
        self.assertEqual(rebar.area, 0.00011309733552923255) 
        self.assertEqual(rebar.boundary, [(0.1, -0.2)])
        
        self.assertAlmostEqual(rebar.get_stiffness(0, 0, 0)[0,0], 22619467.10584651)
        self.assertAlmostEqual(rebar.get_stiffness(0, 0, 0)[0,1], 4523893.4211693)
        self.assertAlmostEqual(rebar.get_stiffness(0, 0, 0)[1,0], 4523893.4211693)
        self.assertAlmostEqual(rebar.get_stiffness(0, 0, 0)[1,1], 904778.68423386)
        
        self.assertAlmostEqual(rebar.get_stiffness(0.001, 0.002, 0.1)[0,0], 22619467.10584651)
        self.assertAlmostEqual(rebar.get_stiffness(0.001, 0.002, 0.1)[0,1], 6785840.13175395)
        self.assertAlmostEqual(rebar.get_stiffness(0.001, 0.002, 0.1)[1,0], 6785840.13175395)
        self.assertAlmostEqual(rebar.get_stiffness(0.001, 0.002, 0.1)[1,1], 2035752.03952619)
        
        self.assertEqual(rebar.get_normal_resistance(e0=0.0001, k=0.002, center=0.3), 24881.413816431163)
        self.assertEqual(rebar.get_normal_resistance(e0=0.001, k=0.02, center=0), 45238.93421169302)
        
        self.assertEqual(rebar.get_moment_resistance(e0=0.0001, k=0.002, center=0.3), 12440.706908215581)
        self.assertEqual(rebar.get_moment_resistance(e0=0.001, k=0.02, center=0),  9047.786842338604)
        
    def test_tendon(self):
        steel = SteelHardening(young=200e9, fy=400e6, ft=1200e6, ultimate_strain=35e-3)
        tendon = Tendon(diameter=0.012, material=steel, initial_strain=1e-3, center=(0.1, -0.2), strain_ULS=10e-3)
        
        self.assertEqual(tendon.area, 0.00011309733552923255) 
        self.assertEqual(tendon.boundary, [(0.1, -0.2)]) 
        
        self.assertEqual(tendon.get_normal_resistance(e0=0.0001, k=0.002, center=0.3), 45513.109570551766)
        self.assertEqual(tendon.get_normal_resistance(e0=0.001, k=0.02, center=0), 56205.94856604285)
        
        self.assertEqual(tendon.get_moment_resistance(e0=0.0001, k=0.002, center=0.3), 22756.554785275883)
        self.assertEqual(tendon.get_moment_resistance(e0=0.001, k=0.02, center=0), 11241.18971320857)

if __name__ == '__main__':
    unittest.main()
