import unittest
import numpy as np

from secan.section import Section
from secan.material import Concrete, SteelIdeal, SteelHardening
from secan.geometry import RectSection, Rebar, Tendon
from secan.exceptions import SectionUnstableError, ConvergenceError

class TestFlavioNeto(unittest.TestCase):
    """
        The tests in this section are from the book "Concreto Estrutural Avançado - Análise de seções transversais sob flexão normal composta" by Professor Flávio Mendes Neto.
    """
    
    def setUp(self):
        self.delta = 1e-4
    
        self.conc = Concrete(0.85*30e6/1.4)
        self.steel = SteelIdeal(210e9, 500e6/1.15)
        self.r = RectSection(0.2, 0.32, self.conc, (0, 0), n_discret=300)
        self.s = Section([self.r])
        
        xy_pairs = [(x,y) for x in [-.07,.07] for y in [-0.11, 0, 0.11]]
        for xy in xy_pairs:
            self.s.addSingleRebar(diameter = 0.01,
                                  material = self.steel,
                                  position = xy)
    
    def get_rebar_normal(self, e0,k):
        return np.array([s.get_normal_resistance(e0=e0, k=k, center=self.s.centroid[1])
                           for s in self.s.section if isinstance(s, Rebar) ]).sum()
    
    def get_rebar_moment(self, e0,k):                       
        return np.array([s.get_moment_resistance(e0=e0, k=k, center=self.s.centroid[1])
                           for s in self.s.section if isinstance(s, Rebar) ]).sum()
                           
    def get_rec_normal(self, e0,k):
        return np.array([s.get_normal_resistance(e0=e0, k=k, center=self.s.centroid[1])
                           for s in self.s.section if isinstance(s, RectSection) ]).sum()
    
    def get_rec_moment(self, e0,k):                       
        return np.array([s.get_moment_resistance(e0=e0, k=k, center=self.s.centroid[1])
                           for s in self.s.section if isinstance(s, RectSection) ]).sum()
                           
    def get_rebar_stiff(self, e0, k):
        return np.array([s.get_stiffness(e0=e0, k=k, center=self.s.centroid[1])
                           for s in self.s.section if isinstance(s, Rebar) ]).sum(axis=0)
    
    def get_rec_stiff(self, e0, k):
        return np.array([s.get_stiffness(e0=e0, k=k, center=self.s.centroid[1])
                           for s in self.s.section if isinstance(s, RectSection) ]).sum(axis=0)
    
    def test_example_3_4(self):
        # 3.4.1
        self.assertEqual(self.r.area, 0.064)
                
        self.assertAlmostEqual(self.conc.get_stress(-3.5e-3)/1e6, -18.21428, delta=self.delta)
        self.assertAlmostEqual(self.steel.get_stress(10e-3)/1e6, 434.7826, delta=self.delta)
        self.assertAlmostEqual(self.steel.yield_strain, 2.0704e-3)
        
        self.assertAlmostEqual(self.s.area_rebar, 4.7123e-4, delta=self.delta)
        
        # 3.4.2
        self.assertAlmostEqual(self.get_rebar_normal(e0=-2e-3, k=0)/1e6, -0.1979, delta=self.delta)
        self.assertAlmostEqual(self.get_rec_normal(e0=-2e-3, k=0)/1e6, -1.1657, delta=self.delta)
        
        self.assertAlmostEqual(self.s.get_normal_res(e0=-2e-3, k=0)/1e6, -1.3636, delta=self.delta)
        self.assertAlmostEqual(self.s.get_moment_res(e0=-2e-3, k=0), 0)
        
        # 3.4.3
        self.assertAlmostEqual(self.s.get_normal_res(e0=10e-3, k=0)/1e6, 0.2049, delta=self.delta)
        self.assertAlmostEqual(self.s.get_moment_res(e0=10e-3, k=0), 0)
        
        # 3.4.4
        self.assertAlmostEqual(self.get_rebar_normal(e0=-0.1990e-3, k=20.6310e-3)/1e6, -0.0066, delta=self.delta)
        self.assertAlmostEqual(self.get_rebar_moment(e0=-0.1990e-3, k=20.6310e-3)/1e6, 0.0150, delta=self.delta)
        self.assertAlmostEqual(self.get_rec_normal(e0=-0.1990e-3, k=20.6310e-3)/1e6, -0.5003, delta=self.delta)
        self.assertAlmostEqual(self.get_rec_moment(e0=-0.1990e-3, k=20.6310e-3)/1e6, 0.0447, delta=self.delta)
        
        self.assertAlmostEqual(self.s.get_normal_res(e0=-0.1990e-3, k=20.6310e-3)/1e6, -0.5069, delta=self.delta)
        self.assertAlmostEqual(self.s.get_moment_res(e0=-0.1990e-3, k=20.6310e-3)/1e6, 0.0598, delta=self.delta)
        
        # 3.4.5
        self.assertAlmostEqual(self.get_rebar_normal(e0=-2e-3, k=6.25e-3)/1e6, -0.1776, delta=self.delta)
        self.assertAlmostEqual(self.get_rebar_moment(e0=-2e-3, k=6.25e-3)/1e6, 0.0028, delta=self.delta)
        self.assertAlmostEqual(self.get_rec_normal(e0=-2e-3, k=6.25e-3)/1e6, -1.1171, delta=self.delta)
        self.assertAlmostEqual(self.get_rec_moment(e0=-2e-3, k=6.25e-3)/1e6, 0.0058, delta=self.delta)
        
        self.assertAlmostEqual(self.s.get_normal_res(e0=-2e-3, k=6.25e-3)/1e6, -1.2947, delta=self.delta)
        self.assertAlmostEqual(self.s.get_moment_res(e0=-2e-3, k=6.25e-3)/1e6, 0.0086, delta=self.delta)
        
        # 3.4.6
        self.assertAlmostEqual(self.get_rebar_normal(e0=-0.5e-3, k=9.3750e-3)/1e6, -0.0495, delta=self.delta)
        self.assertAlmostEqual(self.get_rebar_moment(e0=-0.5e-3, k=9.3750e-3)/1e6, 0.0075, delta=self.delta)
        self.assertAlmostEqual(self.get_rec_normal(e0=-0.5e-3, k=9.3750e-3)/1e6, -0.5181, delta=self.delta)
        self.assertAlmostEqual(self.get_rec_moment(e0=-0.5e-3, k=9.3750e-3)/1e6, 0.0414, delta=self.delta)
        
        self.assertAlmostEqual(self.s.get_normal_res(e0=-0.5e-3, k=9.3750e-3)/1e6, -0.5676, delta=self.delta)
        self.assertAlmostEqual(self.s.get_moment_res(e0=-0.5e-3, k=9.3750e-3)/1e6, 0.0489, delta=self.delta)
        
    def test_example_4_5(self):
        self.s.tolerance_check_section = 1e-8
    
        # 4.5.1
        rebar_stiff = self.get_rebar_stiff(0, 0)        
        self.assertAlmostEqual(rebar_stiff[0][0]/1e7, 210*4.71239e-2, delta=self.delta)
        self.assertEqual(rebar_stiff[0][1], 0,)
        self.assertAlmostEqual(rebar_stiff[1][1]/1e5, 7.9828, delta=self.delta)
        
        rec_stiff = self.get_rec_stiff(0, 0)
        self.assertAlmostEqual(rec_stiff[0][0]/1e9, 1.1657, delta=self.delta)
        self.assertAlmostEqual(rec_stiff[0][1], 0, delta=self.delta)
        self.assertAlmostEqual(rec_stiff[1][1]/1e6, 9.9474, delta=self.delta)
        
        stiff = self.s.get_stiff(0, 0)
        self.assertAlmostEqual(stiff[0][0]/1e9, 1.2647, delta=self.delta)
        self.assertAlmostEqual(stiff[0][1], 0, delta=self.delta)
        self.assertAlmostEqual(stiff[1][1]/1e7, 1.0746, delta=self.delta)
        self.assertAlmostEqual(np.linalg.det(stiff)/1e16, 1.3590, delta=self.delta)
        
        target_normal=-0.8160e6
        target_moment=0.0373e6
        e0, k = self.s.check_section(target_normal, target_moment, n_ite=1000)
        
        self.assertAlmostEqual(e0*1e3, -0.9028, delta=self.delta)
        self.assertAlmostEqual(k*1e3, 5.9864, delta=0.0006) ######
        
        self.assertAlmostEqual(self.s.get_normal_res(e0, k), target_normal, delta=self.delta)
        self.assertAlmostEqual(self.s.get_moment_res(e0, k), target_moment, delta=self.delta)
        
        # 4.5.2
        with self.assertRaises(SectionUnstableError):
            self.s.check_section(target_normal=0, target_moment=0.0373e6, n_ite=1000)
        
        # 4.5.3
        with self.assertRaises(SectionUnstableError):
            self.s.check_section(target_normal=-1.1657e6, target_moment=0.0373e6, n_ite=1000)
        
        # 4.5.4
        e0, k = self.s.check_section(target_normal=-0.8160e6, target_moment=0, n_ite=1000)
        self.assertAlmostEqual(e0*1e3, -0.7884934594071175, delta=self.delta)
        self.assertAlmostEqual(k*1e3, 0, delta=self.delta)
        
        # 4.5.5
        with self.assertRaises(SectionUnstableError):
            self.s.check_section(target_normal=-0.8160e6, target_moment=-0.0560e6, n_ite=1000)
        
        # 4.5.6
        target_normal=0.1166e6
        target_moment=0.0075e6
        e0, k = self.s.check_section(target_normal, target_moment, n_ite=10)
        
        self.assertAlmostEqual(e0*1e3, 1.369521, delta=0.009) ######
        self.assertAlmostEqual(k*1e3, 9.759756, delta=0.09) ######
        
        self.assertAlmostEqual(self.s.get_normal_res(e0, k), target_normal, delta=self.delta)
        self.assertAlmostEqual(self.s.get_moment_res(e0, k), target_moment, delta=self.delta)
        
        self.assertAlmostEqual(self.get_rebar_normal(e0, k)/1e7, 0.0123, delta=self.delta)
        self.assertAlmostEqual(self.get_rebar_moment(e0, k)/1e5, 0.0644, delta=self.delta)
        
        self.assertAlmostEqual(self.get_rec_normal(e0, k)/1e5, 0.0666, delta=0.15) ######
        self.assertAlmostEqual(self.get_rec_moment(e0, k)/1e5, 0.0102, delta=0.0005) ######
        
        rebar_stiff = self.get_rebar_stiff(e0, k)        
        self.assertAlmostEqual(rebar_stiff[0][0]/1e9, 0.0660, delta=self.delta)
        self.assertAlmostEqual(rebar_stiff[0][1]/1e8, -0.0363, delta=self.delta) ######
        self.assertAlmostEqual(rebar_stiff[1][1]/1e7, 0.0399, delta=self.delta)
        
        rec_stiff = self.get_rec_stiff(e0, k)
        self.assertAlmostEqual(rec_stiff[0][0]/1e9, 0.0682, delta=0.003) ######
        self.assertAlmostEqual(rec_stiff[0][1]/1e9, -0.0102, delta=0.0004) ######
        self.assertAlmostEqual(rec_stiff[1][1]/1e8, 0.0154, delta=0.0004) ######

if __name__ == '__main__':
    unittest.main()
