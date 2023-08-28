import unittest
import numpy as np
from secan.section import Section
from secan.material import Concrete, SteelIdeal, SteelHardening
from secan.geometry import RectSection, Rebar, Tendon

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
        self.assertEqual(s.get_stress(1e-3), 200e6)
        self.assertEqual(s.get_stress(10e-3), 593939393.939394)
        self.assertEqual(s.get_stress(25e-3), 957575757.5757575)
        self.assertEqual(s.get_stress(-10e-3), -400e6)

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
        
        self.assertEqual(rect.get_normal_resistance(e0=0.001, k=0.02, center=0.3), -1133312.9999999998)
        self.assertEqual(rect.get_normal_resistance(e0=0.001, k=0.02, center=0), -1133313.0)
        
        self.assertEqual(rect.get_moment_resistance(e0=0.001, k=0.02, center=0.3), 172496.04929999996)
        self.assertEqual(rect.get_moment_resistance(e0=0.001, k=0.02, center=0), 172496.04929999996)

    def test_rebar(self):
        steel = SteelIdeal(200e9, 400e6)
        rebar = Rebar(diameter=0.012, material=steel, center=(0.1, -0.2))
        
        self.assertEqual(rebar.area, 0.00011309733552923255) 
        self.assertEqual(rebar.boundary, [(0.1, -0.2)]) 
        
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
        
class TestSections(unittest.TestCase):
    
    def setUp(self):
        self.conc = Concrete(40e6)
        self.steel = SteelIdeal(200e9, 400e6)
        
    def test_RectSection(self):
        r = RectSection(0.3, 0.5, self.conc, (0.15,0.25))
        self.assertEqual(r.area, 0.15)
        self.assertEqual(r.boundary, [(0,0), (0.3, 0.5)])
        
        stiff = r.get_stiffness(0, 0, 0)
        self.assertEqual(stiff[0,0], 6e9)
        self.assertAlmostEqual(stiff[1,1], 4.99996875e8)
        
        stiff = r.get_stiffness(-0.1, 0.5, 0.2)
        self.assertEqual(stiff[0,0], 22499999.999999974)
        self.assertEqual(stiff[0,1], 4467187.499999995)
        self.assertAlmostEqual(stiff[1,1], 886933.59375)
        
    def test_rebar(self):
        r = Rebar(0.01, self.steel, (0.1,0.4))
        
        stress = r.get_normal_stress(-1e-3)
        self.assertAlmostEqual(stress, -15707.963267948966)
        
        stiff = r.get_stiffness(0.001, 0.001, 0.1)
        self.assertAlmostEqual(stiff[0,0], 15707963.26794896)
        self.assertAlmostEqual(stiff[0,1], -4712388.98038469)
        self.assertAlmostEqual(stiff[1,1], 1413716.69411541)
        
    def test_section(self):
        r1 = RectSection(0.3, 0.5, self.conc, (0,0.25))
        
        s = Section([r1])

        xy_pairs = [(x,y) for x in [-.11,.11] for y in [0.04,0.25,0.45]]
        for xy in xy_pairs:
            s.addSingleRebar(diameter = 0.02,
                             material = self.steel,
                             position = xy)
        
        self.assertAlmostEqual(s.centroid[1], 0.24980294200852762)
        
        moment = s.get_moment_curvature(k_max=0.0125, normal_force=-2e6)[1]
        self.assertAlmostEqual(moment[1], 29104.696190344974)
        self.assertAlmostEqual(moment[-3], 430528.2097363019)
        
        moment = s.get_max_moment(n_points=500, error=1e1)
        self.assertAlmostEqual(moment, 165455.6077315667)
        
        s.plot_stress_strain(0.0025, 0.00588)
        s.plot_section()
        
    def test_big_section(self):
        web_40x160_left = RectSection(width=0.4,
                                      height=1.60,
                                      material=self.conc,
                                      center = (-1.5,0.8))

        web_40x160_right = RectSection(width=0.4,
                                       height=1.60,
                                       material=self.conc,
                                       center = (1.5,0.8))

        flange_650x30 = RectSection(width=6.5,
                                    height=0.30,
                                    material=self.conc,
                                    center = (0.0,1.75))
                                                 
        beam = Section(section=[web_40x160_left,
                               web_40x160_right,
                               flange_650x30])
                                   
        #Rebars
        center = -1.5
        beam.addLineRebar(diameter=0.01905,material=self.steel, spacing = 0.035,position= [[-0.17+center,0.05],[-0.17+center,0.21]])
        beam.addLineRebar(diameter=0.01905,material=self.steel, spacing = 0.04 ,position= [[-0.14+center,0.05],[-0.14+center,0.17]])
        beam.addLineRebar(diameter=0.01905,material=self.steel, spacing = 0.04 ,position= [[-0.11+center,0.05],[-0.11+center,0.17]])
        beam.addLineRebar(diameter=0.01905,material=self.steel, spacing = 0.04 ,position= [[-0.08+center,0.05],[-0.08+center,0.17]])
        beam.addLineRebar(diameter=0.01905,material=self.steel, spacing = 0.035,position= [[0.17+center,0.05],[0.17+center,0.21]])
        beam.addLineRebar(diameter=0.01905,material=self.steel, spacing = 0.04 ,position= [[0.14+center,0.05],[0.14+center,0.17]])
        beam.addLineRebar(diameter=0.01905,material=self.steel, spacing = 0.04 ,position= [[0.11+center,0.05],[0.11+center,0.17]])
        beam.addLineRebar(diameter=0.01905,material=self.steel, spacing = 0.04 ,position= [[0.08+center,0.05],[0.08+center,0.17]])

        center = 1.5
        beam.addLineRebar(diameter=0.01905,material=self.steel, spacing = 0.035,position= [[-0.17+center,0.05],[-0.17+center,0.21]])
        beam.addLineRebar(diameter=0.01905,material=self.steel, spacing = 0.04 ,position= [[-0.14+center,0.05],[-0.14+center,0.17]])
        beam.addLineRebar(diameter=0.01905,material=self.steel, spacing = 0.04 ,position= [[-0.11+center,0.05],[-0.11+center,0.17]])
        beam.addLineRebar(diameter=0.01905,material=self.steel, spacing = 0.04 ,position= [[-0.08+center,0.05],[-0.08+center,0.17]])
        beam.addLineRebar(diameter=0.01905,material=self.steel, spacing = 0.035,position= [[0.17+center,0.05],[0.17+center,0.21]])
        beam.addLineRebar(diameter=0.01905,material=self.steel, spacing = 0.04 ,position= [[0.14+center,0.05],[0.14+center,0.17]])
        beam.addLineRebar(diameter=0.01905,material=self.steel, spacing = 0.04 ,position= [[0.11+center,0.05],[0.11+center,0.17]])
        beam.addLineRebar(diameter=0.01905,material=self.steel, spacing = 0.04 ,position= [[0.08+center,0.05],[0.08+center,0.17]])
        
        self.assertAlmostEqual(beam.centroid[1], 1.3368960502351404)
        self.assertEqual(beam.get_section_boundary(), ((-3.25, 0), (3.25, 1.9)))
        
        self.assertAlmostEqual(beam.section[2].get_strains(0.012158068992549786, 0.05)[10], 0.018870568992549786)
        
        #self.assertAlmostEqual(beam.get_e0(k=0.05, e0=-0.02, target_normal=-100000), 0.027954338097796818)
        
        self.assertAlmostEqual(beam.get_normal_res(-0.001, 0.01), -8973304.87181267)
        self.assertAlmostEqual(beam.get_moment_res(-0.001, 0.01), 923619.105205293)
        self.assertAlmostEqual(beam.get_stiff(-0.001, 0.01)[0][0], 3100673607.5244894)
        self.assertAlmostEqual(beam.get_stiff(-0.001, 0.01)[0][1], 96871664.88501751)
        self.assertAlmostEqual(beam.get_stiff(-0.001, 0.01)[1][1], 9687798.75024157)
        self.assertAlmostEqual(beam.get_stiff(0.012158068992549786, 0.05)[0][0], 1665601868.5202813)
        
        self.assertAlmostEqual(beam.get_moment_curvature(0.005)[1][-1], 13526949.18408586)
        self.assertAlmostEqual(beam.get_max_moment(n_points=100), 13545087.817052273)
        
        self.assertAlmostEqual(beam.check_section(0, 12767717)[0], 0.00042183539410122254)
        self.assertAlmostEqual(beam.check_section(0, 12767717)[1], 0.001232568505796175)
        
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
        self.assertAlmostEqual(self.s.check_section(target_normal=0, target_moment=0.0373e6, n_ite=1000), (None, None))
        
        # 4.5.3
        self.assertAlmostEqual(self.s.check_section(target_normal=-1.1657e6, target_moment=0.0373e6, n_ite=1000), (None, None))
        
        # 4.5.4
        e0, k = self.s.check_section(target_normal=-0.8160e6, target_moment=0, n_ite=1000)
        self.assertAlmostEqual(e0*1e3, -0.7884934594071175, delta=self.delta)
        self.assertAlmostEqual(k*1e3, 0, delta=self.delta)
        
        # 4.5.5
        self.assertAlmostEqual(self.s.check_section(target_normal=-0.8160e6, target_moment=-0.0560e6, n_ite=1000), (None, None))
        
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
