import unittest
import numpy as np

from secan.section import Section
from secan.material import Concrete, SteelIdeal, SteelHardening
from secan.geometry import RectSection, Rebar, Tendon
from secan.exceptions import SectionUnstableError, ConvergenceError

class TestSections(unittest.TestCase):
    
    def setUp(self):
        self.conc = Concrete(40e6)
        self.steel = SteelIdeal(200e9, 400e6)

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

        s.include_rebar_in_centroid = False
        s._compute_centroid()
        self.assertAlmostEqual(s.centroid[1], 0.25)

        
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
        
        with self.assertRaises(ConvergenceError):
            beam.get_e0(k=0.05, e0=-0.02, target_normal=-100000)
        
        beam.n_ite_e0 = 100
        self.assertAlmostEqual(beam.get_e0(k=0.05, e0=-0.02, target_normal=-100000), 0.027954338097796818)
        
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
   
if __name__ == '__main__':
    unittest.main()
