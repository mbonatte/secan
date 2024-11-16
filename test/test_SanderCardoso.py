import unittest

from secan.section import Section
from secan.material import Concrete, SteelIdeal
from secan.geometry import RectSection, Rebar

class TestSanderCardoso(unittest.TestCase):
    """
        The tests in this section are from the document "Sistema computacional para análise não linear de pilares de concreto armado" by Sander David Cardoso Júnior.
    """
    
    def setUp(self):
        fck = 20e6
        fyk = 500e6
        young_steel = 210e9
        
        gamma_c = 1.4
        gamma_s = 1.15
        
        rusch_factor = 0.85
        
        self.conc = Concrete(rusch_factor*fck/gamma_c)
        self.steel = SteelIdeal(young_steel, fyk/gamma_s)
        
        self.r_x = RectSection(0.6, 0.3, self.conc, (0, 0), n_discret=200)
        self.r_y = RectSection(0.3, 0.6, self.conc, (0, 0), n_discret=200)
        
        self.s_x = Section([self.r_x])
        self.s_y = Section([self.r_y])
        
        xy_pairs = [(x,y) for x in [-.264, -.1584, -.0528, .0528, .1584, .264] for y in [-0.114, 0.114]]
        for xy in xy_pairs:
            self.s_x.addSingleRebar(diameter = 0.02,
                                    material = self.steel,
                                    position = xy)
        xy_pairs = [(x,y) for x in [-.264, .264] for y in [-0.038, 0.038]]
        for xy in xy_pairs:
            self.s_x.addSingleRebar(diameter = 0.02,
                                    material = self.steel,
                                    position = xy)
                                    
        xy_pairs = [(x,y) for x in [-0.114, 0.114] for y in [-.264, -.1584, -.0528, .0528, .1584, .264]]
        for xy in xy_pairs:
            self.s_y.addSingleRebar(diameter = 0.02,
                                    material = self.steel,
                                    position = xy)
        xy_pairs = [(x,y) for x in [-0.038, 0.038] for y in [-.264, .264]]
        for xy in xy_pairs:
            self.s_y.addSingleRebar(diameter = 0.02,
                                    material = self.steel,
                                    position = xy)
    
    # 
    
    def test_moment_res_x(self):
        # k and e0 based on the Figura 5.2
        k = (4.49-1.26)*1e-3/0.076
        e0 = (4.49+1.26)*1e-3/2
        
        
        concrete_moment = sum([s.get_moment_resistance(e0=e0, k=k, center=self.s_x.centroid[1])
                               for s in self.s_x.section if isinstance(s, RectSection)])
        
        rebar_moment = sum([s.get_moment_resistance(e0=e0, k=k, center=self.s_x.centroid[1]) 
                            for s in self.s_x.section if isinstance(s, Rebar)])
        
        
        concrete_moment_cardoso = -(-1.214*11.71*6.59*60) * 10
        rebar_moment_cardoso = -(-41.35*18.85*11.4 + 26.51*6.28*3.8 + 43.48*6.28*(-3.8) + 43.48*18.85*(-11.4)) * 10
        
        self.assertAlmostEqual(concrete_moment, concrete_moment_cardoso, delta=10)
        self.assertAlmostEqual(rebar_moment, rebar_moment_cardoso, delta=50)
        
        target_normal = 0
        target_moment = 242.62e3
        
        # There is a difference of 12 N.m, which is acceptable.
        self.assertAlmostEqual(self.s_x.get_moment_res(e0, k), target_moment, delta=12)
        
        # There is a difference of 5.6 kN.m, which is very high acceptable.
        self.assertAlmostEqual(self.s_x.get_max_moment(), target_moment, delta=5.61e3)
        
        # Getting the max_moment using the moments from curvature
        max_moment = max(self.s_x.get_moment_curvature(0.05, n_points=1000)[1])
        
        # There is a difference of 255 N.m, which is a bit high.
        self.assertAlmostEqual(max_moment, target_moment, delta=255)
        
        e0_, k_ = self.s_x.check_section(target_normal, max_moment, n_ite=100)
        self.assertLess((e0_-e0)/e0, 0.025)
        self.assertLess((k_-k)/k, 0.015)

if __name__ == '__main__':
    unittest.main()
