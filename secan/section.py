from typing import List
import numpy as np
import matplotlib.pyplot as plt

from . import geometry


class Section:

    def __init__(self, section: List[geometry.Geometry]=None):
        self.area_rebar = 0
        self.centroid_rebar = 0
        self.centroid = (0, 0)
        if (section is None):
            self.section = []
        else:
            self.section = section
            self.centroid = self._compute_centroid()
        
        self.residual_e0 = 10
        self.max_increment_e0 = 0.0005
        self.n_ite_e0 = 30

        self.tolerance_check_section = 0.01

    def _compute_centroid(self) -> None:
        areas = np.array([s.area for s in self.section])
        stiffnesses = np.array([s.material.get_stiff() for s in self.section])
        weighted_areas = areas * stiffnesses
        
        cx = np.sum(weighted_areas * [s.center[0] for s in self.section]) / np.sum(weighted_areas)
        cy = np.sum(weighted_areas * [s.center[1] for s in self.section]) / np.sum(weighted_areas)
    
        self.centroid = np.array([cx, cy])
        self.set_rebar_centroid()

    def set_rebar_centroid(self):
        self.area_rebar = 0
        sum_moments_y = 0
        for section in self.section:
            if isinstance(section, geometry.Rebar):
                area = section.area
                self.area_rebar += area
                sum_moments_y += area * section.center[1]
        if self.area_rebar > 0:
            self.centroid_rebar = sum_moments_y/self.area_rebar

    def get_moment_res(self, e0: float, k: float) -> np.ndarray:
        moments = np.array([s.get_moment_resistance(e0, k, self.centroid[1])
                            for s in self.section])
        return moments.sum()

    def get_normal_res(self, e0: float, k: float) -> np.ndarray:
        normal = np.array([s.get_normal_resistance(e0, k, self.centroid[1])
                           for s in self.section])
        return normal.sum()

    def get_stiff(self, e0: float, k: float) -> np.ndarray:
        K = np.array([s.get_stiffness(e0, k, self.centroid[1]) for s in self.section])
        return K.sum(axis=0)
        

    def get_e0(self, k=0, e0=0, target_normal=0):
        for _ in range(self.n_ite_e0):
            normal_int = self.get_normal_res(e0, k)
            
            residual = abs(abs(target_normal)-abs(normal_int))
            if residual < self.residual_e0:
                return e0
        
            stiff = self.get_stiff(e0, k)[0, 0]
            if stiff < 1e-10:
                raise ZeroDivisionError
            
            increment = (target_normal - normal_int)/stiff
            
            e0 += np.sign(increment) * min(abs(increment), self.max_increment_e0)
        
        return None

    def check_section(self, target_normal, target_moment, n_ite=10):
        e0 = 0
        k = 0

        for _ in range(n_ite):
            normal = self.get_normal_res(e0, k)
            moment = self.get_moment_res(e0, k)
            
            residual = np.array([[target_normal-normal],
                                [target_moment-moment]])
            if np.linalg.norm(residual) < self.tolerance_check_section:
                return e0, k
            
            stiff = self.get_stiff(e0, k)
            if np.linalg.det(stiff) < self.tolerance_check_section:
                break

            # update section state
            inv_stiff = np.linalg.inv(stiff)
            e0 += np.matmul(inv_stiff, residual )[0][0]
            k += np.matmul(inv_stiff, residual )[1][0]
        return None, None

    def get_strain_base_top(self, f, inverted=False):
        if f <= 0.5:
            eb = 15/1000
            et = -2*3.5*f/1000
        elif f >= 0.5:
            eb = -f*3e-2+3e-2
            et = -3.5/1000
        if inverted:
            (eb, et) = (et, eb)
        return eb, et

    def get_max_moment(self, n_points=50, is_inverted=False, error=1e3):
        height = self.get_section_boundary()[1][1]
        bottom = self.get_section_boundary()[0][1]
        et, eb = -4e-3, 15e-3
        max_curvature = (eb - et) / (height - bottom)
        
        if is_inverted:
            max_curvature *= -1

        curvatures, moments = self.get_moment_curvature(max_curvature, normal_force=0, n_points=5)
                
        max_moment = moments.max() if not is_inverted else moments.min()
        index = moments.argmax() if not is_inverted else moments.argmin()
               
        min_curvature = curvatures[index] 
        
        e0_start = 0.1*(10e-3 - min_curvature * (self.centroid[1]-bottom)) #Uma tentativa muito louca de fazer convergir. Preciso ver isso melhor depois.
        
        if is_inverted:
            e0_start = -1 * e0_start
        
        try:
            e0_start = self.get_e0(min_curvature, e0_start)
        except ZeroDivisionError:
             e0_start = 0
        
        for j in range(n_points):
          
          curvature_range = max_curvature - min_curvature
          curvature = min_curvature + (curvature_range / 2)
          
          try:
              e0 = self.get_e0(curvature, e0_start)
              moment = self.get_moment_res(e0, curvature)
          except ZeroDivisionError:
                moment = 0
          
          # Update curvature bounds          
          if abs(moment) > abs(max_moment):
              if abs(max_moment-moment) < error:
                  max_moment = moment
                  break
              max_moment = moment
              min_curvature = curvature
              e0_start = e0
          else:
              max_curvature = curvature
        
        return max_moment

    def get_max_moment_simplified(self):
        Rst = 0
        for section in self.section:
            if isinstance(section, geometry.Rebar):
                Rst += section.get_area()*section.material.fy
        x = Rst/(self.section[1].material.fc*0.8*self.section[1].width)
        h = (self.section[1].center_y+self.section[1].height/2)
        M = Rst*(h-self.centroid_rebar-0.5*0.8*x)
        return M

    def get_max_moment_reversed_simplified(self):
        Rst = 0
        for section in self.section:
            if isinstance(section, geometry.Rebar):
                Rst += section.get_area()*section.material.fy
        x = Rst/(self.section[0].material.fc*0.8*self.section[0].width)
        bottom = (self.section[0].center_y-self.section[0].height/2)
        M = Rst*(self.centroid_rebar-(bottom + 0.5*0.8*x))
        return M

    def get_moment_curvature(self, k_max, normal_force=0, n_points=50):
        k = np.linspace(0, k_max, n_points)
        moment = np.empty(n_points)
        e0 = 0
        for i in range(n_points):
            try:
                e0 = self.get_e0(k[i], e0, normal_force)
                moment[i] = self.get_moment_res(e0, k[i])
            except ZeroDivisionError:
                moment[i] = 0
        return k, moment

    def plot_moment_curvature(self, k_max, normal_force=0,
                              n_points=50, graph=None):
        if graph is None:
            fig, graph = plt.subplots(1, figsize=(5, 4))
        k, moment = self.get_moment_curvature(k_max, normal_force, n_points)
        if (k_max > 0):
            print(f'Max moment = {max(moment)}')
        else:
            print(f'Max moment = {min(moment)}')
        graph.plot(k, moment)
        graph.grid()
        graph.set(xlabel='Curvature')
        graph.set(ylabel='Moment')
        graph.set_title("Total moment")

    def plot_stress_strain(self, e0, k):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.plot_stress(e0, k, ax1)
        self.plot_strain(e0, k, ax2)
        print('Normal = ', self.get_normal_res(e0, k))
        print('Moment = ', self.get_moment_res(e0, k))
        

    def plot_stress(self, e0, k, graph=None):
        if graph is None:
            fig, graph = plt.subplots(1, figsize=(10, 10))
        graph.set_title("Stress Driagram")
        graph.set_xlabel('Concrete - Stress')
        graph.set_ylabel('Height')
        graph.grid()
        ax2 = graph.twiny()
        ax2.set_xlabel("Rebar - Stress")
        for section in self.section:
            if isinstance(section, geometry.RectSection):
                section.plot_stress(graph, e0, k, self.centroid[1])
            elif isinstance(section, geometry.Rebar):
                section.plot_stress(ax2, e0, k, self.centroid[1])

    def plot_strain(self, e0, k, graph=None):
        if graph is None:
            fig, graph = plt.subplots(1, figsize=(10, 10))
        graph.set_title("Strain Driagram")
        graph.set_xlabel('Strain')
        graph.set_ylabel('Height')
        graph.grid()
        for section in self.section:
            section.plot_strain(graph, e0, k, self.centroid[1])

    def plot_section(self, graph=None):
        if graph is None:
            fig, graph = plt.subplots(1, figsize=(8, 8))
        graph.set_aspect('equal', adjustable='box')
        graph.margins(0.05)
        graph.autoscale()
        graph.set_title("Cross-section")
        graph.set(xlabel='X')
        graph.set(ylabel='Y')
        for section in self.section:
            section.plot_geometry(graph)

    def addSingleRebar(self, diameter, material, position):
        self.section.append(geometry.Rebar(diameter, material, position))
        self._compute_centroid()
        
    def addSingleTendon(self, diameter, material, initial_strain, position):
        self.section.append(geometry.Tendon(diameter, material, initial_strain, position))
        self._compute_centroid()

    def addLineRebar(self, diameter, material, spacing, position):
        x = (position[1][0] - position[0][0])
        y = (position[1][1] - position[0][1])
        length = (x**2+y**2)**0.5
        nBars = (int(length/spacing))+1
        for i in range(nBars):
            self.section.append(geometry.Rebar(diameter, material,
                                               [x*i/(nBars-1)+position[0][0],
                                                y*i/(nBars-1)+position[0][1]]))
        self._compute_centroid()

    def addRebars(self, rebars):
        if (type(rebars) == list):
            for rebar in rebars:
                self.section.append(rebar)
        else:
            self.section.append(rebars)
        self._compute_centroid()

    def get_section_boundary(self):
        x0 = 0
        y0 = 0
        x1 = 0
        y1 = 0
        for section in self.section:
            for bound in section.boundary:
                if bound[0] < x0:
                    x0 = bound[0]
                if bound[0] > x1:
                    x1 = bound[0]
                if bound[1] < y0:
                    y0 = bound[1]
                if bound[1] > y1:
                    y1 = bound[1]
        return ((x0, y0),
                (x1, y1))

    def get_section_height(self):
        bound = self.get_section_boundary()
        return bound[1][1]-bound[0][1]

    def plot_interaction_curve(self, n_points=100):
        fig, graph = plt.subplots(figsize=(5, 5))
        graph.grid()
        graph.set_title("Interaction curve")
        graph.set(xlabel='Normal')
        graph.set(ylabel='Moment')
        failure = [i/n_points for i in range(n_points+1)]
        height = self.get_section_height()
        bottom = self.get_section_boundary()[0][1]
        nu = []
        mu = []
        for i in range(2):
            for f in failure:
                if(f < 1/3):
                    eb = 10/1000
                    et = -40.5*f/1000 + 10/1000
                if(1/3 <= f <= 2/3):
                    eb = -30*f/1000+20/1000
                    et = -3.5/1000
                if(1 >= f > 2/3):
                    eb = -6*f/1000+4/1000
                    et = 4.5*f/1000-6.5/1000
                if(i == 1):
                    a = eb
                    eb = et
                    et = a
                k = (eb-et)/height
                e0 = (eb - k * (self.centroid[1]-bottom))
                normal = self.get_normal_res(e0, k)
                moment = self.get_moment_res(e0, k)
                nu.append(normal)
                mu.append(moment)
        graph.plot(nu, mu, color='b')
