import numpy as np
import matplotlib.pyplot as plt

from . import geometry


class Section:
    def __init__(self, section=None):
        self.area_rebar = 0
        self.centroid_rebar = 0
        self.centroid = (0, 0)
        if (section is None):
            self.section = []
        else:
            self.section = section
            self.set_centroid()

    def set_centroid(self):
        self.centroid = (0, 0)
        sum_area = 0
        sum_moments_x = 0
        sum_moments_y = 0
        for section in self.section:
            stiff = section.material.get_stiff()
            area = section.get_area()
            sum_area += stiff * area
            sum_moments_x += stiff * area * (section.center_x-self.centroid[0])
            sum_moments_y += stiff * area * (section.center_y-self.centroid[1])
        self.centroid = (sum_moments_x/sum_area, sum_moments_y/sum_area)
        self.set_rebar_centroid()

    def set_rebar_centroid(self):
        self.area_rebar = 0
        sum_moments_y = 0
        for section in self.section:
            if isinstance(section, geometry.Rebar):
                area = section.get_area()
                self.area_rebar += area
                sum_moments_y += area * section.center_y
        if self.area_rebar > 0:
            self.centroid_rebar = sum_moments_y/self.area_rebar

    def get_moment_res(self, e0, k):
        return sum(s.get_moment_resistance(e0, k, self.centroid[1])
                   for s in self.section)

    def get_normal_res(self, e0, k):
        return sum(s.get_normal_resistance(e0, k, self.centroid[1])
                   for s in self.section)

    def get_stiff(self, e0, k):
        return sum(s.get_stiffness(e0, k, self.centroid[1])
                   for s in self.section)

    def get_e0(self, k=0, e0=0, normal_force=0):
        normal_int = self.get_normal_res(e0, k)
        normal_ext = normal_force
        normal_dif = abs(abs(normal_ext)-abs(normal_int))
        n_ite = 30
        n = 0
        while(normal_dif > 10 and n < n_ite):
            stiff = self.get_stiff(e0, k)[0, 0]
            if stiff < 1e-10:
                raise ZeroDivisionError
            e0 += (normal_force - self.get_normal_res(e0, k))/stiff
            normal_int = self.get_normal_res(e0, k)
            normal_dif = abs(abs(normal_ext)-abs(normal_int))
            n += 1
        return e0

    def check_section(self, normal, moment, n_ite=10):
        norm, stiff_def = 10, 10
        e0, k = 0, 0
        n_int = self.get_normal_res(e0, k)
        m_int = self.get_moment_res(e0, k)
        n = 0
        load_dif = np.array([[normal-n_int],
                             [moment-m_int]])
        stiff = self.get_stiff(e0, k)
        while((norm > 0.01) and (stiff_def > 0.01) and (n < n_ite)):
            inv_stiff = np.linalg.inv(stiff)
            e0 += np.matmul(inv_stiff, load_dif)[0][0]
            k += np.matmul(inv_stiff, load_dif)[1][0]
            n_int = self.get_normal_res(e0, k)
            m_int = self.get_moment_res(e0, k)
            load_dif = np.array([[normal-n_int],
                                 [moment-m_int]])
            norm = np.linalg.norm(load_dif)
            stiff = self.get_stiff(e0, k)
            stiff_def = np.linalg.det(stiff)
            n += 1
        if (norm < 0.01):
            print("e0: ", e0)
            print("k: ", k)
            return e0, k
        else:
            print ("Section is not stable")

    def get_strain_base_top(self, f, inverted=False):
        if f <= 0.5:
            eb = 10/1000
            et = -2*3.5*f/1000
        elif f >= 0.5:
            eb = -f*2e-2+2e-2
            et = -3.5/1000
        if inverted:
            (eb, et) = (et, eb)
        return eb, et

    def get_max_moment(self, n_points=5, inverted=False):
        height = self.get_section_height()
        bottom = self.get_section_boundary()[0][1]
        initial, final = 0, 1
        for j in range(5):
            failure = np.linspace(initial, final, n_points)
            for i in range(n_points):
                eb, et = self.get_strain_base_top(failure[i], inverted)
                k = (eb-et)/height
                e0 = (eb - k * (self.centroid[1]-bottom))
                normal = self.get_normal_res(e0, k)
                if normal < 0:
                    final = failure[i]
                    initial = failure[i-1]
                    break
        moment = self.get_moment_res(e0, k)
        return moment

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
        k = [(k_max/n_points)*i for i in range(n_points)]
        moment = [0 for i in k]
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
            if isinstance(section, geometry.Rect_section):
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
        self.set_centroid()

    def addLineRebar(self, diameter, material, spacing, position):
        x = (position[1][0] - position[0][0])
        y = (position[1][1] - position[0][1])
        length = (x**2+y**2)**0.5
        nBars = (int(length/spacing))+1
        for i in range(nBars):
            self.section.append(geometry.Rebar(diameter, material,
                                               [x*i/(nBars-1)+position[0][0],
                                                y*i/(nBars-1)+position[0][1]]))
        self.set_centroid()

    def addRebars(self, rebars):
        if (type(rebars) == list):
            for rebar in rebars:
                self.section.append(rebar)
        else:
            self.section.append(rebars)
        self.set_centroid()

    def get_section_boundary(self):
        x0 = 0
        y0 = 0
        x1 = 0
        y1 = 0
        for section in self.section:
            for bound in section.get_boundary():
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
        for i in range(2):
            nu = []
            mu = []
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
