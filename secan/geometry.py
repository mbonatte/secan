import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from math import pi
from abc import ABC, abstractmethod


class Geometry(ABC):
    @abstractmethod
    def get_area(self):
        pass

    @abstractmethod
    def get_normal_resistance(self, e0, k, center):
        pass

    @abstractmethod
    def get_moment_resistance(self, e0, k, center):
        pass

    @abstractmethod
    def get_stiffness(self, e0, k, center):
        pass

    @abstractmethod
    def get_boundary(self):
        pass

    def set_material(self, material):
        self.material = material

    def plot_geometry(self, graph):
        print("Plot is not implemented for this section")

    def set_x_plot(self, graph, value):
        xabs_max = abs(max(graph.get_xlim(), key=abs))
        value *= 1.05
        if value > xabs_max:
            graph.set_xlim(xmin=-value, xmax=value)
        else:
            graph.set_xlim(xmin=-xabs_max, xmax=xabs_max)


class Rect_section(Geometry):
    def __init__(self, width, height, material, center=(0, 0),
                 rotation=0, n_discret=200):
        self._width = width
        self._height = height
        self.center = height/2
        self.center_x = center[0]
        self.center_y = center[1]
        self.material = material
        self.n_discret = n_discret
        self.h_discret = [(self.height/2+self.height*i)/n_discret
                          for i in range(n_discret)]
        self.h_discret = np.array(self.h_discret)
        self.area_discret = self.width*self.height/self.n_discret

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, new_width):
        self._width = new_width
        self.area_discret = self.width*self.height/self.n_discret

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, new_height):
        self._height = new_height
        self.h_discret = [(self.height/2+self.height*i)/self.n_discret
                          for i in range(self.n_discret)]
        self.h_discret = np.array(self.h_discret)
        self.area_discret = self.width*self.height/self.n_discret

    def get_area(self):
        return self.width*self.height

    def get_strain(self, e0=0, k=0, pos=0):
        return e0+k*(self.center-pos)

    def get_strains(self, e0, k):
        strains = [e0+k*(self.center-h_discret)
                   for h_discret in self.h_discret]
        return strains

    def get_e0_sec(self, e0, k, center):
        return e0 + k*(center-self.center_y)

    def get_normal_resistance_discrete(self, e0, k, center):
        e0_sec = self.get_e0_sec(e0, k, center)
        strains = self.get_strains(e0_sec, k)
        normal = map(self.material.get_stress,strains)
        normal = np.fromiter(normal, dtype=float)
        return normal * self.area_discret

    def get_normal_resistance(self, e0, k, center):
        return sum(self.get_normal_resistance_discrete(e0, k, center))

    def get_moment_resistance(self, e0, k, center):
        normal = self.get_normal_resistance_discrete(e0, k, center)
        dist = (center-self.center_y)+(self.center-self.h_discret)
        return sum(normal * dist)

    def get_normal_stiff_discrete(self, e0, k, center):
        e0_sec = self.get_e0_sec(e0, k, center)
        strains = self.get_strains(e0_sec, k)
        normal = map(self.material.get_stiff,strains)
        normal = np.fromiter(normal, dtype=float)
        return normal * self.area_discret
    
    def get_stiffness(self, e0, k, center):
        e0_sec = self.get_e0_sec(e0, k, center)
        strains = self.get_strains(e0_sec, k)
        normal = self.get_normal_stiff_discrete(e0, k, center)
        dist = (center-self.center_y)+(self.center-self.h_discret)
        a00 = sum(normal)
        a01 = sum(normal * dist)
        a11 = sum(normal * dist**2)
        a10 = a01
        return np.array(([a00, a01],
                         [a10, a11]))

    def plot_stress(self, graph, e0, k, center):
        e0_sec = self.get_e0_sec(e0, k, center)
        strain = self.get_strains(e0_sec, k)
        stress = [self.material.get_stress(strain[i])
                  for i in range(len(strain))]
        bottom = self.center_y-self.height/2
        top = self.center_y+self.height/2
        h_discret = self.h_discret+bottom

        graph.plot(stress, h_discret, color='b')
        graph.plot([0, stress[0]], [bottom, bottom], color='b')
        graph.plot([0, stress[-1]], [top, top], color='gray')
        self.set_x_plot(graph, abs(max(stress, key=abs)))

    def plot_strain(self, graph, e0, k, center):
        e0_sec = self.get_e0_sec(e0, k, center)
        strain = self.get_strains(e0_sec, k)
        bottom = self.center_y - self.height/2
        top = self.center_y + self.height/2
        h_discret = self.h_discret + bottom

        graph.plot(strain, h_discret, color='b')
        graph.plot([0, strain[0]], [bottom, bottom], color='b')
        graph.plot([0, strain[-1]], [top, top], color='gray')
        self.set_x_plot(graph, abs(max(strain, key=abs)))

    def get_boundary(self):
        x0 = self.center_x-self.width/2
        x1 = self.center_x+self.width/2
        y0 = self.center_y-self.height/2
        y1 = self.center_y+self.height/2
        return ((x0, y0), (x1, y1))

    def plot_geometry(self, graph=None):
        if graph is None:
            fig, graph = plt.subplots(1, figsize=(10, 10))
        graph.add_patch(Rectangle(self.get_boundary()[0],
                                  self.width,
                                  self.height,
                                  edgecolor='blue',
                                  lw=2))


class Rebar(Geometry):
    def __init__(self, diameter, material, position=(0, 0)):
        self._diameter = diameter
        self._area = pi*(diameter**2)/4
        self.material = material
        self.center_x = position[0]
        self.center_y = position[1]

    @property
    def diameter(self):
        return self._diameter

    @diameter.setter
    def diameter(self, new_diameter):
        if (new_diameter >= 0):
            self._diameter = new_diameter
            self._area = 3.141592*(new_diameter**2)/4
        else:
            raise Exception("Diameter must be higher than 0")

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, new_area):
        if (new_area >= 0):
            self._area = new_area
            self._diameter = (new_area*4/3.141592)**0.5
        else:
            raise Exception("Area must be higher than 0")

    def get_area(self):
        return self.area

    def get_normal_stress(self, strain=0):
        return self.area * self.material.get_stress(strain)

    def get_normal_resistance(self, e0, k, center):
        strain = e0 + k * (center - self.center_y)
        return self.get_normal_stress(strain)

    def get_moment_resistance(self, e0=0, k=0, center=0):
        strain = e0 + k * (center - self.center_y)
        return self.get_normal_stress(strain) * (center-self.center_y)

    def get_stiffness(self, e0, k, center):
        strain = e0 + k * (center - self.center_y)
        normal = self.area * self.material.get_stiff(strain)
        dist = (center-self.center_y)
        a00 = normal
        a01 = normal * dist
        a10 = a01
        a11 = normal * dist**2
        return np.array(([a00, a01],
                         [a10, a11]))

    def plot_stress(self, graph, e0, k, center):
        stress = self.material.get_stress(e0+k*(center-self.center_y))
        graph.plot([0, stress], [self.center_y, self.center_y], color='r')
        self.set_x_plot(graph, abs(stress))

    def plot_strain(self, graph, e0, k, center):
        strain = e0+k*(center-self.center_y)
        graph.plot([0, strain], [self.center_y, self.center_y], color='r')
        self.set_x_plot(graph, abs(strain))

    def get_boundary(self):
        x = self.center_x
        y = self.center_y
        return ((x, y),)

    def plot_geometry(self, graph=None):
        if graph is None:
            fig, graph = plt.subplots(1, figsize=(10, 10))
        graph.add_patch(Circle((self.get_boundary()[0]),
                               self.diameter/2,
                               color='red'))
