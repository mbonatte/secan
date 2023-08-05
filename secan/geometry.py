from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from math import pi
from abc import ABC, abstractmethod


class Geometry(ABC):

    @property
    @abstractmethod
    def area(self) -> float:
        pass
        
    @property
    @abstractmethod
    def boundary(self) -> List[tuple[float, float]]:
        pass

    @abstractmethod
    def get_normal_resistance(self, e0: float, k: float, center: float) -> np.ndarray:
        pass

    @abstractmethod
    def get_moment_resistance(self, e0: float, k: float, center: float) -> np.ndarray:
        pass

    @abstractmethod
    def get_stiffness(self, e0: float, k: float, center: float) -> np.ndarray:
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


class RectSection(Geometry):
    def __init__(self, width: float, height: float, material,
                 center: tuple[float, float]=(0, 0),
                 rotation=0, n_discret=200):
        self._width = width
        self._height = height
        self.center = np.array(center)
        self.material = material
        self._discretize(n_discret)

    def _discretize(self, n: int = 200):
        self.n_discret = n
        self.h_discret = np.array([(self.height/2+self.height*i)/n
                                   for i in range(n)])
        self.area_discret = self.area / n
    
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
        self._discretize(self.n_discret)
        self.area_discret = self.width*self.height/self.n_discret

    @property
    def area(self) -> float:
        return self.width * self.height
        
    @property
    def boundary(self) -> List[tuple[float, float]]:
        x0, x1 = self.center[0] - self.width/2, self.center[0] + self.width/2  
        y0, y1 = self.center[1] - self.height/2, self.center[1] + self.height/2
        return [(x0, y0), (x1, y1)]

    def get_area(self):
        return self.width*self.height

    def get_strain(self, e0=0, k=0, pos=0):
        return e0+k*(self.self.height/2-pos)

    def get_strains(self, e0: float, k: float) -> np.ndarray:
        return e0 + k * (self.height/2 - self.h_discret) 
        
    def get_stress(self, e0: float, k: float, center: float) -> np.ndarray:
        strains = self.get_strain(e0, k)
        return self.material.get_stress(strains)

    def get_e0_sec(self, e0, k, center):
        return e0 + k * (center - self.center[1])

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
        #dist = (center-self.center[1])+(self.center[1]-self.h_discret)
        dist = (center-self.center[1])+(self._height/2-self.h_discret)
        #print(self._height/2, self.center[1])
        return sum(normal * dist)

    def get_normal_stiff_discrete(self, e0, k, center):
        e0_sec = self.get_e0_sec(e0, k, center)
        strains = self.get_strains(e0_sec, k)
        
        normal = np.array([
            self.material.get_stiff(strain) for strain in strains
        ])
        return normal * self.area_discret
    
    def get_stiffness(self, e0: float, k: float, 
                      center: float) -> np.ndarray:
                      
        normal = self.get_normal_stiff_discrete(e0, k, center)
        #dist = (center-self.center[1]) + (self.center[1]-self.h_discret)
        dist = (center-self.center[1])+(self._height/2-self.h_discret)
        
        a00 = normal.sum()
        a01 = (normal * dist).sum()
        a11 = (normal * dist**2).sum()
        
        return np.array(([a00, a01],
                         [a01, a11]))

    def plot_stress(self, graph, e0, k, center):
        e0_sec = self.get_e0_sec(e0, k, center)
        strain = self.get_strains(e0_sec, k)
        stress = [self.material.get_stress(strain[i])
                  for i in range(len(strain))]
        bottom = self.center[1]-self.height/2
        top = self.center[1]+self.height/2
        h_discret = self.h_discret+bottom

        graph.plot(stress, h_discret, color='b')
        graph.plot([0, stress[0]], [bottom, bottom], color='b')
        graph.plot([0, stress[-1]], [top, top], color='gray')
        self.set_x_plot(graph, abs(max(stress, key=abs)))

    def plot_strain(self, graph, e0, k, center):
        e0_sec = self.get_e0_sec(e0, k, center)
        strain = self.get_strains(e0_sec, k)
        bottom = self.center[1] - self.height/2
        top = self.center[1] + self.height/2
        h_discret = self.h_discret + bottom

        graph.plot(strain, h_discret, color='b')
        if k > 0:
            graph.plot([-3.5e-3,-3.5e-3], [bottom, top], color='gray', linestyle='dotted')
        graph.plot([0, strain[0]], [bottom, bottom], color='b')
        graph.plot([0, strain[-1]], [top, top], color='gray')
        self.set_x_plot(graph, abs(max(strain, key=abs)))

    def plot_geometry(self, graph=None):
        if graph is None:
            fig, graph = plt.subplots(1, figsize=(10, 10))
        graph.add_patch(Rectangle(self.boundary[0],
                                  self.width,
                                  self.height,
                                  edgecolor='blue',
                                  lw=2))


class Rebar(Geometry):
    def __init__(self, diameter, material, center=(0, 0)):
        self._diameter = diameter
        self._area = pi * diameter**2 / 4
        self.material = material
        self.center = np.array(center)

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
    
    @property    
    def boundary(self) -> List[tuple[float, float]]:
        return [(self.center[0], self.center[1])]

    def get_normal_stress(self, strain=0):
        return self.area * self.material.get_stress(strain)

    def get_normal_resistance(self, e0, k, center):
        strain = e0 + k * (center - self.center[1])
        return self.get_normal_stress(strain)

    def get_moment_resistance(self, e0=0, k=0, center=0):
        strain = e0 + k * (center - self.center[1])
        return self.get_normal_stress(strain) * (center-self.center[1])

    def get_stiffness(self, e0, k, center):
        strain = e0 + k * (center - self.center[1])
        normal = self.area * self.material.get_stiff(strain)
        dist = center - self.center[1]
        
        a00 = normal
        a01 = normal * dist
        a11 = normal * dist**2
        
        return np.array(([a00, a01],
                         [a01, a11]))

    def plot_stress(self, graph, e0, k, center):
        stress = self.material.get_stress(e0+k*(center-self.center[1]))
        graph.plot([0, stress], [self.center[1], self.center[1]], color='r')
        self.set_x_plot(graph, abs(stress))

    def plot_strain(self, graph, e0, k, center):
        strain = e0+k*(center-self.center[1])
        graph.plot([0, strain], [self.center[1], self.center[1]], color='r')
        self.set_x_plot(graph, abs(strain))

    def plot_geometry(self, graph=None):
        if graph is None:
            fig, graph = plt.subplots(1, figsize=(10, 10))
        graph.add_patch(Circle((self.boundary[0]),
                               self.diameter/2,
                               color='red'))

class Tendon(Rebar):
    def __init__(self,
                 diameter,
                 material,
                 initial_strain,
                 position=(0, 0),
                 strain_ULS=10e-3 # Ultimate Limite State
                ):
        super().__init__(diameter, material, position)
        self.initial_strain = initial_strain
        self.strain_ULS = strain_ULS

    def get_normal_stress(self, strain=0):
        if strain > self.strain_ULS:
            return 0
        strain += self.initial_strain  #This still has to be tested
        return self.area * self.material.get_stress(strain)

    def get_stiffness(self, e0, k, center):
        strain = e0 + k * (center - self.center[1])
        if strain > self.strain_ULS:
            return 0
        strain += self.initial_strain  #This still has to be tested
        normal = self.area * self.material.get_stiff(strain)
        dist = (center-self.center[1])
        a00 = normal
        a01 = normal * dist
        a10 = a01
        a11 = normal * dist**2
        return np.array(([a00, a01],
                         [a10, a11]))

    def plot_stress(self, graph, e0, k, center):
        strain = e0 + k * (center - self.center[1])
        strain += self.initial_strain  #This still has to be tested
        stress = self.material.get_stress(strain)
        graph.plot([0, stress], [self.center[1], self.center[1]], color='lime')
        self.set_x_plot(graph, abs(stress))

    def plot_strain(self, graph, e0, k, center):
        strain = e0 + k * (center - self.center[1])
        strain += self.initial_strain  #This still has to be tested
        graph.plot([0, strain], [self.center[1], self.center[1]], color='lime')
        self.set_x_plot(graph, abs(strain))

    def plot_geometry(self, graph=None):
        if graph is None:
            fig, graph = plt.subplots(1, figsize=(10, 10))
        graph.add_patch(Circle((self.boundary[0]),
                               self.diameter/2,
                               color='lime'))
