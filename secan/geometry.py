from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from math import pi
from abc import ABC, abstractmethod


from abc import ABC, abstractmethod
from typing import List, Tuple

class Geometry(ABC):
    """
    Abstract base class representing a geometry.

    Attributes:
        None

    Methods:
        area -> float: Abstract property to get the area of the geometry.
        boundary -> List[Tuple[float, float]]: Abstract property to get the boundary coordinates of the geometry.
        get_normal_resistance(e0: float, k: float, center: float) -> np.ndarray: Abstract method to calculate normal resistance.
        get_moment_resistance(e0: float, k: float, center: float) -> np.ndarray: Abstract method to calculate moment resistance.
        get_stiffness(e0: float, k: float, center: float) -> np.ndarray: Abstract method to calculate stiffness.
        set_material(material): Set the material for the geometry.
        plot_geometry(graph): Method to plot the geometry (default implementation prints a message).
        set_x_plot(graph, value): Method to set x-axis limits of a plot (adjusting with value).
    """

    @property
    @abstractmethod
    def area(self) -> float:
        """
        Get the area of the geometry.

        Returns:
            float: Area of the geometry.
        """
        pass
        
    @property
    @abstractmethod
    def boundary(self) -> List[Tuple[float, float]]:
        """
        Get the boundary coordinates of the geometry.

        Returns:
            List[Tuple[float, float]]: List of boundary coordinates as tuples.
        """
        pass

    @abstractmethod
    def get_normal_resistance(self, e0: float, k: float, center: float) -> np.ndarray:
        """
        Calculate normal resistance based on the given parameters.

        Args:
            e0 (float): Normal strain value.
            k (float): Curvature value.
            center (float): Center value.

        Returns:
            np.ndarray: Array of normal resistance values.
        """
        pass

    @abstractmethod
    def get_moment_resistance(self, e0: float, k: float, center: float) -> np.ndarray:
        """
        Calculate moment resistance based on the given parameters.

        Args:
            e0 (float): Normal strain value.
            k (float): Curvature value.
            center (float): Center value.

        Returns:
            np.ndarray: Array of moment resistance values.
        """
        pass

    @abstractmethod
    def get_stiffness(self, e0: float, k: float, center: float) -> np.ndarray:
        """
        Calculate stiffness based on the given parameters.

        Args:
            e0 (float): Normal strain value.
            k (float): Curvature value.
            center (float): Center value.

        Returns:
            np.ndarray: Array of stiffness values.
        """
        pass

    def set_material(self, material) -> None:
        """
        Set the material for the geometry.

        Args:
            material: Material instance to be set for the geometry.
        """
        self.material = material

    def plot_geometry(self, graph: object) -> None:
        """
        Plot the geometry (default implementation prints a message).

        Args:
            graph (object): Plotting object (e.g., Matplotlib axis).
        """
        print("Plot is not implemented for this section")

    def set_x_plot(self, graph: object, value: float) -> None:
        """
        Set x-axis limits of a plot (adjusting with value).

        Args:
            graph (object): Plotting object (e.g., Matplotlib axis).
            value (float): Value used to adjust x-axis limits.
        """
        xabs_max = abs(max(graph.get_xlim(), key=abs))
        value *= 1.05
        if value > xabs_max:
            graph.set_xlim(xmin=-value, xmax=value)
        else:
            graph.set_xlim(xmin=-xabs_max, xmax=xabs_max)



class RectSection(Geometry):
    """
    Rectangle section geometry class.
    """

    def __init__(self, width: float, height: float, material,
                 center: tuple[float, float] = (0, 0),
                 rotation=0, n_discret=200):
        """
        Initialize the RectSection geometry.

        Args:
            width (float): Width of the rectangle.
            height (float): Height of the rectangle.
            material: Material instance for the geometry.
            center (tuple[float, float]): Center coordinates of the rectangle.
            rotation: Rotation of the rectangle (not used).
            n_discret (int): Number of discretization points.
        """
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

    def get_strains(self, e0: float, k: float) -> np.ndarray:
        """
        Calculate strains based on normal strain and curvature.

        Args:
            e0 (float): Normal strain value.
            k (float): Curvature value.

        Returns:
            np.ndarray: Array of calculated strains.
        """
        return e0 + k * (self.height/2 - self.h_discret) 

    def get_e0_sec(self, e0, k, center):
        """
        Calculate normal strain of the section based on the normal strain, curvature and center of the main section.

        Args:
            e0: Normal strain value.
            k: Curvature value.
            center: Center of the main section.

        Returns:
            float: Calculated e0_sec.
        """
        return e0 + k * (center - self.center[1])

    def get_normal_resistance_discrete(self, e0, k, center):
        e0_sec = self.get_e0_sec(e0, k, center)
        strains = self.get_strains(e0_sec, k)
        normal = self.material.get_stress(strains)
        return normal * self.area_discret

    def get_normal_resistance(self, e0, k, center):
        return np.sum(self.get_normal_resistance_discrete(e0, k, center))

    def get_moment_resistance(self, e0, k, center):
        normal = self.get_normal_resistance_discrete(e0, k, center)
        dist = (center - self.center[1]) + (self._height / 2 - self.h_discret)
        return np.sum(normal * dist)

    def get_normal_stiff_discrete(self, e0, k, center):
        e0_sec = self.get_e0_sec(e0, k, center)
        strains = self.get_strains(e0_sec, k)
        normal = self.material.get_stiff(strains)
        return normal * self.area_discret
    
    def get_stiffness(self, e0: float, k: float, 
                      center: float) -> np.ndarray:   
        normal = self.get_normal_stiff_discrete(e0, k, center)
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
        """
        Plot the rectangle geometry.

        Args:
            graph: Matplotlib axis to plot the geometry.
        """
        if graph is None:
            fig, graph = plt.subplots(1, figsize=(10, 10))
        graph.add_patch(Rectangle(self.boundary[0],
                                  self.width,
                                  self.height,
                                  edgecolor='blue',
                                  lw=2))


class Rebar(Geometry):
    def __init__(self, diameter, material, center=(0, 0)):
        self.diameter = diameter
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
            self._diameter = (new_area*4/pi)**0.5
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
                 center=(0, 0),
                 strain_ULS=10e-3 # Ultimate Limite State
                ):
        super().__init__(diameter, material, center)
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
            return np.array(([0, 0],
                             [0, 0]))
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
