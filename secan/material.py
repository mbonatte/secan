import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Material(ABC):
    """
    Abstract base class representing a material.

    Attributes:
        None

    Methods:
        get_stiff(strain: float) -> float: Abstract method to calculate material stiffness.
        get_stress(strain: float) -> float: Abstract method to calculate material stress.
        plot(plot: object) -> None: Method to plot the material (default implementation prints a message).
    """

    @abstractmethod
    def get_stiff(self, strain: float) -> float:
        """
        Calculate material stiffness based on the given strain.

        Args:
            strain (float): Strain value.

        Returns:
            float: Material stiffness.
        """
        pass

    @abstractmethod
    def get_stress(self, strain: float) -> float:
        """
        Calculate material stress based on the given strain.

        Args:
            strain (float): Strain value.

        Returns:
            float: Material stress.
        """
        pass

    def plot(self, plot=None):
        """
        Plot the material (default implementation prints a message).

        Args:
            plot (object): Plotting object (optional).
        """
        print("Plot is not implemented for this material")


class Linear(Material):
    """
    Linear material class representing a material with constant Young's modulus.

    Attributes:
        young (float): Young's modulus of the material.

    Methods:
        get_stiff(strain: float) -> float: Calculate material stiffness based on the given strain.
        get_stress(strain: float) -> float: Calculate material stress based on the given strain.
    """
    def __init__(self, young=0):
        """
        Initialize the Linear material with a given Young's modulus.

        Args:
            young (float): Young's modulus of the material.
        """
        self.young = young

    def get_stiff(self, strain=0):
        """
        Calculate material stiffness based on the given strain.

        Args:
            strain (float): Strain value.

        Returns:
            float: Material stiffness.
        """
        return self.young

    def get_stress(self, strain=0):
        """
        Calculate material stress based on the given strain.

        Args:
            strain (float): Strain value.

        Returns:
            float: Material stress.
        """
        return self.young*strain


class Concrete(Material):  # According to NBR6118 and EN1992
    def __init__(self, fc=0):
        """
        Initialize the Concrete material with a given fc value.

        Args:
            fc (float): Compressive strength of concrete.
        """
        self.fc = fc
        
                
    @staticmethod
    def _get_constants(fc: float) -> tuple[float, float, float]:
        """
        Compute material constants based on the given fc value.

        Args:
            fc (float): Compressive strength of concrete.

        Returns:
            tuple[float, float, float]: Tuple containing ec2, ecu, and n constants.
        """
        if 0e6 <= fc < 55e6:
            return -2 / 1e3, -3.5 / 1e3, 2
        elif 55e6 <= fc <= 90e6:
            return (-1*((2/1e3) + (0.085/1e3 * (fc/1e6 - 50)**0.53)), 
                    (-1*((2.6/1e3) + (35/1e3 * ((90-fc/1e6)/100)**4))),
                    1.4+23.4*((90-fc/1e6)/100)**4)
        else:
            raise ValueError('fc must be between 0MPa and 90MPa')

    @property
    def fc(self):
        return self._fc   
     
    @fc.setter
    def fc(self, new_fc):
        self._fc = new_fc
        self.ec2, self.ecu, self.n = self._get_constants(new_fc)

    def get_stiff(self, strain: float=0) -> float:
        """
        Calculate material stiffness based on the given strain.

        Args:
            strain (float): Strain value.

        Returns:
            float: Material stiffness.
        """
        condition1 = (self.ec2 <= strain) & (strain <= 0)
        condition2 = (strain <= self.ec2)
        
        result1 = -1*(self.fc*self.n * (1-strain/self.ec2)**(self.n-1) / self.ec2)
        result2 = 0
        
        stiff = np.select(
            [condition1, condition2],
            [result1, result2],
            default=0
        )
        
        return stiff
    
    def get_stress(self, strain: float=0) -> float:
        """
        Calculate material stress based on the given strain.

        Args:
            strain (float): Strain value.

        Returns:
            float: Material stress.
        """
        condition1 = (self.ec2 <= strain) & (strain <= 0)
        condition2 = (self.ecu <= strain) & (strain <= self.ec2)
        condition3 = strain <= self.ecu
        
        result1 = -1 * self.fc * (1 - (1 - strain / self.ec2) ** self.n)
        result2 = -self.fc
        result3 = 0
        
        stress = np.select(
            [condition1, condition2, condition3],
            [result1, result2, result3],
            default=0
        )
        
        return stress

    def plot(self, graph=None) -> None:
        """
        Plot the stress-strain diagram for the Concrete material.

        Args:
            graph: Matplotlib axis to plot the diagram.
        """
        if graph is None:
            fig, graph = plt.subplots(1, figsize=(10, 10))
        
        strain = np.arange(1.05*self.ecu, 1/5000, 1/50000)
        stress = self.get_stress(strain)
        
        graph.set(xlabel='Strain')
        graph.set(ylabel='Stress')
        graph.set_title("Concrete Driagram")
        graph.grid()
        graph.plot(strain, stress)


class SteelIdeal(Material):
    """
    SteelIdeal material class representing an idealized steel material.

    Attributes:
        young (float): Young's modulus of the material.
        fy (float): Yield strength of the material.
        ultimate_strain (float): Ultimate strain of the material.

    Methods:
        get_stiff(strain: float) -> float: Calculate material stiffness based on the given strain.
        get_stress(strain: float) -> float: Calculate material stress based on the given strain.
        plot(graph: object) -> None: Plot the stress-strain diagram for the SteelIdeal material.
    """
    def __init__(self,
                 young: float = 0,
                 fy: float = 0,
                 ultimate_strain: float = 10e-3):
        """
        Initialize the SteelIdeal material with given parameters.

        Args:
            young (float): Young's modulus of the material.
            fy (float): Yield strength of the material.
            ultimate_strain (float): Ultimate strain of the material.
        """
        self.young = young
        self.fy = fy
        self.ultimate_strain = ultimate_strain
        
    @property
    def fy(self):
        return self._fy    
     
    @fy.setter
    def fy(self, new_fy):
        self._fy = new_fy
        self.yield_strain = new_fy / self.young

    def get_stiff(self, strain: float = 0) -> float:
        """
        Calculate material stiffness based on the given strain.

        Args:
            strain (float): Strain value.

        Returns:
            float: Material stiffness.
        """
        if (-self.yield_strain <= strain <= self.yield_strain):
            return self.young
        else:
            return 0

    def get_stress(self, strain: float = 0) -> float:
        """
        Calculate material stress based on the given strain.

        Args:
            strain (float): Strain value.

        Returns:
            float: Material stress.
        """
        if (-self.yield_strain <= strain <= self.yield_strain):
            return self.young*strain
        elif (-self.ultimate_strain <= strain <= self.ultimate_strain):
            return self.fy * strain/abs(strain)
        else:
            return 0

    def plot(self, graph=None) -> None:
        """
        Plot the stress-strain diagram for the SteelIdeal material.

        Args:
            graph: Matplotlib axis to plot the diagram.
        """
        if graph is None:
            fig, graph = plt.subplots(1, figsize=(10, 10))
        strain = np.arange(-self.ultimate_strain,
                           self.ultimate_strain,
                           self.ultimate_strain/100)
        stress = [self.get_stress(strain[i]) for i in range(len(strain))]
        graph.set(xlabel='Strain', ylabel='Stress', title="Steel Diagram")
        graph.grid()
        graph.plot(strain, stress)
        
class SteelHardening(SteelIdeal):
    """
    SteelHardening material class representing an idealized steel material with hardening behavior.

    Attributes:
        young (float): Young's modulus of the material.
        fy (float): Yield strength of the material.
        ft (float): Ultimate tensile strength of the material.
        ultimate_strain (float): Ultimate strain of the material.

    Methods:
        get_stiff(strain: float) -> float: Calculate material stiffness based on the given strain.
        get_stress(strain: float) -> float: Calculate material stress based on the given strain.
        plot(graph: object) -> None: Plot the stress-strain diagram for the SteelHardening material.
    """

    def __init__(self,
                 young: float = 0,
                 fy: float = 0,
                 ft: float = 0,
                 ultimate_strain: float = 35e-3):
        """
        Initialize the SteelHardening material with given parameters.

        Args:
            young (float): Young's modulus of the material.
            fy (float): Yield strength of the material.
            ft (float): Ultimate tensile strength of the material.
            ultimate_strain (float): Ultimate strain of the material.
        """
        self.young = young
        self.fy = fy
        self.ultimate_strain = ultimate_strain
        self.ft = ft

    @property
    def ft(self) -> float:
        """
        Get the ultimate tensile strength property.

        Returns:
            float: Ultimate tensile strength of the material.
        """
        return self._ft
    
    @ft.setter
    def ft(self, new_ft: float) -> None:
        """
        Set the ultimate tensile strength property and update related parameters.

        Args:
            new_ft (float): New ultimate tensile strength of the material.
        """
        self._ft = new_ft
        self.hardening_stiffness = (new_ft - self.fy) / (self.ultimate_strain - self.yield_strain)

    def get_stiff(self, strain: float = 0) -> float:
        """
        Calculate material stiffness based on the given strain.

        Args:
            strain (float): Strain value.

        Returns:
            float: Material stiffness.
        """
        if (-self.yield_strain <= strain <= self.yield_strain):
            return self.young
        elif self.yield_strain < strain < self.ultimate_strain:
            return self.hardening_stiffness
        else:
            return 0

    def get_stress(self, strain: float = 0) -> float:
        """
        Calculate material stress based on the given strain.

        Args:
            strain (float): Strain value.

        Returns:
            float: Material stress.
        """
        if (-self.yield_strain <= strain <= self.yield_strain):
            return self.young * strain
        elif self.yield_strain < strain < self.ultimate_strain:
            return self.fy + (strain - self.yield_strain) * self.hardening_stiffness
        elif (-self.ultimate_strain <= strain <= -self.yield_strain):
            return self.fy * strain / abs(strain)
        else:
            return 0

    def plot(self, graph=None) -> None:
        """
        Plot the stress-strain diagram for the SteelHardening material.

        Args:
            graph: Matplotlib axis to plot the diagram.
        """
        if graph is None:
            fig, graph = plt.subplots(1, figsize=(10, 10))
        strain = np.arange(-self.ultimate_strain,
                           self.ultimate_strain,
                           self.ultimate_strain / 100)
        stress = [self.get_stress(strain[i]) for i in range(len(strain))]
        graph.set(xlabel='Strain', ylabel='Stress', title="Steel Diagram")
        graph.grid()
        graph.plot(strain, stress)
