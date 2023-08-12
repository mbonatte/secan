import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Material(ABC):

    @abstractmethod
    def get_stiff(self, strain: float) -> float:
        pass

    @abstractmethod
    def get_stress(self, strain: float) -> float:
        pass

    def plot(self, plot=None):
        print("Plot is not implemented for this material")


class Linear(Material):
    def __init__(self, young=0):
        self.young = young

    def get_stiff(self, strain=0):
        return self.young

    def get_stress(self, strain=0):
        return self.young*strain


class Concrete(Material):  # According to NBR6118 and EN1992
    def __init__(self, fc=0):
        self.fc = fc
        
                
    @staticmethod
    def _get_constants(fc: float) -> tuple[float, float, float]:
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
        if (self.ec2 < strain <= 0):
            return -1*(self.fc*self.n * (1-strain/self.ec2)**(self.n-1) / self.ec2)
        elif (strain <= self.ec2):
            return 0
        else:
            return 0

    def get_stress(self, strain: float=0) -> float:
        if (self.ec2 <= strain <= 0):
            return -1*self.fc*(1-(1-strain/self.ec2)**self.n) 
        elif (self.ecu <= strain <= self.ec2):
            return -self.fc
        elif (strain <= self.ecu):
            return 0
        else:
            return 0

    def plot(self, graph=None):
        if graph is None:
            fig, graph = plt.subplots(1, figsize=(10, 10))
        strain = np.arange(1.05*self.ecu, 1/5000, 1/50000)
        stress = [self.get_stress(strain[i]) for i in range(len(strain))]
        graph.set(xlabel='Strain')
        graph.set(ylabel='Stress')
        graph.set_title("Concrete Driagram")
        graph.grid()
        graph.plot(strain, stress)


class SteelIdeal(Material):
    def __init__(self,
                 young=0,
                 fy=0,
                 ultimate_strain=10e-3):
        self.young = young
        self.fy = fy
        self.ultimate_strain = ultimate_strain
        
    @property
    def fy(self):
        return self._fy    
     
    @fy.setter
    def fy(self, new_fy):
        self._fy = new_fy
        self.yeild_strain = new_fy / self.young

    def get_stiff(self, strain=0):
        if (-self.yeild_strain <= strain <= self.yeild_strain):
            return self.young
        else:
            return 0

    def get_stress(self, strain=0):
        if (-self.yeild_strain <= strain <= self.yeild_strain):
            return self.young*strain
        elif (-self.ultimate_strain <= strain <= self.ultimate_strain):
            return self.fy * strain/abs(strain)
        else:
            return 0

    def plot(self, graph=None):
        if graph is None:
            fig, graph = plt.subplots(1, figsize=(10, 10))
        strain = np.arange(-self.ultimate_strain,
                           self.ultimate_strain,
                           self.ultimate_strain/100)
        stress = [self.get_stress(strain[i]) for i in range(len(strain))]
        graph.set(xlabel='Strain')
        graph.set(ylabel='Stress')
        graph.set_title("Steel Driagram")
        graph.grid()
        graph.plot(strain, stress)
        
class SteelHardening(Material):
    def __init__(self,
                 young=0,
                 fy=0,
                 ft=0,
                 ultimate_strain=35e-3):
        self.young = young
        self.fy = fy
        self.ultimate_strain = ultimate_strain
        self.ft = ft
        
        
        
    @property
    def ft(self):
        return self._ft   
     
    @ft.setter
    def ft(self, new_ft):
        self._ft = new_ft
        self.hardening_stiffness = (new_ft - self.fy) / (self.ultimate_strain - self.yeild_strain)

    def get_stiff(self, strain=0):
        if (-self.yeild_strain <= strain <= self.yeild_strain):
            return self.young
        elif self.yeild_strain < strain < self.ultimate_strain:
            return self.hardening_stiffness
        else:
            return 0

    def get_stress(self, strain=0):
        if (-self.yeild_strain <= strain <= self.yeild_strain):
            return self.young*strain
        elif self.yeild_strain < strain < self.ultimate_strain:
            return self.fy + (strain - self.yeild_strain) * self.hardening_stiffness
        elif (-self.ultimate_strain <= strain <= -self.yeild_strain):
            return self.fy * strain/abs(strain)
        else:
            return 0

    def plot(self, graph=None):
        if graph is None:
            fig, graph = plt.subplots(1, figsize=(10, 10))
        strain = np.arange(-self.ultimate_strain,
                           self.ultimate_strain,
                           self.ultimate_strain/100)
        stress = [self.get_stress(strain[i]) for i in range(len(strain))]
        graph.set(xlabel='Strain')
        graph.set(ylabel='Stress')
        graph.set_title("Steel Driagram")
        graph.grid()
        graph.plot(strain, stress)
