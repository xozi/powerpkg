import math
import numpy as np

class Xfmer:
    def __init__(self, rated_power, primary_voltage, secondary_voltage):
        self.rated_power = rated_power
        self.primary_voltage = primary_voltage
        self.secondary_voltage = secondary_voltage
        self.nt = primary_voltage / secondary_voltage

    def add_Zpu_phasor(self, pu, angle):
        self.Z = pu * np.exp(1j * angle) 

    def build(self):
        #Zbase from the lowside of voltage
        Zbase = self.secondary_voltage / self.rated_power
        self.Z = self.Z * Zbase

class DeltaWyeGrounded(Xfmer):
    def __init__(self, xfmer):
        if xfmer.secondary_voltage > xfmer.primary_voltage:
            self.at = (xfmer.nt/3) * np.array([[2,0,1],[1,2,0],[0,1,2]])
            self.bt =  (xfmer.nt/3) * np.array([[2*xfmer.Z,0,xfmer.Z],[xfmer.Z,2*xfmer.Z,0],[0,xfmer.Z,2*xfmer.Z]])
            #Backward Sweep Factors
            self.At = (1/xfmer.nt) * np.array([[1,0,-1],[0,1,0],[-1,-1,1]])
            self.Bt = xfmer.Z;
        else:
            #Foward Sweep Factors
            self.at = (-xfmer.nt/3) * np.array([[0,1,2],[2,0,1],[1,2,0]])
            self.bt =  (-xfmer.nt/3) * np.array([[0,xfmer.Z,2*xfmer.Z],[2*xfmer.Z,0,xfmer.Z],[xfmer.Z,2*xfmer.Z,0]])
            #Backward Sweep Factors
            self.At = (1/xfmer.nt) * np.array([[1,-1,0],[0,1,-1],[-1,0,1]])
            self.Bt = np.zeros((3,3), dtype=complex)
            for i,j in self.Bt:
                if i == j:
                    self.Bt[i,j] = xfmer.Z;
                else:
                    self.Bt[i,j] = 0;




class WyeUngroundedDelta(Xfmer):
    def __init__(self, xfmer):
        #Foward Sweep Factors
        self.at = (xfmer.nt) * np.array([[1,0,-1],[-1,1,0],[0,-1,1]])
        #need mutual impedance calculation
        #self.bt =  (xfmer.nt/3) * np.array([[xfmer.Z,xfmer.Z,-2*xfmer.Z],[-xfmer.Z,2*xfmer.Z,xfmer.Z],[0,0,0]])
        #Backward Sweep Factors
        self.At = (1/(3*xfmer.nt)) * np.array([[2,0,1],[1,2,0],[0,1,2]])
        #self.Bt = (1/9)*np.array([[(2*xfmer.Z+xfmer.Z),(2*xfmer.Z+xfmer.Z),0],[0,(2*xfmer.Z+xfmer.Z),0],[0,0,(2*xfmer.Z+xfmer.Z)]])
        

class WyeWye(Xfmer):
    def __init__(self, primary_voltage, secondary_voltage, Y, Z):
        super().__init__(primary_voltage, secondary_voltage, Y, Z)

class DeltaDelta(Xfmer):
    def __init__(self, primary_voltage, secondary_voltage, Y, Z):
        super().__init__(primary_voltage, secondary_voltage, Y, Z)