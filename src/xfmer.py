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

    def add_Zpu(self, Zpu):
        self.Z = Zpu

    def build(self):
        #Zbase from the lowside of voltage
        Zbase = (self.secondary_voltage**2) / self.rated_power
        #Pu to ohms
        self.Z = self.Z*1e-2 * Zbase

class DeltaWyeGrounded:
    def __init__(self, xfmer1, xfmer2, xfmer3):
        #Step up transformer
        if xfmer1.secondary_voltage > xfmer1.primary_voltage:
            #Forward Sweep Factors
            self.at = (xfmer1.nt/3) * np.array([[2.0,0.0,1.0],[1.0,2.0,0.0],[0.0,1.0,2.0]])
            self.bt =  (xfmer1.nt/3) * np.array([[2.0*xfmer1.Z,xfmer2.Z,0.0],
                                                 [0.0,2.0*xfmer2.Z,xfmer3.Z],
                                                 [xfmer1.Z,0.0,2.0*xfmer3.Z]])
            self.ct = np.zeros((3,3), dtype=complex)
            #Backward Sweep Factors
            self.At = (1/xfmer1.nt) * np.array([[1.0,0.0,-1.0],[0.0,1.0,-1.0],[-1.0,0.0,1.0]])
            self.dt = self.At
            self.Bt = np.zeros((3,3), dtype=complex)
            for i in range(3):
                for j in range(3):
                    if i == j == 0:
                        self.Bt[i,j] = xfmer1.Z
                    elif i == j == 1:
                        self.Bt[i,j] = xfmer2.Z
                    elif i == j == 2:
                        self.Bt[i,j] = xfmer3.Z
                    else:
                        self.Bt[i,j] = 0.0
        #Step down transformer
        else:
            #Foward Sweep Factors
            self.at = (-xfmer1.nt/3) * np.array([[0.0,1.0,2.0],[2.0,0.0,1.0],[1.0,2.0,0.0]])
            self.bt =  (-xfmer1.nt/3) * np.array([[0.0,2.0*xfmer2.Z,xfmer3.Z],
                                                  [xfmer1.Z,0.0,2.0*xfmer3.Z],
                                                  [2.0*xfmer1.Z,xfmer2.Z,0.0]])
            self.ct = np.zeros((3,3), dtype=complex)
            #Backward Sweep Factors
            self.At = (1/xfmer1.nt) * np.array([[1.0,0.0,-1.0],[-1.0,1.0,0.0],[0.0,-1.0,1.0]])
            self.dt = (1/xfmer1.nt) * np.array([[1.0,-1.0,0.0],[0.0,1.0,-1.0],[-1.0,0.0,1.0]])
            self.Bt = np.zeros((3,3), dtype=complex)
            for i in range(3):
                for j in range(3):
                    if i == j == 0:
                        self.Bt[i,j] = xfmer1.Z;
                    elif i == j == 1:
                        self.Bt[i,j] = xfmer2.Z;
                    elif i == j == 2:
                        self.Bt[i,j] = xfmer3.Z;
                    else:
                        self.Bt[i,j] = 0.0;
        self.xfmer1 = xfmer1
        self.xfmer2 = xfmer2
        self.xfmer3 = xfmer3




class WyeUngroundedDelta:
    def __init__(self, xfmer):
        #Foward Sweep Factors
        self.at = (xfmer.nt) * np.array([[1,0,-1],[-1,1,0],[0,-1,1]])
        #need mutual impedance calculation
        #self.bt =  (xfmer.nt/3) * np.array([[xfmer.Z,xfmer.Z,-2*xfmer.Z],[-xfmer.Z,2*xfmer.Z,xfmer.Z],[0,0,0]])
        #Backward Sweep Factors
        self.At = (1/(3*xfmer.nt)) * np.array([[2,0,1],[1,2,0],[0,1,2]])
        #self.Bt = (1/9)*np.array([[(2*xfmer.Z+xfmer.Z),(2*xfmer.Z+xfmer.Z),0],[0,(2*xfmer.Z+xfmer.Z),0],[0,0,(2*xfmer.Z+xfmer.Z)]])
        

class WyeWye:
    def __init__(self, xfmer1, xfmer2, xfmer3):
        #Foward Sweep Factors
        self.at = np.zeros((3,3), dtype=complex)
        for i in range(3):
            for j in range(3):
                if i == j:
                    self.at[i,j] = xfmer1.nt;
                else:
                    self.at[i,j] = 0.0;
        self.bt = np.zeros((3,3), dtype=complex)
        for i in range(3):
            for j in range(3):
                if i == j == 0:
                    self.bt[i,j] = xfmer1.Z*xfmer1.nt;
                elif i == j == 1:
                    self.bt[i,j] = xfmer2.Z*xfmer2.nt;
                elif i == j == 2:
                    self.bt[i,j] = xfmer3.Z*xfmer3.nt;
                else:
                    self.bt[i,j] = 0.0;
        self.ct = np.zeros((3,3), dtype=complex)
        self.At = np.zeros((3,3), dtype=complex)
        for i in range(3):
            for j in range(3):
                if i == j:
                    self.At[i,j] = 1/(xfmer1.nt);
                else:
                    self.At[i,j] = 0.0;
        self.dt = self.At
        self.Bt = np.zeros((3,3), dtype=complex)
        for i in range(3):
            for j in range(3):
                if i == j == 0:
                    self.Bt[i,j] = xfmer1.Z;
                elif i == j == 1:
                    self.Bt[i,j] = xfmer2.Z;
                elif i == j == 2:
                    self.Bt[i,j] = xfmer3.Z;
                else:
                    self.Bt[i,j] = 0.0;
        self.xfmer1 = xfmer1
        self.xfmer2 = xfmer2
        self.xfmer3 = xfmer3

class DeltaDelta:
    def __init__(self, primary_voltage, secondary_voltage, Y, Z):
        super().__init__(primary_voltage, secondary_voltage, Y, Z)