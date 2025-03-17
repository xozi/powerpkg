import numpy as np

def fsw_vol(at, bt, V, I):
    V_f = np.zeros((3,3), dtype=complex)
    V_f= np.matmul(at,V) + np.matmul(bt,I)
    return V_f

def bksw_vol(At, Bt, V, I):
    V_b = np.zeros((3,3), dtype=complex)
    V_b = np.matmul(At,V) + np.matmul(Bt,I)
    return V_b

__all__ = ['fsw_vol', 'bksw_vol']

'''
class PowerSystem:
    def __init__(self, V = None, Z = None, S = None):
        self.V = V(V) if V is not None else []  
        self.Z = Z(Z) if Z is not None else []
        self.S = S(S) if S is not None else []


    
class I:
    def __init__(self, I = None):
        self.I = I if I is not None else []
    
    def polar(self):
        magnitude = np.abs(self.I)
        angle = np.degrees(np.angle(self.I))
        return magnitude, angle
    
class S:
    def __init__(self, S = None):
        self.S = S if S is not None else []

    @classmethod
    def from_power_factor(cls, P, PF):
        Q = P * (1 / PF)  
        S = complex(P, Q)  
        return cls(S)
    
    def polar(self):
        magnitude = np.abs(self.S)
        angle = np.degrees(np.angle(self.S))
        return magnitude, angle
    
    def PF_Angle(self):
        return np.radians(np.angle(self.S))
'''