import numpy as np

def fsw(at, bt, V, I):
    V_f = np.zeros((3,3), dtype=complex)
    I_f = np.zeros((3,3), dtype=complex)
    for i,j in V_f:
        V_f[i,j] = at[i,j] * V[i,j] + bt[i,j] * I[i,j]
    return V_f, I_f

def bksw(At, Bt, V, I):
    V_b = np.zeros((3,3), dtype=complex)
    I_b = np.zeros((3,3), dtype=complex)
    for i,j in V_b:
        V_b[i,j] = At[i,j] * V[i,j] + Bt[i,j] * I[i,j]
    return V_b, I_b

'''
class PowerSystem:
    def __init__(self, V = None, Z = None, S = None):
        self.V = V(V) if V is not None else []  
        self.Z = Z(Z) if Z is not None else []
        self.S = S(S) if S is not None else []

class V:
    def __init__(self, v_base = None, v_angle = None):
        self.V = v_base*np.exp(1j*v_angle) if v_base is not None and v_angle is not None else []

    def LL(self):
        return self.V   
    
    def LN(self):
        return self.V / np.sqrt(3)
    
    @classmethod
    def from_VLN (cls, V):
        v_angles = [0.0, -120.0 if len(V) >= 2 else 0.0, 120.0 if len(V) == 3 else 0.0]  # Phase A, B, C angles
        complex_V = [v  * np.exp(1j * np.radians(angle)) for v, angle in zip(V, v_angles)]
        return cls(complex_V)
    
    @classmethod
    def from_VLL_to_VLN(cls, V):
        LL_angles = [30.0, -90.0 if len(V) >= 2 else 0.0, 150.0 if len(V) == 3 else 0.0]  # Phase A, B, C angles
        complex_V = [(v / np.sqrt(3)) * np.exp(1j * np.radians(angle)) for v, angle in zip(V, LL_angles)]
        return cls(complex_V)
    
    def polar(self):
        magnitude = np.abs(self.V)
        angle = np.degrees(np.angle(self.V))
        return magnitude, angle
    
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