import numpy as np

class PQLoad:
    def __init__(self, P, Q=None, PF=None):
        self.P = P
        if Q is None and PF is not None:
            self.S = toRectangular(P, np.arccos(PF), True)
        else:
            self.Q = Q
            self.S = P + 1j * self.Q

class PVLoad:
    def __init__(self, P, V):
        self.P = P
        self.V = V
        self.S = P + 1j * 0
class VLL_Source:
    def __init__(self, V, LL=True):
        LL_angles = [30.0, -90.0 if len(V) >= 2 else 0.0, 150.0 if len(V) == 3 else 0.0] 
        self.V = []
        for v, angle in zip(V, LL_angles):
            if LL:
                self.V.append(v * np.exp(1j * np.radians(angle)))
            else:
                self.V.append((v * np.sqrt(3)) * np.exp(1j * np.radians(angle)))

class VLN_DeltaSource:
    def __init__(self, V, LN=True):
        v_angles = [30.0, -90.0 if len(V) >= 2 else 0.0, 150.0 if len(V) == 3 else 0.0]  
        self.V = []
        for v, angle in zip(V, v_angles):
            if LN:
                self.V.append(v  * np.exp(1j * np.radians(angle)))
            else:
                self.V.append((v / np.sqrt(3)) * np.exp(1j * np.radians(angle)))

def toPolar(V):
    magnitude = np.abs(V)
    angle = np.degrees(np.angle(V))
    return magnitude, angle

def toRectangular(magnitude, angle, radians=False):
    if radians:
        return magnitude * np.exp(1j * angle)
    else:
        return magnitude * np.exp(1j * np.radians(angle))

__all__ = ['PQLoad', 'PVLoad', 'VLL_Source', 'VLN_DeltaSource', 'toPolar', 'toRectangular']