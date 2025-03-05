import numpy as np
import math
import cmath
import scipy

class TransmissionSystem:
    def __init__(self, Sbase):
        self.Sbase = Sbase
        self.path = [];

    def add_source(self, voltage, voltage_angle):
        self.path.append({"type": "source", "voltage": voltage, "voltage_angle": voltage_angle})
        return self 

    def add_impedance(self, impedance):
        admittance = 1/impedance;
        self.path.append({"type": "admittance", "value": admittance})
        return self  

    def add_adimittance(self, adimittance):
        self.path.append({"type": "admittance", "value": adimittance})
        return self  

    def add_PV_load(self, P,  V):
        #Connected close to generator
        #If we know P and V we can get the Q.
        Q = math.sqrt(P*P + V*V)
        self.path.append({"type": "load", "P": P, "Q": Q, "V": V})
        return self

    def add_PQ_load(self, P, Q):
        #Purely load buses
        #If we know P and Q, we can can get an angle.
        theta = math.atan(Q/P)
        admittance = 
        self.path.append({"type": "load", "P": P, "Q": Q, "theta": theta})
        return self

    def slack_bus(self, V, theta):
        #Conneted close to generator, a bus with large generation capacity
        #Voltage assumed as pu.
        self.V_i = V
        self.T_i = theta
        return self

    def build(self):
        #theta_ik = theta_i - theta_k
        for object in self.path:
            if object is hasattr["type"] == "admittance":
                self.admittance = object
            elif object is hasattr["type"] == "load":
                self.load = object
                
def P_equation(P, V, theta):
    return P_i - sum(abs(V_i)*abs(V_k)*(Yik.real*cos(theta_ik) + Yik.imag*sin(theta_ik)))

def Q_equation(Q, V, theta):
    return Q_i - sum(abs(V_i)*abs(V_k)*(Yik.real*sin(theta_ik) - Yik.imag*cos(theta_ik)))