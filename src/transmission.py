import numpy as np
import math
import cmath
import scipy

class Node:
    def __init__(self, type, P, Q=None, vi=1.0, thetai=0.0):
        self.type = type
        self.P = P
        self.Q = Q
        self.vi = vi
        self.thetai = thetai
        self.vj = vi
        self.thetaj = thetai

#Simple line model, omits transiteance and shunt admittances
class Line:
    def __init__(self, z, fromNode: Node, toNode: Node):
        self.y = 1/z
        self.fromNode = fromNode
        self.toNode = toNode


class Transmission:
    def __init__(self, Sbase):
        self.Sbase = Sbase
        self.Nodes = [];
        self.Lines = [];
    
    def add_line(self, line: Line):
        #Simple line model, omits transiteance and shunt admittances
        #The Y bus will have ybus creation done differently if multinodal
        self.Lines.append(line)
        return self
    
    def slack_node(self, V, theta):
        #Connected close to generator, a bus with large generation capacity
        #Voltage assumed as pu.
        self.Nodes.append(Node("slack", 0, 0, vi=V, thetai=theta))
        return self

    def PV_node(self, P,  V):
        #Connected close to generator, generative node
        #If we know P and V we can get the Q.
        self.Nodes.append(Node("PV", P, Q=None, vi=V, thetai=0.0))
        return self

    def PQ_node(self, P, Q):
        #Purely load buses, make P and Q negative
        #If we know P and Q, we can can get an angle.
        self.Nodes.append(Node("PQ", -P, -Q, vi=1.0, thetai=0.0))
        return self
            
    def build(self):
        self.Y = np.zeros((len(self.Nodes), len(self.Nodes)), dtype=complex)
        for line in self.Lines:
            self.Y[line.fromNode.id, line.toNode.id] = line.y
            self.Y[line.toNode.id, line.fromNode.id] = line.y

        #G and B Matrix
        self.G = self.Y.real
        self.B = self.Y.imag
        
        # Get voltage magnitudes and angles
        self.voltages = np.array([node.vf for node in self.Nodes])
        self.angles = np.array([node.thetai for node in self.Nodes])

        #P and Q values init - extract from nodes
        self.Pi = np.array([node.P for node in self.Nodes])
        self.Qi = np.array([node.Q for node in self.Nodes])
        Pi = self.Pi
        Qi = self.Qi
        while True:
            (PqChange, QqChange) = self.PQ_iterate(Pi, Qi)
            dPa = self.Pi - PqChange
            dQa = self.Qi - QqChange
            if np.linalg.norm(dPa) < 1e-6 and np.linalg.norm(dQa) < 1e-6:
                break
            Pi = PqChange
            Qi = QqChange
        return


    def PQ_iterate(self, Pi, Qi):
        for i, node in enumerate(self.Nodes):
            for k in range(len(self.Nodes)):
                Pi[i] += self.voltages[i]*self.voltages[k]*(self.G[i, k]*np.cos(self.angles[i]-self.angles[k]) + self.B[i,k]*np.sin(self.angles[i]-self.angles[k]))
                Qi[i] += self.voltages[i]*self.voltages[k]*(self.G[i, k]*np.sin(self.angles[i]-self.angles[k]) - self.B[i,k]*np.cos(self.angles[i]-self.angles[k]))
        return Pi, Qi


def P_equation(P, V, theta):
    return P_i - sum(abs(V_i)*abs(V_k)*(Yik.real*cos(theta_ik) + Yik.imag*sin(theta_ik)))

def Q_equation(Q, V, theta):
    return Q_i - sum(abs(V_i)*abs(V_k)*(Yik.real*sin(theta_ik) - Yik.imag*cos(theta_ik)))