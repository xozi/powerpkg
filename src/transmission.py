import numpy as np



class Node:
    def __init__(self, type, P=None, Q=None, vi=1.0):
        self.type = type
        if type == "slack":
            self.P = 0
            self.Q = 0
            self.vi = 1.0
            self.thetai = 0.0
        elif type == "PV":
            self.P = P
            self.Q = None
            self.vi = vi
            self.thetai = 0.0
        elif type == "PQ":
            self.P = -P
            self.Q = -Q
            self.vi = 1.0
            self.thetai = 0.0

#Simple line model, omits transiteance and shunt admittances
class Line:
    def __init__(self, Z, fromNode: Node, toNode: Node):
        self.Y = 1/Z
        self.fromNode = fromNode
        self.toNode = toNode

class Transmission:
    def __init__(self, Sbase):
        self.Sbase = Sbase
        self.Nodes = [];
        self.Lines = [];
    
    def add_line(self, line: Line):
        self.Lines.append(line)
        return self
    
    def add_node(self, node: Node):
        self.Nodes.append(node)
        return self

    def ymatrix(self):
        self.Y = np.zeros((len(self.Nodes), len(self.Nodes)), dtype=complex)
        # Fill Y-bus matrix based on line data
        for line in self.Lines:
            from_idx = self.Nodes.index(line.fromNode)
            to_idx = self.Nodes.index(line.toNode)
            self.Y[from_idx, to_idx] = -line.Y
            self.Y[to_idx, from_idx] = -line.Y
            self.Y[from_idx, from_idx] += line.Y
            self.Y[to_idx, to_idx] += line.Y
        
        # G and B matrices (real and imaginary parts of Y)
        self.G = self.Y.real
        self.B = self.Y.imag

    def build(self):
        self.ymatrix()
        # Get voltage magnitudes and angles
        self.voltages = np.array([node.vi for node in self.Nodes])
        self.angles = np.array([node.thetai for node in self.Nodes])
        
        # Extract P and Q values from nodes
        self.Pi = np.array([node.P for node in self.Nodes])
        self.Qi = np.array([node.Q if node.Q is not None else 0 for node in self.Nodes])

        # Newton-Raphson iteration
        max_iter = 500
        tolerance = 1e-6
        self.nlen = len(self.Nodes)
        for iter in range(max_iter):
            # Calculate power mismatches
            P_calc, Q_calc = self.PQ_iterate()

            # Calculate mismatches (only for non-slack buses)
            dP = np.zeros(self.nlen)
            dQ = np.zeros(self.nlen)
            mismatch = [];

            for i in range(0, self.nlen):
                if self.Nodes[i].type == "slack":
                    continue
                elif self.Nodes[i].type == "PV":
                    dP[i] = self.Pi[i] - P_calc[i]
                    mismatch.append(dP[i])
                elif self.Nodes[i].type == "PQ":
                    dP[i] = self.Pi[i] - P_calc[i]
                    dQ[i] = self.Qi[i] - Q_calc[i]
                    mismatch.append(dP[i])
                    mismatch.append(dQ[i])

            mismatch = np.array(mismatch)

            #Check convergence
            if np.max(np.abs(dP[1:])) < tolerance and np.max(np.abs(dQ[1:])) < tolerance:
                print(f"Converged in {iter} iterations")
                break

            #Build Jacobian
            J = self.build_jacobian()

                #Solve for updates
            updates = np.linalg.solve(J, mismatch)

            # Apply updates - track position in updates array
            update_pos = 0
        
            # First update angles for non-slack buses
            for i in range(self.nlen):
                if self.Nodes[i].type != "slack":
                    self.angles[i] += updates[update_pos]
                    update_pos += 1
            
            # Then update voltages for PQ buses only
            for i in range(self.nlen):
                if self.Nodes[i].type == "PQ":
                    self.voltages[i] += updates[update_pos]
                    update_pos += 1

            # Update node values with final solution
            for i, node in enumerate(self.Nodes):
                node.vj = self.voltages[i]
                node.thetaj = self.angles[i]

        return self

    def PQ_iterate(self):
        # Initialize arrays for calculated P and Q
        P_calc = np.zeros(self.nlen)
        Q_calc = np.zeros(self.nlen)
        
        # Calculate P and Q for each bus
        for i in range(self.nlen):
            for j in range(self.nlen):
                P_calc[i] += self.voltages[i] * self.voltages[j] * (
                    self.G[i, j] * np.cos(self.angles[i] - self.angles[j]) +
                    self.B[i, j] * np.sin(self.angles[i] - self.angles[j])
                )
                Q_calc[i] += self.voltages[i] * self.voltages[j] * (
                    self.G[i, j] * np.sin(self.angles[i] - self.angles[j]) -
                    self.B[i, j] * np.cos(self.angles[i] - self.angles[j])
                )
        
        return P_calc, Q_calc

    def build_jacobian(self):
        # Initialize Jacobian submatrices
        # J1: dP/dθ (n-1 × n-1)
        # J2: dP/dV (n-1 × npq)
        # J3: dQ/dθ (npq × n-1)
        # J4: dQ/dV (npq × npq)
        npq =  sum(1 for node in self.Nodes if node.type == "PQ")
        nonslack = sum(1 for node in self.Nodes if node.type != "slack")
        
        #initialize Jacobian submatrices
        J1 = np.zeros((nonslack, nonslack))
        J2 = np.zeros((nonslack, npq))
        J3 = np.zeros((npq, nonslack))
        J4 = np.zeros((npq, npq))

        # Get indices of non-slack buses and PQ buses
        nonslack_idx = [i for i, node in enumerate(self.Nodes) if node.type != "slack"]
        pq_idx = [i for i, node in enumerate(self.Nodes) if node.type == "PQ"]
        
            
        # Fill J1 (dP/dθ) for non-slack buses
        for i, bus_i in enumerate(nonslack_idx):
            for j, bus_j in enumerate(nonslack_idx):
                if bus_i == bus_j:  # Diagonal elements
                    J1[i, j] = 0
                    for k in range(self.nlen):
                        if k != bus_i:
                            angle_diff = self.angles[bus_i] - self.angles[k]
                            J1[i, j] += self.voltages[bus_i] * self.voltages[k] * (
                                self.G[bus_i, k] * np.sin(angle_diff) -
                                self.B[bus_i, k] * np.cos(angle_diff)
                            )
                else:  # Off-diagonal elements
                    angle_diff = self.angles[bus_i] - self.angles[bus_j]
                    J1[i, j] = -self.voltages[bus_i] * self.voltages[bus_j] * (
                        self.G[bus_i, bus_j] * np.sin(angle_diff) -
                        self.B[bus_i, bus_j] * np.cos(angle_diff)
                    )
        
          # Fill J2 (dP/dV) for non-slack buses to PQ buses
        for i, bus_i in enumerate(nonslack_idx):
            for j, bus_j in enumerate(pq_idx):
                if bus_i == bus_j:  # Diagonal elements
                    J2[i, j] = 0
                    for k in range(self.nlen):
                        angle_diff = self.angles[bus_i] - self.angles[k]
                        if k == bus_i:
                            J2[i, j] += 2 * self.voltages[bus_i] * self.G[bus_i, k]
                        else:
                            J2[i, j] += self.voltages[k] * (
                                self.G[bus_i, k] * np.cos(angle_diff) +
                                self.B[bus_i, k] * np.sin(angle_diff)
                            )
                else:  # Off-diagonal elements
                    angle_diff = self.angles[bus_i] - self.angles[bus_j]
                    J2[i, j] = self.voltages[bus_i] * (
                        self.G[bus_i, bus_j] * np.cos(angle_diff) +
                        self.B[bus_i, bus_j] * np.sin(angle_diff)
                    ) 

        # Fill J3 (dQ/dθ) for PQ buses to non-slack buses
        for i, bus_i in enumerate(pq_idx):
            for j, bus_j in enumerate(nonslack_idx):
                if bus_i == bus_j:  # Diagonal elements
                    J3[i, j] = 0
                    for k in range(self.nlen):
                        if k != bus_i:
                            angle_diff = self.angles[bus_i] - self.angles[k]
                            J3[i, j] += self.voltages[bus_i] * self.voltages[k] * (
                                self.G[bus_i, k] * np.cos(angle_diff) +
                                self.B[bus_i, k] * np.sin(angle_diff)
                            )
                else:  # Off-diagonal elements
                    angle_diff = self.angles[bus_i] - self.angles[bus_j]
                    J3[i, j] = -self.voltages[bus_i] * self.voltages[bus_j] * (
                        self.G[bus_i, bus_j] * np.cos(angle_diff) +
                        self.B[bus_i, bus_j] * np.sin(angle_diff)
                    )
            
    # Fill J4 (dQ/dV) for PQ buses to PQ buses
        for i, bus_i in enumerate(pq_idx):
            for j, bus_j in enumerate(pq_idx):
                if bus_i == bus_j:  # Diagonal elements
                    J4[i, j] = 0
                    for k in range(self.nlen):
                        angle_diff = self.angles[bus_i] - self.angles[k]
                        if k == bus_i:
                            J4[i, j] -= 2 * self.voltages[bus_i] * self.B[bus_i, k]
                        else:
                            J4[i, j] += self.voltages[k] * (
                                self.G[bus_i, k] * np.sin(angle_diff) -
                                self.B[bus_i, k] * np.cos(angle_diff)
                            )
                else:  # Off-diagonal elements
                    angle_diff = self.angles[bus_i] - self.angles[bus_j]
                    J4[i, j] = self.voltages[bus_i] * (
                        self.G[bus_i, bus_j] * np.sin(angle_diff) -
                        self.B[bus_i, bus_j] * np.cos(angle_diff)
                    )
        
        # Combine the submatrices to form the full Jacobian
        top_half = np.hstack((J1, J2))
        bottom_half = np.hstack((J3, J4))
        J = np.vstack((top_half, bottom_half))
        
        return J