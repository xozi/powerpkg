import numpy as np
import math

class OverheadLineBuilder:
    """Builder pattern for creating power line models"""
    def __init__(self):
        self.conductors = []
        self.frequency = 60.0  # Default frequency in Hz

    def add_conductor(self, resistance, gmr, x_pos, y_pos, diameter):
        """Add a conductor to the line model"""
        self.conductors.append({
            'resistance': resistance,  # ohms/mile
            'gmr': gmr,                # feet
            'x': x_pos,                # feet
            'y': y_pos,                # feet
            'diameter': diameter       # inches
        })
        return self

    def build_primitive_matrices(self):
        """Build the primitive impedance and potential coefficient matrices"""
        Z_primitive = np.zeros((self.size, self.size), dtype=complex)
        P_primitive = np.zeros((self.size, self.size), dtype=complex)
        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    Z_primitive[i, j], P_primitive[i, j] = self.calculate_self_impedance(i)
                else:
                    Z_primitive[i, j], P_primitive[i, j] = self.calculate_mutual_impedance(i, j)
        return Z_primitive, P_primitive
    
    def calculate_self_impedance(self, i):
        """Calculate self impedance for conductor i"""
        resistance = self.conductors[i]['resistance']
        Sij = 2.0 * self.conductors[i]['y']
        Dij = self.conductors[i]['gmr']
        # Convert diameter to radius into inches
        rd = (self.conductors[i]['diameter'] / 2.0) / 12.0
        
        # Series impedance component
        z = complex(
            resistance + 0.09530,  # Resistance + earth return resistance
            0.12134 * (math.log(1.0 / Dij) + 7.93402)  # Reactance
        )
        
        # Shunt admittance component (inverse)
        p = complex(
            11.17689 * math.log(Sij / rd),
            0.0
        )
        
        return z, p
    
    def calculate_mutual_impedance(self, i, j):
        """Calculate mutual impedance between conductors i and j"""
        # Direct distance between conductors
        Dij = math.sqrt(
            (self.conductors[j]['x'] - self.conductors[i]['x'])**2 + 
            (self.conductors[j]['y'] - self.conductors[i]['y'])**2
        )
        
        # Distance including earth return (image conductor)
        Sij = math.sqrt(
            (self.conductors[j]['x'] - self.conductors[i]['x'])**2 + 
            (self.conductors[i]['y'] + self.conductors[j]['y'])**2
        )
        
        # Series impedance component
        z = complex(
            0.09530,  # Earth return resistance
            0.12134 * (math.log(1.0 / Dij) + 7.93402)  # Reactance
        )
        
        # Shunt admittance component (inverse)
        p = complex(
            11.17689 * math.log(Sij / Dij),
            0.0
        )
        
        return z, p

    def reduce_primitives(self, Z_primitive, P_primitive):
        """Kron reduce primitive matrices, get tn, and additional operations for Y matrix"""
        Z = np.zeros((self.phase_count, self.phase_count), dtype=complex)
        P = np.zeros((self.phase_count, self.phase_count), dtype=complex)
        tn = np.zeros(self.phase_count, dtype=complex)
        
        # Kron reduction
        for i in range(self.phase_count):
            for j in range(self.phase_count):
                # Reduce Z matrix (series impedance)
                Z[i, j] = Z_primitive[i, j] - (
                    Z_primitive[i, self.phase_count] * 
                    Z_primitive[self.phase_count, j] / 
                    Z_primitive[self.phase_count, self.phase_count]
                )
                
                # Reduce P matrix (shunt admittance inverse)
                P[i, j] = P_primitive[i, j] - (
                    P_primitive[i, self.phase_count] * 
                    P_primitive[self.phase_count, j] / 
                    P_primitive[self.phase_count, self.phase_count]
                )
            
            # Calculate neutral current factors
            tn[i] = -(
                Z_primitive[self.phase_count, i] / 
                Z_primitive[self.phase_count, self.phase_count]
            )
        
        # Calculate C matrix through inverse of P matrix
        C = np.linalg.inv(P)
        # Calculate Y matrix through C matrix
        omega = 2.0 * math.pi * self.frequency

        
        Y = np.zeros((self.phase_count, self.phase_count), dtype=complex)
        for i in range(self.phase_count):
            for j in range(self.phase_count):
                Y[i, j] = complex(
                    -C[i, j].imag * omega,  # Conductance
                    C[i, j].real * omega    # Susceptance
                )

        return Z, Y, tn


    def build(self):
        """Build the line model from the added conductors"""
        if len(self.conductors) < 2:
            raise ValueError("At least two conductors are required (phase + neutral)")
        self.size = len(self.conductors)
        self.phase_count = len(self.conductors) - 1
        (Z_primitive, P_primitive) = self.build_primitive_matrices()
        (self.Z, self.Y, self.tn) = self.reduce_primitives(Z_primitive, P_primitive)
        self.type = "Overhead"


class LineObject:
    """Object for storing line model data"""
    def __init__(self, builder):
        """Initialize a line object from a LineBuilder"""     
        # Identity matrix of phase size
        u = np.eye(builder.phase_count, dtype=complex)

        # Operation: a = (1/2)ZY + u
        self.a = 0.5 * np.matmul(builder.Z, builder.Y) + u
        
        # Operation: b = Z
        self.b = builder.Z

        # Operation: Matrix inversion A = a^(-1)
        self.A = np.linalg.inv(self.a)

        # Operation: B = Ab
        self.B = np.matmul(self.A, self.b)
        