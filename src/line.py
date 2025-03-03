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

    def build(self):
        """Build the line model from the added conductors"""
        if len(self.conductors) < 2:
            raise ValueError("At least two conductors are required (phase + neutral)")
        self.size = len(self.conductors)

class OverheadLineObject(OverheadLineBuilder):
    def __init__(self, builder):
        """Initialize with an optional builder"""
        if builder is None:
            raise ValueError("Builder is required");
        else:
            (Z_primitive, P_primitive) = self.build_primitive_matrices(builder)
            (Z, Y, tn) = reduce_primitives(Z_primitive, P_primitive, builder)
            self.type = "Overhead"
            self.Z = Z
            self.Y = Y
            self.tn = tn

    def build_primitive_matrices(self, builder):
        """Build the primitive impedance and potential coefficient matrices"""
        Z_primitive = np.zeros((builder.size, builder.size), dtype=complex)
        P_primitive = np.zeros((builder.size, builder.size), dtype=complex)
        for i in range(builder.size):
            for j in range(builder.size):
                if i == j:
                    Z_primitive[i, j], P_primitive[i, j] = self.calculate_self_impedance(i, builder)
                else:
                    Z_primitive[i, j], P_primitive[i, j] = self.calculate_mutual_impedance(i, j, builder)
        return Z_primitive, P_primitive
    
    def calculate_self_impedance(self, i, builder):
        """Calculate self impedance for conductor i"""
        resistance = builder.conductors[i]['resistance']
        Sij = 2.0 * builder.conductors[i]['y']
        Dij = builder.conductors[i]['gmr']
        # Convert diameter to radius into inches
        rd = (builder.conductors[i]['diameter'] / 2.0) / 12.0
        
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
    
    def calculate_mutual_impedance(self, i, j, builder):
        """Calculate mutual impedance between conductors i and j"""
        # Direct distance between conductors
        Dij = math.sqrt(
            (builder.conductors[j]['x'] - builder.conductors[i]['x'])**2 + 
            (builder.conductors[j]['y'] - builder.conductors[i]['y'])**2
        )
        
        # Distance including earth return (image conductor)
        Sij = math.sqrt(
            (builder.conductors[j]['x'] - builder.conductors[i]['x'])**2 + 
            (builder.conductors[i]['y'] + builder.conductors[j]['y'])**2
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

def reduce_primitives(Z_primitive, P_primitive, builder):
    """Kron reduce primitive matrices, get tn, and additional operations for Y matrix"""
    final_size = builder.size - 1
    Z = np.zeros((final_size, final_size), dtype=complex)
    P = np.zeros((final_size, final_size), dtype=complex)
    tn = np.zeros(final_size, dtype=complex)
    
    # Kron reduction
    for i in range(final_size):
        for j in range(final_size):
            # Reduce Z matrix (series impedance)
            Z[i, j] = Z_primitive[i, j] - (
                Z_primitive[i, final_size] * 
                Z_primitive[final_size, j] / 
                Z_primitive[final_size, final_size]
            )
            
            # Reduce P matrix (shunt admittance inverse)
            P[i, j] = P_primitive[i, j] - (
                P_primitive[i, final_size] * 
                P_primitive[final_size, j] / 
                P_primitive[final_size, final_size]
            )
        
        # Calculate neutral current factors
        tn[i] = -(
            Z_primitive[final_size, i] / 
            Z_primitive[final_size, final_size]
        )
    
    # Calculate C matrix through inverse of P matrix
    C = np.linalg.inv(P)
    # Calculate Y matrix through C matrix
    omega = 2.0 * math.pi * builder.frequency

    
    Y = np.zeros((final_size, final_size), dtype=complex)
    for i in range(final_size):
        for j in range(final_size):
            Y[i, j] = complex(
                -C[i, j].imag * omega,  # Conductance
                C[i, j].real * omega    # Susceptance
            )

    return Z, Y, tn

