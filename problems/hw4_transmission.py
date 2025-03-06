'''
Consider the two-bus power system shown in the figure above. The system consists of:

A slack bus (Bus 1) with a voltage magnitude of 1 per unit (p.u.) and a zero voltage angle.
A load bus (Bus 2), where active power (P) and reactive power (Q) are specified.
A transmission line connecting Bus 1 and Bus 2, modeled as an impedance of Z=j0.1 p.u.
System Base Power: Sbase=100 MVA
Tasks:
1. Formulate the power flow equations using the Newton-Raphson method for this system.
2. Determine the unknown variables:
    > Voltage magnitude and angle at Bus 2.
3. Solve the power flow problem iteratively using the Newton-Raphson method.
4. Discuss the convergence of the NR method for this simple two-bus system.
5. Validate your results using either PowerWorld or MATPOWER.
'''
from src.transmission import Transmission, Node, Line

transmission = Transmission(Sbase=100e6)

#Add nodes
slack = Node(type="slack")
load = Node(type="PQ", P=-200e6/transmission.Sbase, Q=-100e6/transmission.Sbase)

#Add line
transmission.add_line(Line(Z=0.1j, fromNode=slack, toNode=load))
transmission.add_node(slack)
transmission.add_node(load)

#Build
transmission.build()

print(transmission.voltages)
print(transmission.angles)
