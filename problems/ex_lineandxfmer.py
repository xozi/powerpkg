'''
A very simple distribution feeder is shown in Figure 7. 
For the system in Fig.7, the infinite bus voltages are balanced three phase of 12.47 kV line to line. 
The “source” line segment from node 1 to node 2 is a three-wire delta 2000 ft long line and is constructed on the pole configuration of as shown without the neutral. 
The “load” line segment from node 3 to node 4 is 2500 ft long and also is constructed on the pole configuration as shown but is a four-wire wye so the neutral is included. 
Both line segments use 336,400 26/7 ACSR phase conductors and the neutral conductor on the four-wire wye line is 4/0 6/1 ACSR. 
Since the lines are short, the shunt admittance will be neglected. The 25°C resistance is used for the phase and neutral conductors:
336,400 26/7 ACSR: 0.278 Ω/mile (25 C)
4/0 6/1 ACSR: 0.445 Ω/mile (25 C)

The transformer bank is connected delta–grounded wye and consists of three single-phase transformers each rated:
2000 𝑘𝑉𝐴, 12.47−2.4 𝑘𝑉, 𝑍=1.0+𝑗6.0%
12.47kV infinite bus is source Delta (VLL), we have a load at the end with the values:
PQ:
Sa = 750 kVA at 0.85 lagging
Sb = 1000 kVA at 0.90 lagging
Sc = 1230 kVA at 0.95 lagging


'''
from src.line import *
from src.xfmer import *
from src.system import *
from src.sweep import *
import numpy as np
#2000/2500 ft line, 5280 ft per mile
scale1 = 2000/5280
scale2 = 2500/5280;
#  def add_conductor(self, resistance, gmr, x_pos, y_pos, diameter):
#Delta Line
line_builder1 = OverheadLineBuilder()
line_builder1.add_conductor(0.278, 0.0244, 0.0, 29.0, 0.721)  # Phase A
line_builder1.add_conductor(0.278, 0.0244, 2.5, 29.0, 0.721)  # Phase B
line_builder1.add_conductor(0.278, 0.0244, 7.0, 29.0, 0.721)  # Phase C
line_builder1.add_scale(scale1)
line_builder1.build(False)

#No Shunt Admittance
line_builder1.Y = np.zeros((3,3), dtype=complex)
Line1 = LineObject(line_builder1)




#Wye Line
line_builder2 = OverheadLineBuilder()
line_builder2.add_conductor(0.278, 0.0244, 0.0, 29.0, 0.721)  # Phase A
line_builder2.add_conductor(0.278, 0.0244, 2.5, 29.0, 0.721)  # Phase B
line_builder2.add_conductor(0.278, 0.0244, 7.0, 29.0, 0.721)  # Phase C
line_builder2.add_conductor(0.445, 0.0081, 4.0, 25.0, 0.563)  # Neutral
line_builder2.add_scale(scale2)
line_builder2.build(True)

#No Shunt Admittance
line_builder2.Y = np.zeros((3,3), dtype=complex)
Line2 = LineObject(line_builder2)

#Xfmer
# def __init__(self, rated_power, primary_voltage, secondary_voltage):
xfmer = Xfmer(2000e3, 12.47e3, 2.4e3)
xfmer.add_Zpu(1.0 + 1j * 6.0)
xfmer.build()

#Add all single phase transformers
xfmer1 = DeltaWyeGrounded(xfmer,xfmer,xfmer)

#Loads
Sload = [PQLoad(750e3, None, 0.85).S, PQLoad(1000e3, None, 0.90).S, PQLoad(1250e3, None, 0.95).S]


#VLL
V1_LL = VLL_Source([12.47e3, 12.47e3, 12.47e3])

#VLN
V1_LN = VLN_DeltaSource([12.47e3, 12.47e3, 12.47e3], False)


#Initial Conditions
V4_LN = [1, 1, 1]
V3_LN = [1, 1, 1]
V2_LN = [1, 1, 1]
tol = 1;
max_iter = 1000;
iter = 0;
V4_prev = [0, 0, 0];
Ip = np.zeros(3, dtype=complex)
Is = np.zeros(3, dtype=complex)
while tol > 1e-6 and iter < max_iter:
    #Sweep Equation
    V2_LN = sweep_vol(Line1.A, Line1.B, V1_LN.V, Ip)
    V3_LN = sweep_vol(xfmer1.At, xfmer1.Bt, V2_LN, Is)
    V4_LN = sweep_vol(Line2.A, Line2.B, V3_LN, Is)

    tol = np.max(np.abs(abs(max(V4_LN)) - abs(max(V4_prev)))/2.4e3)
    
    #Max Iterations Break
    if iter == max_iter - 1:
        break
    
    iter += 1

    for i in range(3):
        Is[i] = np.conj(Sload[i] / V4_LN[i])
 
    V4_prev = np.copy(V4_LN)
    Ip = np.matmul(xfmer1.dt, Is)

if tol < 1e-6:
    print("Converged")
else:
    print("Not converged")

#120/2.4e3
Vload =V4_LN*(120/2.4e3)
(vol,angle) = toPolar(Vload)
print("Voltages\n {}".format(vol))
print("Angles\n {}".format(angle))





