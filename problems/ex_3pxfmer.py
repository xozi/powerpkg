"""
In the example system of Figure 4, an unbalanced constant impedance load is being served 
at the end of a 1 mile section of a three-phase line. The 1 mile long line is being fed 
from a substation transformer rated 5000 kVA, 115 kV delta–12.47 kV grounded wye with 
a per-unit impedance of 0.085 ∠ 85. 

The phase conductors of the line are 336,400 26/7 ACSR 
with a neutral conductor 4/0 ACSR. The configuration and 
computation of the phase impedance matrix are given in Example 4.1. 
From that example, the phase impedance matrix was computed to be

[〖𝑍𝑙𝑖𝑛𝑒〗_𝑎𝑏𝑐 ]=[(
0.4576+𝑗1.07800.1560+𝑗0.5017 0.1535+𝑗0.3849 
0.1560+𝑗0.5017 0.4666+𝑗1.0482 0.1580+𝑗0.4236
0.1535+𝑗0.3849 0.1580+𝑗0.4236 0.4615+𝑗1.0651)] Ω/𝑚𝑖𝑙𝑒


"""
from src.xfmer import *
import numpy as np

Z_line = np.array([
    [0.4576 + 1j * 1.0780, 0.1560 + 1j * 0.5017, 0.1535 + 1j * 0.3849],
    [0.1560 + 1j * 0.5017, 0.4666 + 1j * 1.0482, 0.1580 + 1j * 0.4236],
    [0.1535 + 1j * 0.3849, 0.1580 + 1j * 0.4236, 0.4615 + 1j * 1.0651]
])

xfmer = Xfmer(115000, 12470, 0.085, 85)


