# MOSinLine

In this repo we integrate the implementations of the ASBP for the RLRP (Johannes part) with the ALNS for the Delivery Pattern Optimization (Kailin) and the Simulation.
The goal was to keep as much of the code untouched as possible. We just had to slightly modify the ALNS in order to read all necessary info from ythe instance/input file.

The main.py contains all the data structures and control flows that connect everything. For each part (RLRP, PATT, SIM) there is a Results class which contains the relevant result of the algorithm, and, for RLRP and PATT, can be passed to the next step of the algorithm.

For the PATT (alns), the code creates a temporary instance file which is read by the algorithm, and then deletes this file again.