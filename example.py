import simulator
import numpy as np
from math import pi
import random
import math

qc = simulator.get_ground_state(2)
circuit = [{"gate":"U3","target":[0],"params":{"theta":"global1","phi":"global2","lambda":pi/5}},
           {"gate":"U1","target":[0],"params":{"lambda":pi}}]
result_vector = simulator.run_program(qc,circuit,{"global1":pi/2,"global2":pi/4})
result = simulator.get_counts(result_vector,1000)
print(result)

unitary_gate = simulator.get_operator(1,"RX",[0],{"theta": pi})
print(unitary_gate)





