import numpy as np
import random
import math
from math import pi,sqrt,cos,sin

gates = {"X":[[0,1],[1,0]], 
        "H": [[1/sqrt(2),1/sqrt(2)],[1/sqrt(2),-1/sqrt(2)]],
        "I": [[1,0],[0,1]],
        "Z": [[1,0],[0,-1]],
        "Y": [[0,-1j],[1j,0]],
        "S": [[1,0],[0,1j]],
        "T": [[1,0],[0,np.exp(1j*(pi/4))]],
        "Sdag": [[1,0],[0,-1j]], 
        "Tdag": [[1,0],[0,np.exp(-1j*(pi/4))]],
        "0><0": [[1,0],[0,0]],
        "1><1": [[0,0],[0,1]]}


#Helper functions start
def Log2(x):
    if x == 0:
        return false
    return (math.log10(x)/math.log10(2))
def isPowerOfTwo(n):
    return (math.ceil(Log2(n)) == math.floor(Log2(n)))
def decimalToBinary(n): 
    return bin(n).replace("0b", "")
#Helper functions end

def get_ground_state(num_qubits):
    state = np.zeros(2**num_qubits)
    state[0]=1
    return state

def measure_state(state):
    all_states = np.size(state) 
    assert isPowerOfTwo(all_states),"statevector doesn't have amplitudes for all possible states"
    probab = np.abs(state)**2
    total_weight=np.sum(probab)             #sum of probabilities
    #rounded for cases such as 0.99.., change second parameter according to precision required, 3 for 0.999..
    total_weight = round(total_weight,2)    
    assert total_weight==1,"statevector is not normalized"
    probab = 100*probab
    rnd = random.randint(0,99)
    for i in range(all_states):
        if rnd<probab[i]:
            return decimalToBinary(i)
        rnd = rnd - probab[i]
    assert "something is wrong with statevector"

def get_counts(state_vector, num_shots):
    all_states = np.size(state_vector)
    result_counts = {}
    for i in range(all_states):
        result_counts[decimalToBinary(i)] = 0
    for i in range(num_shots):
        measure_ans = measure_state(state_vector)
        if measure_ans is not None:
            result_counts[measure_ans] += 1
    return result_counts

def U1(lamda): 
    return [[1,0],[0,np.exp(1j*lamda)]]
def U2(lamda,phi): 
    return [[1/sqrt(2) , (-1*np.exp(1j*lamda))/sqrt(2)] , [np.exp(1j*phi)/sqrt(2) , np.exp(1j*(lamda+phi))/sqrt(2)]]
def U3(lamda,phi,theta): 
    return [[cos(theta/2) , -1*np.exp(1j*lamda)*sin(theta/2)] , [np.exp(1j*phi)*sin(theta/2) , np.exp(1j*(lamda+phi))*cos(theta/2)]]
def RX(theta):
    return [[cos(theta/2) , -1j*sin(theta/2)],[-1j*sin(theta/2) , cos(theta/2)]]
def RY(theta):
    return [[cos(theta/2) , -1*sin(theta/2)],[sin(theta/2) , cos(theta/2)]]

def get_parameterized_gate(gate,params,global_params={}):
    numParams = len(params)
    assert (numParams>=1 and numParams<=3), "unitary quantum gate has only three parameters: lambda, theta, phi"
    if gate=='U1':
        assert ("lambda" in params), "lambda required for U1"
        params["lambda"] = eval(str(params["lambda"]),global_params)
        return U1(params["lambda"])
    elif gate=='U2':
        assert ("lambda" in params and "phi" in params), "lambda and phi required for U2"
        params["lambda"] = eval(str(params["lambda"]),global_params)
        params["phi"] = eval(str(params["phi"]),global_params)
        return U2(params["lambda"],params["phi"])
    elif gate=='U3':
        assert ("lambda" in params and "phi" in params and "theta" in params), "lambda, phi and theta required for U3"
        params["lambda"] = eval(str(params["lambda"]),global_params)
        params["phi"] = eval(str(params["phi"]),global_params)
        params["theta"] = eval(str(params["theta"]),global_params)
        return U3(params["lambda"],params["phi"],params["theta"])
    elif gate=='RX':
        assert ("theta" in params), "theta required for RX"
        return RX(params["theta"])
    else:
        assert gate=='RY', "Gate not available in simulator"
        assert ("theta" in params), "theta required for RY"
        return RY(params["theta"])
    
def transform(state_vector, gate):
    state_vector = np.dot(gate,state_vector)
    return state_vector

def get_operator(total_qubits,gate_unitary,target_qubits,params={},global_params={}):
    assert len(target_qubits)>=1,"No target qubits specified"
    for i in target_qubits:
        assert i<total_qubits, "Target qubit not present in the circuit"
    if len(target_qubits)==1:
        if (gate_unitary in ["U3","U2","U1","RX","RY"]):
            gate = get_parameterized_gate(gate_unitary,params,global_params)
        else:
            gate = gates[gate_unitary]
        return operator_one_qubit(total_qubits,gate,target_qubits)
    else:
        assert gate_unitary=="CNOT", "Gate not available in simulatorc"
        return operator_CNOT(total_qubits,target_qubits)
    
def operator_CNOT(total_qubits,target_qubits):
    if target_qubits[0]==0:
        operator1=gates["0><0"]
        operator2=gates["1><1"]
    elif target_qubits[1]==0:
        operator1=gates["I"]
        operator2=gates["X"]
    else:
        operator1=gates["I"]
        operator2=gates["I"]
    for i in range(1,total_qubits):
        if target_qubits[0]==i:
            operator1=np.kron(operator1,gates["0><0"])
            operator2=np.kron(operator2,gates["1><1"])
        elif target_qubits[1]==i:
            operator1=np.kron(operator1,gates["I"])
            operator2=np.kron(operator2,gates["X"])
        else:
            operator1=np.kron(operator1,gates["I"])
            operator2=np.kron(operator2,gates["I"])
    return operator1+operator2

def operator_one_qubit(total_qubits,gate_unitary,target_qubits):
    assert len(target_qubits)>=1,"No target qubits specified"
    if target_qubits[0]==0:
        operator=gate_unitary
    else:
        operator=gates["I"]
    for i in range(1,total_qubits):
        if i==target_qubits[0]:
            operator=np.kron(operator,gate_unitary)
        else:
            operator=np.kron(operator,gates["I"])
    return operator

def run_program(state_vector, circuit,global_params={}):
    all_states = np.size(state_vector) 
    assert isPowerOfTwo(all_states),"statevector doesn't have amplitudes for all possible states"
    total_qubits = math.floor(Log2(all_states))
    for i in circuit:
        if "params" in i:
            operator = get_operator(total_qubits,i["gate"],i["target"],i["params"],global_params)
        else:
            operator = get_operator(total_qubits,i["gate"],i["target"])
        state_vector = transform(state_vector,operator)
    return state_vector

