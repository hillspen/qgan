import pennylane as qml
import numpy as np

def generator(thetas, res):
    for i in range(res):
        qml.RY(thetas[i], wires=i+1+res)

    for i in range(res-1):
        qml.IsingYY(thetas[i+res], wires=[i+1+res, i+2+res])
    
    for i in range(res):
        if (i+2)%res == 0:
            qml.CRY(thetas[i+2*res-1], wires=[i+1+res, 2*res])
        else:
            qml.CRY(thetas[i+2*res-1], wires=[i+1+res, ((i+2)%res)+res])

def discriminator(thetas, res):
    for i in range(res):
        qml.RY(thetas[i], wires=i+1)
    
    for i in range(res-1):
        qml.IsingYY(thetas[i+res], wires=[i+1, i+2])
    
    for i in range(res):
        if (i+2)%res == 0:
            qml.CRY(thetas[i+2*res-1], wires=[i+1, res])
        else:
            qml.CRY(thetas[i+2*res-1], wires=[i+1, (i+2)%res])

res = 4
numQubits = 2 * res + 1
numThetas = 3 * res - 1
dev = qml.device('default.qubit', wires=numQubits)

@qml.qnode(dev)
def genCircuit(thetasD, thetasG, res, real=True):
    qml.Hadamard(wires=0)
    discriminator(thetasD, res)

    if not real:
        generator(thetasG, res)

    for i in range(res):
        qml.CSWAP(wires=[0, i+1, i+1+res])

    return qml.expval(qml.PauliZ(0))


thetasD = np.pi * np.ones(numThetas)
thetasG = (np.pi / 2.0) * np.ones(numThetas)

drawer = qml.draw(genCircuit)
print(drawer(thetasD, thetasG, res, real=False))