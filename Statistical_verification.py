import matplotlib.pyplot as plt
import numpy as np
import time
from mpmath import ellipk

dimension = 20
J = 1
iterations = 400*800
kb = 1
T = 2.2



#Montecarlo simulation
def setup(dimension):
    matrix = []
    for i in range(dimension):
        line_i = []
        for j in range(dimension):
            line_i.append(int(np.random.choice([-1,1])))
        matrix.append(line_i)
    return matrix

class system:
    def __init__(self,matrix):
        self.matrix = matrix
        self.dimension = len(self.matrix)
    def change(self,i,j):
        self.matrix[i][j] = -self.matrix[i][j]
    
    def neighbors(self,i,j):
        dim = self.dimension
        return [(i+1)%dim,(i-1)%dim,(j+1)%dim,(j-1)%dim]

class Ising:
    def __init__(self,system):
        self.E = self.energy(system)
        self.mag = self.magnetization(system)
    
    def energy(self,system):
        H = 0
        dim = system.dimension
        for i in range(dim):
            for j in range(dim):
                H-= J*system.matrix[i][j]*system.matrix[(i+1)%dim][j] 
                H-= J*system.matrix[i][j]*system.matrix[i][(j+1)%dim]
        return H
    
    def magnetization(self,system):
        return sum(sum(k) for k in system.matrix)

class montecarlo:
    def __init__(self,system,model):
        self.system = system
        self.dimension = system.dimension
        self.model = model
        self.E = model.energy(system)
        self.mag = model.magnetization(system)

    def metropolis(self,dE,T):
        if dE < 0:
            return True
        elif np.random.random() < np.e**(-dE/(kb*T)):
            return True
        else:
            return False
        
    def evolution(self,T):
        dim = self.dimension
        i = np.random.randint(0,dim)
        j = np.random.randint(0,dim)
        spins = self.system.matrix
        neighbors = self.system.neighbors(i,j)

        dE = 2*J*spins[i][j] * (self.system.matrix[neighbors[0]][j] + self.system.matrix[neighbors[1]][j]+self.system.matrix[i][neighbors[2]] + self.system.matrix[i][neighbors[3]])

        if self.metropolis(dE,T):
            self.system.change(i,j)
            self.E += dE
            self.mag += 2*self.system.matrix[i][j]


#THERMALIZATION TIME

#Generate two hot starts and two cold starts
systems = []
systems.append(system(setup(dimension)))
systems.append(system(setup(dimension)))
systems.append(system([[-1 for i in range(dimension)] for i in range(dimension)]))
systems.append(system([[1 for i in range(dimension)] for i in range(dimension)]))



for syst in systems:

    model_1 = Ising(syst)
    simulation = montecarlo(syst,model_1)
    energies = []
    magnetizations = []

    for i in range(iterations):
        simulation.evolution(T)
        energies.append(simulation.E)
        magnetizations.append(simulation.mag)
    plt.plot(np.linspace(0,iterations,iterations),magnetizations)
plt.xlabel("Step")
plt.ylabel("Energy")
plt.show()

