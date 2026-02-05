import matplotlib.pyplot as plt
import numpy as np
from mpmath import ellipk

#Constant values defined, the program is currently defined for the 20x20 system
dimension = 20
J = 1
h = 0.001
iterations = dimension**2 * 1700
threshold = 300000
n = iterations - threshold
kb = 1
Temperatures = list(np.linspace(1,4,10))

#ONSAGER'S SOLUTION

#Critical temperature
T_c = 2*J / np.log(1+2**(1/2))

#Magnetization
def ons_mag(Temperatures):
    mag = []
    for T in Temperatures:
        if T<T_c:
            K = J/T
            mag.append((1 - np.sinh(2*K)**(-4))**(1/8))
        else:
            mag.append(0)
    return mag

#Energy
def ons_energy(Temperatures):
    energy = []
    for T in Temperatures:
        K = J/T
        kappah = 2*np.sinh(2*K) / (np.cosh(2*K)**2)
        el = ellipk(kappah**2)
        energy.append(-J * (np.cosh(2*K)/np.sinh(2*K))*(1+(2/np.pi)*(2*np.tanh(2*K)**2 -1)*el))
    return energy

#MONTECARLO SIMULATION

#Hot start initial lattice
def setup(dimension):
    matrix = np.random.choice([-1,1], size = [dimension,dimension])
    return matrix

#Cold start initial matrix
def coldstart(dimension):
    return np.ones((dimension,dimension))

#System change mechanism defined
class system:
    def __init__(self,matrix):
        self.matrix = matrix
        self.dim = len(self.matrix)
    def change(self,i,j):
        self.matrix[i][j] = -self.matrix[i][j]

#Method of energy and magnetization computation
class Ising:
    def __init__(self,system):
        self.E = self.energy(system)
        self.mag = self.magnetization(system)
    
    def energy(self,system):
        H = 0
        dim = system.dim
        for i in range(dim):
            for j in range(dim):
                H-= J*system.matrix[i][j]*system.matrix[(i+1)%dim][j]  - h * system.matrix[i][j]
                H-= J*system.matrix[i][j]*system.matrix[i][(j+1)%dim]
        return H
    
    def magnetization(self,system):
        return sum(sum(k) for k in system.matrix)

#Method of time evolution and acceptance conditions
class montecarlo:
    def __init__(self,system,model):
        self.system = system
        self.dimension = system.dim
        self.model = model
        self.E = model.energy(system)
        self.mag = model.magnetization(system)

    def metropolis(self, dE, probs):
        if dE <= 0: return True
        return np.random.random() < probs.get(dE, 0)

        
    def evolution(self,T,probs):
        dim = self.dimension
        i = np.random.randint(0,dim)
        j = np.random.randint(0,dim)
        spins = self.system.matrix
        S = spins[(i+1)%dim, j] + spins[(i-1)%dim, j] + spins[i, (j+1)%dim] + spins[i, (j-1)%dim]     

        dE = 2*J*spins[i][j] * S + 2 * h * spins[i][j]

        if self.metropolis(dE,probs):
            self.system.change(i,j)
            self.E += dE
            self.mag += 2*self.system.matrix[i][j]

#Method of the computation of autocorrelation times
def autocorr_fft(x):
    x = np.asarray(x)
    x = x - np.mean(x)
    N = len(x)

    f = np.fft.fft(x, n=2*N)
    acf = np.fft.ifft(f * np.conjugate(f))[:N].real
    return acf/acf[0]

def tau(rho, c=6):
    if np.isnan(rho).any() or len(rho) < 2:
        return 0.5
    W = 1
    tau = 0.5 + np.sum(rho[1:W])
    W_new = int(c * tau)

    while W_new > W and W_new < len(rho):
        W = W_new
        tau = 0.5 + np.sum(rho[1:W])
        W_new = int(c * tau)

    return tau

average_E = []
average_mag = []
average_Cv = []
average_susc = []
errors_E = []
errors_mag = []
errors_Cv = []
errors_susc = []

#Definition of the initial system and method of evolution, the data is taken only every N^2 steps
system_1 = system(setup(dimension))
model_1 = Ising(system_1)
simulation = montecarlo(system_1,model_1)
data_size = n/400

#Loop for a number of temperatures around the critical one
for T in Temperatures:
    probs = {4: np.exp(-4*J/(kb*T)), 8: np.exp(-8*J/(kb*T))}
    energies = []
    magnetizations = []
    sq_E = []
    sq_mag = []


    #Time evolution of the system
    for i in range(iterations):
        simulation.evolution(T,probs)

        if i>threshold and i%400 == 0:
            energies.append(simulation.E)
            magnetizations.append(abs(simulation.mag))
            sq_E.append(simulation.E**2)
            sq_mag.append(simulation.mag**2)

    #Computation of averages and errors
    av_E = sum(energies)/data_size
    av_mag = sum(magnetizations)/data_size
    av_E2 = sum(sq_E)/data_size
    av_mag2 = sum(sq_mag) / data_size

    rho_E = autocorr_fft(energies)
    tau_E = tau(rho_E)
    rho_mag = autocorr_fft(magnetizations)
    tau_mag = tau(rho_mag)

    var_E = av_E2 - (av_E**2)
    var_mag = av_mag2 - (av_mag**2)

    Cv = var_E / (kb * T**2 * dimension**2)
    susc = var_mag / (kb * T * dimension**2)

    err_E = np.sqrt(var_E * 2 * tau_E / data_size) / dimension**2
    err_mag = np.sqrt(var_mag * 2 * tau_mag / data_size) / dimension**2
    
    err_Cv = Cv * np.sqrt(2 * (2 * tau_E) / data_size)
    err_susc = susc * np.sqrt(2 * (2 * tau_mag) / data_size)

    errors_E.append(err_E)
    errors_mag.append(err_mag)
    errors_Cv.append(err_Cv)
    errors_susc.append(err_susc)
    average_E.append(av_E/(dimension**2))
    average_mag.append(av_mag/(dimension**2))
    average_Cv.append(Cv)
    average_susc.append(susc)

#Function for better visual plots

plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

def plot(x, y, y_error, y_teoric, title, ylabel, label_sim="Simulation"):
    plt.figure(figsize=(10, 6))
    
    if y_teoric is not None:
        plt.plot(x, y_teoric, color='black', linewidth=1.5, label="Onsager solution", zorder=1)
    
    plt.fill_between(x, np.array(y) - np.array(y_error), np.array(y) + np.array(y_error), 
                     color='#1f77b4', alpha=0.2, label="Statistical error")
    
    plt.scatter(x, y, s=10, color='#1f77b4', label=label_sim, zorder=3)
    
    plt.axvline(T_c, color='red', linestyle=':', alpha=0.6, label=f'Tc ≈ {T_c:.3f}')
    
    plt.xlabel("Temperature")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

plot(Temperatures, average_E, errors_E, ons_energy(Temperatures), 
           "Mean energy per spin", r"$\langle E \rangle / N^2$")
plot(Temperatures, average_mag, errors_mag, ons_mag(Temperatures), 
           "Mean magnetization per spin", r"$\langle m \rangle / N^2$")
plot(Temperatures, average_susc, errors_susc, None, 
           "Mean susceptibility per spin", r"$\chi$")
plot(Temperatures, average_Cv, errors_Cv, None, 
           "Mean heat capacity per spin", r"Cv")
