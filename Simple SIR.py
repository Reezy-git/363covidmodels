import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def SIR_model(y, t, b, g):
    S, I, R = y
    dS = -b * S * I
    dI = b * S * I - g * I
    dR = g * I
    return [dS, dI, dR]


S0 = 0.90  # initial Susceptible Population
I0 = 1 - S0  # initial infected
R0 = 1 - S0 - I0  # initial removed
beta = 0.35  # rate of infection
gamma = 0.1  # recovery rate
T = np.linspace(0, 100, 101)
N = 5e6
hospitalisation_rate = 0.2

solution = odeint(SIR_model, [S0, I0,R0], T, args=(beta, gamma))

print("Final Numbers \nSusceptable: ", solution[-1:, 0], "\nInfected: ",
      solution[-1:, 1], "\nRemoved: ", solution[-1:, 2], "\nPeak infections: ",
      max(solution[:, 1]) * N, "\nPeak immunity: ", max(solution[:, 2]) * N,
      "\nPeak in hospital: ", max(solution[:, 1] * N * hospitalisation_rate))

plt.figure(figsize=[8, 6])
plt.plot(T, solution[:, 0], label="S(t)")
plt.plot(T, solution[:, 1], label="I(t)")
plt.plot(T, solution[:, 2], label="R(t)")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Proportion")
plt.title('Basic Case SIR Model')
plt.show()
