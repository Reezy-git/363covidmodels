import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def SIR_model(y, t, b, g, d, v):
    S, I, R = y
    dS = -b * S * I + d * R - v * S
    dI = b * S * I - g * I
    dR = g * I - d * R + v * S
    return [dS, dI, dR]


S0, I0, R0 = 1.0, 0.01, 0.0  # initial conditions
beta = 0.35  # rate of infection
gamma = 0.1  # recovery rate
delta =1/148  # rate of loss of immunity
vax_r = 1/90  # vaccination rate
days = 400  # how many days to model

T = np.linspace(0, days, days + 1)

solution = odeint(SIR_model, [S0, I0,R0], T, args=(beta, gamma, delta, vax_r))
print("Final Numbers \nSusceptable: ", solution[-1:,0], "\nInfected: ", 
      solution[-1:,1], "\nRemoved: ", solution[-1:,2])

plt.figure('Simple SIR with Vaccination and Waning Immunity', figsize=[8, 5])
plt.plot(T, solution[:, 0], label="Susceptable(t)")
plt.plot(T, solution[:, 1], label="Infected(t)")
plt.plot(T, solution[:, 2], label="Removed(t)")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Proportion")
plt.title('Simple SIR with Vaccination and Waning Immunity')
plt.show()
