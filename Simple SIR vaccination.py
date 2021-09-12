import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def SIR_model(y, t, b, g, d, v):
    S, I, R = y
    dS = -b * S * I + d * R - v * S
    dI = b * S * I - g * I
    dR = g * I - d * R + v * S
    return [dS, dI, dR]


S0 = 0.99  # initial Susceptible Population
I0 = 1 - S0  # initial infected
R0 = 1 - S0 - I0  # initial removed
beta = 0.35  # rate of infection
gamma = 0.1  # recovery rate
vax_r = 1 / 90  # vaccination rate
days = 1000  # how many days to model
T = np.linspace(0, days, days + 1)

solution = odeint(SIR_model, [S0, I0, R0], T, args=(beta, gamma))

plt.figure("Simple SIR with Vaccination and Waning Immunity", figsize=[8, 6])
plt.plot(T, solution[:, 0], label="S(t)")
plt.plot(T, solution[:, 1], label="I(t)")
plt.plot(T, solution[:, 2], label="R(t)")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Proportion")
plt.title('SIR Model')
plt.show()
