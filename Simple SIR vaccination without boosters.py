import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
import math
from statistics import mean


def SIR_model(y, t, b, g, d, v):
    S, I, R = y
    dS = -b * S * I + d * R - v * S
    dI = b * S * I - g * I
    dR = g * I - d * R + v * S
    return [dS, dI, dR]


def no_vax_SIR_model(y, t, b, g, d):
    S, I, R = y
    dS = -b * S * I + d * R
    dI = b * S * I - g * I
    dR = g * I - d * R
    return [dS, dI, dR]


beta = 0.35  # rate of infection
gamma = 0.1  # recovery rate
delta = 1 / 180  # rate of loss of immunity
vax_r = 1 / 90  # vaccination rate
days = 600  # how many days to model
N = 5e6
T = np.linspace(0, days, days + 1)
hospitalisation_rate = 0.2

days_pre = 180  # days until outbreak
days_remaining = days - days_pre
S0, I0, R0 = 1, 0, 0  # initial conditions
T_pre = np.linspace(0, days_pre - 1, days_pre)

pre_vax = odeint(SIR_model, [S0, I0, R0], T_pre,
                 args=(beta, gamma, delta, vax_r))

S1, I1, R1 = pre_vax[-1:, 0][0], 1 / N, pre_vax[-1:, 2][0]  # initial conditions
T_post = np.linspace(0, days_remaining, days_remaining + 1)

post_vax = odeint(no_vax_SIR_model, [S1, I1, R1], T_post,
                  args=(beta, gamma, delta))

solution = np.vstack((pre_vax, post_vax))

print("Final Numbers \nSusceptable: ", solution[-1:, 0], "\nInfected: ",
      solution[-1:, 1], "\nRemoved: ", solution[-1:, 2], "\nPeak infections: ",
      max(solution[:, 1]) * N, "\nPeak immunity: ", max(solution[:, 2]) * N,
      "\nPeak in hospital: ", max(solution[:, 1] * N * hospitalisation_rate))

plt.figure('Simple SIR with Vaccination, no boosters and Waning Immunity',
           figsize=[12, 5])
plt.plot(T, solution[:, 0], label="Susceptable(t)")
plt.plot(T, solution[:, 1], label="Infected(t)")
plt.plot(T, solution[:, 2], label="Removed(t)")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Proportion")
plt.title('Simple SIR with Vaccination, no boosters and Waning Immunity \n'
          '180 days until outbreak, vaccination rate 1/90, ceasing at 180days')
plt.show()
