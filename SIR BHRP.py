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


beta = 0.35  # rate of infection
gamma = 0.1  # recovery rate
delta = 1 / 208  # rate of loss of immunity
vax_r = 1 / 180  # vaccination rate
days = 1000  # how many days to model
CENZ = 6e5 * 14 * 4000 / 7.67e9  # Covid cases entering nz / day
BVR = 0.9  # Boarder vaccination rate
BTPr = 1 / 1e2  # Boarder worker leak/exposure probability, with 3 occurrences in a year this
                # is close to 1 / 2e1.  So 1 / 1e2 is conservative.
N = 5e6  # total population
T = np.linspace(0, days, days + 1)  # time array for plotting
num_trials = 50  # number of repeats of the random trial

dp = []
li = []


def boarder_sim(days):
    """simulates the leaking of the boarder"""
    for i in range(days):
        x = random.uniform(0, 1)
        if random.uniform(0, 1) < BTPr * (1 - BVR) * beta * CENZ:
            print("Boarder leak")
            return i, math.ceil(x / BTPr * (1 - BVR) * beta * CENZ)
    else:
        return days, 0


for i in range(num_trials):
    x, y = boarder_sim(days)
    dp.insert(0, x)
    li.insert(0, y)

days_pre = math.floor(mean(dp))
days_remaining = days - days_pre
leaked_infections = mean(li)
S0, I0, R0 = 1, 0, 0
T_pre = np.linspace(0, days_pre - 1, days_pre)

pre_leak = odeint(SIR_model, [S0, I0, R0], T_pre,
                  args=(beta, gamma, delta, vax_r))  # pre leak conditions

S1, I1, R1 = pre_leak[-1:, 0], leaked_infections / N, pre_leak[-1:, 2]  # post-leak conditions
T_post = np.linspace(0, days_remaining, days_remaining + 1)

post_leak = odeint(SIR_model, [S1, I1, R1], T_post,
                   args=(beta, gamma, delta, vax_r))

solution = np.vstack((pre_leak, post_leak))

print("Final Numbers \nSusceptible: ", solution[-1:, 0], "\nInfected: ",
      solution[-1:, 1], "\nRemoved: ", solution[-1:, 2], "\nPeak infections: ",
      max(solution[:, 1]) * N, "\nPeak immunity: ", max(solution[:, 2]) * N,
      "\nDays until breach average: ", days_pre)

plt.figure('SIR BHRP/Vector Host Model', figsize=[8, 5])
plt.plot(T, solution[:, 0], label="Susceptible(t)")
plt.plot(T, solution[:, 1], label="Infected(t)")
plt.plot(T, solution[:, 2], label="Removed(t)")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Proportion")
plt.title('SIR Model')
plt.show()
