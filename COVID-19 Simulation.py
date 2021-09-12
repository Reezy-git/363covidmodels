import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Initialisation Variables
num_tests = 4
R0 = np.linspace(1.1, 2.6, num_tests)  # test range of R0
print(R0)
hosp_capacity = 1000  # Number of hospital beds
inf_days = 10  # Days infectious
IFR = 0.01  # Infection fatality rate
hosp_rate = 0.2  # Hospitalisation rate
len_simul = 400
T = np.linspace(0, len_simul, len_simul + 1)  # Timespan 0-1000 in 1001 increments


def SIR_model(y, t, b, g):
    S, I, R = y
    dS = -b * S * I
    dI = b * S * I - g * I
    dR = g * I
    return [dS, dI, dR]

peak = np.zeros((num_tests, 2))
new_cases = np.zeros((num_tests, len_simul))
t_infected = np.zeros((num_tests, len_simul +1))
f_infected = np.zeros((num_tests))
in_hosp = np.zeros((num_tests, len_simul+1))
for i in range(len(R0)):
    S0 = 5*10**6  # initial susceptible population
    beta = R0[i]/inf_days  # key parameter beta
    y0 = [1, 100/S0, 0] # initial condition 100 infectious, no removed
    gamma = 1/10
    # y = [S, I, R]
    y = odeint(SIR_model, y0, T, args=(beta, gamma))
    inf_peak = max(y[:, 1])
    peak_day = np.where(y[:,1] == inf_peak)[0][0]
    peak[i] = inf_peak, peak_day
    new_cases[i] = (y[0:-1,0] - y[1:,0])*S0
    t_infected[i] = y[0:,1]*S0
    f_infected[i] = y[-1:,2]
    in_hosp[i] = y[:,1] * S0 * hosp_rate
over_hosp_cap = np.sum(in_hosp> hosp_capacity, axis=1)
print(peak)

plt.figure('New Cases', figsize=[8, 4.5])
plt.plot(T[1:], new_cases[0], label="R0 = 1.1")
plt.plot(T[1:], new_cases[1], label="R0 = 1.6")
plt.plot(T[1:], new_cases[2], label="R0 = 2.1")
plt.plot(T[1:], new_cases[3], label="R0 = 2.6")
plt.legend()
plt.xlabel("Time (days)")
plt.ylabel("Daily New Cases")
plt.title('New Cases')
plt.show()

plt.figure('Total Infected', figsize=[8, 4.5])
plt.plot(T[:], t_infected[0], label="R0 = 1.1")
plt.plot(T[:], t_infected[1], label="R0 = 1.6")
plt.plot(T[:], t_infected[2], label="R0 = 2.1")
plt.plot(T[:], t_infected[3], label="R0 = 2.6")
plt.legend()
plt.xlabel("Time (days)")
plt.ylabel("Infected")
plt.title('Total Infected')
plt.show()

plt.figure('Final_Infected', figsize=[8, 3])
plt.plot(R0, f_infected)
plt.legend()
plt.xlabel("R0")
plt.ylabel("Proportion Infected")
plt.title('Final Infected')
plt.show()



plt.figure('Over Hospital Cap', figsize=[8,3])
plt.plot(R0, over_hosp_cap)
plt.xlabel('R0')
plt.ylabel('#Days Health Providers Overwhelmed')
plt.title('Over Capacity Days')
plt.show()

plt.figure('Peak Day', figsize=[8,3])
plt.plot(R0, peak[:,1])
plt.xlabel("R0")
plt.ylabel("Peak Day")
plt.title("Peak Day")
plt.show()
