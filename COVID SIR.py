import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

s_dot = -r*S*I #transmision rate
I_dot = r*S*I-a*I
r_dot = a*I

