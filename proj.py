import numpy as np
from math import sqrt, cos, sin, radians, atan2, acos, degrees, e
import matplotlib.pyplot as plt

def braket(u, v):
    return np.dot(u.conj(), v)


def prob(u, v):
    return abs(braket(u, v))**2


def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm


# Set up basis eigenstates
sz_pos = np.array([1, 0])
sz_neg = np.array([0, 1])

sx_pos = np.array([1, 1])/sqrt(2)
sx_neg = np.array([1, -1])/sqrt(2)

sy_pos = np.array([1, 1j])/sqrt(2)
sy_neg = np.array([1, -1j])/sqrt(2)

# Define eigenbasis
E_1 = normalize(np.array([1, sqrt(2) - 1.]))
E_2 = normalize(np.array([1, -sqrt(2) - 1.]))

# Define initial state constants
c_1 = sqrt(4 + 2.*sqrt(2))/sqrt(8.)
c_2 = sqrt(4 - 2.*sqrt(2))/sqrt(8.)

# Define particle properties
w = 1.
q = 1.
m = 1.
h = 1.

# Define state with respect to time
def psi(t):
    f = 1j*q/(m*h*w) * sin(w * t)
    return c_1 * e**(-f) * E_1 + c_2 * e**(f) * E_2

# Make sure the initial state is the initial state
assert(prob(sz_pos, psi(0)) == 1.0)

# Generate values for the horizontal axis (time)
t_vals = np.linspace(0, 4*np.pi, 1000)

def prob_for_all_t(t_vals, u):
    return np.array([prob(psi(t), u) for t in t_vals])


time_varaying_sz_prob = prob_for_all_t(t_vals, sz_pos)
time_varaying_sy_prob = prob_for_all_t(t_vals, sy_pos)
time_varaying_sx_prob = prob_for_all_t(t_vals, sx_pos)

# time_varaying_sz_prob_neg = prob_for_all_t(t_vals, sz_neg)
# time_varaying_sy_prob_neg = prob_for_all_t(t_vals, sy_neg)
# time_varaying_sx_prob_neg = prob_for_all_t(t_vals, sx_neg)
# 
# print(time_varaying_sz_prob + time_varaying_sz_prob_neg)
# print(time_varaying_sy_prob + time_varaying_sy_prob_neg)
# print(time_varaying_sx_prob + time_varaying_sx_prob_neg)



# Create the plot
fig, ax = plt.subplots()
ax.plot(t_vals, time_varaying_sz_prob, color='blue', label='P(S_z = +h/2)')
ax.plot(t_vals, time_varaying_sy_prob, color='green', label='P(S_y = +h/2)')
ax.plot(t_vals, time_varaying_sx_prob, color='red', label='P(S_x = +h/2)')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Probability')
ax.set_title('Measurement probabilities')
ax.legend()

plt.show()
