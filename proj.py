import numpy as np
from math import sqrt, cos, sin, radians, atan2, acos, degrees, e
import matplotlib.pyplot as plt

def braket(u, v):
    return np.dot(u.conj(), v)


def prob(u, v):
    return abs(braket(u, v))**2


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v/norm


def expected_val(op, psi):
    return braket(psi, np.dot(op, psi))


def uncertainty(op, psi):
    mu = expected_val(op, psi)
    op2 = np.dot(op, op)
    v = expected_val(op2, psi)
    k = v - mu**2
    assert(k.imag == 0.0)
    return sqrt(k.real)



# Define particle properties
w = 1.     # B-field oscillation rate
q = 1.     # Particle charge
m = 1.     # Particle mass
hbar = 1.  # Planck's constant

# Set up basis eigenstates
sz_pos = np.array([1, 0])
sz_neg = np.array([0, 1])

sx_pos = np.array([1, 1])/sqrt(2)
sx_neg = np.array([1, -1])/sqrt(2)

sy_pos = np.array([1, 1j])/sqrt(2)
sy_neg = np.array([1, -1j])/sqrt(2)

# Set up spin matrices
s_z = hbar/2. * np.array([[1., 0.],
                          [0., -1.]])

s_x = hbar/2. * np.array([[0., 1.],
                          [1., 0.]])

s_y = hbar/2. * np.array([[0., -1j],
                          [1j, 0.]])



# Define eigenbasis
E_1 = normalize(np.array([1, sqrt(2) - 1.]))
E_2 = normalize(np.array([1, -sqrt(2) - 1.]))

# Define initial state constants
c_1 = sqrt(4 + 2.*sqrt(2))/sqrt(8.)
c_2 = sqrt(4 - 2.*sqrt(2))/sqrt(8.)

# B-field
def b_field(t):
    # B_0 = 1
    return np.cos(w * t)


# Define state with respect to time
def psi(t):
    f = 1j*q/(m*hbar*w) * sin(w * t)
    return c_1 * e**(-f) * E_1 + c_2 * e**(f) * E_2

# Make sure the initial state is the initial state
assert(prob(sz_pos, psi(0)) == 1.0)

# Generate values for the horizontal axis (time)
t_vals = np.linspace(0, 4*np.pi, 1000)

def prob_for_all_t(t_vals, u, psi):
    return np.array([prob(psi(t), u) for t in t_vals])


def expect_for_all_t(t_vals, op, psi):
    return np.array([expected_val(op, psi(t)) for t in t_vals])


def uncertainty_for_all_t(t_vals, op, psi):
    return np.array([uncertainty(op, psi(t)) for t in t_vals])


time_varying_sz_prob = prob_for_all_t(t_vals, sz_pos, psi)
time_varying_sy_prob = prob_for_all_t(t_vals, sy_pos, psi)
time_varying_sx_prob = prob_for_all_t(t_vals, sx_pos, psi)

time_varying_sz_expect = expect_for_all_t(t_vals, s_z, psi)
time_varying_sy_expect = expect_for_all_t(t_vals, s_y, psi)
time_varying_sx_expect = expect_for_all_t(t_vals, s_x, psi)

time_varying_sz_uncertain = uncertainty_for_all_t(t_vals, s_z, psi)
time_varying_sy_uncertain = uncertainty_for_all_t(t_vals, s_y, psi)
time_varying_sx_uncertain = uncertainty_for_all_t(t_vals, s_x, psi)


time_varying_b_field = b_field(t_vals)

# time_varying_sz_prob_neg = prob_for_all_t(t_vals, sz_neg)
# time_varying_sy_prob_neg = prob_for_all_t(t_vals, sy_neg)
# time_varying_sx_prob_neg = prob_for_all_t(t_vals, sx_neg)
# 
# print(time_varying_sz_prob + time_varying_sz_prob_neg)
# print(time_varying_sy_prob + time_varying_sy_prob_neg)
# print(time_varying_sx_prob + time_varying_sx_prob_neg)



# # Create the plot
# fig, ax = plt.subplots()
# ax.plot(t_vals, time_varying_sz_prob, color='blue', label='P(S_z = +h/2)')
# ax.plot(t_vals, time_varying_sy_prob, color='green', label='P(S_y = +h/2)')
# ax.plot(t_vals, time_varying_sx_prob, color='red', label='P(S_x = +h/2)')
# ax.plot(t_vals, time_varying_b_field, color='black', label='B-field')
# ax.set_xlabel('Time (seconds)')
# ax.set_ylabel('Probability')
# ax.set_title('Measurement probabilities')
# ax.legend()

# # Create the plot
# fig, ax = plt.subplots()
# ax.plot(t_vals, time_varying_sz_expect, color='blue', label='S_z')
# ax.plot(t_vals, time_varying_sy_expect, color='green', label='S_y')
# ax.plot(t_vals, time_varying_sx_expect, color='red', label='S_x')
# ax.plot(t_vals, time_varying_b_field, color='black', label='B-field')
# ax.set_xlabel('Time (seconds)')
# ax.set_ylabel('Probability')
# ax.set_title('Measurement expectations')
# ax.legend()

# Create the plot
fig, ax = plt.subplots()
ax.plot(t_vals, time_varying_sz_uncertain, color='blue', label='S_z')
ax.plot(t_vals, time_varying_sy_uncertain, color='green', label='S_y')
ax.plot(t_vals, time_varying_sx_uncertain, color='red', label='S_x')
ax.plot(t_vals, time_varying_b_field, color='black', label='B-field')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Probability')
ax.set_title('Measurement expectations')
ax.legend()



plt.show()
