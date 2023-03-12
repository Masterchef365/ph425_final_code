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

# Create the plot
plots = [
    (sz_neg, sz_pos, s_z, 'z'),
    (sy_neg, sy_pos, s_y, 'y'),
    (sx_neg, sx_pos, s_x, 'x'),
]

time_varying_b_field = b_field(t_vals)

for (neg, pos, op, name) in plots:
    tvar_prob = prob_for_all_t(t_vals, pos, psi)
    tvar_prob_neg = prob_for_all_t(t_vals, neg, psi)
    tvar_expect = expect_for_all_t(t_vals, op, psi)
    tvar_uncertain = uncertainty_for_all_t(t_vals, op, psi)

    fig, ax = plt.subplots()
    ax.plot(t_vals, tvar_prob, color='blue', label='P(+hbar/2)')
    ax.plot(t_vals, tvar_prob_neg, color='purple', label='P(-hbar/2)')
    ax.plot(t_vals, tvar_uncertain, color='green', label='Uncertainty')
    ax.plot(t_vals, tvar_expect, color='red', label='Expectation')
    ax.plot(t_vals, time_varying_b_field, color='black', label='B-field')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Magnitude')
    ax.set_title(f'Measurement expectations ({name})')
    ax.legend()



plt.show()
