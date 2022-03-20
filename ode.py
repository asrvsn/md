'''
Solve a uniform-mass system of harmonic oscillators coupled to a central particle.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.linalg as LA
from tqdm import tqdm
import pdb

# Mass of all particles
m = 1

# No. oscillators
N = int(1e4)

# Times to evaluate solution
Tspan = np.linspace(1, 1e6, 1000)

# Frequencies
fs = np.sqrt(2) * np.tan(
	np.random.uniform(low=0.0, high=1.0, size=N) * np.pi / 2
)

# Spring constants
ks = m * fs ** 2

# Weighted adjacency matrix
A_rows = np.arange(1, N + 1) 
A_cols = np.full(N, 1)
A = sp.coo_matrix((ks, (A_rows, A_cols)), shape=(N+1, N+1)).tocsr()
A += A.T

# Weighted degree matrix
D = sp.diags(
	np.insert(ks, 0, ks.sum())
).tocsr()

# Weighted Laplacian matrix
L = D - A

# Diagonalization
evs, U = LA.eigh(L.toarray().astype(np.float64)) # U is unitary

# Solve position
def x(t, x_0=None, dx_0=None):

	# Initial conditions
	if x_0 is None:
		x_0 = np.insert(np.random.uniform(low=-1.0, high=1.0, size=N), 0, 0)
	if dx_0 is None:
		dx_0 = np.insert(np.random.uniform(low=-1.0, high=1.0, size=N), 0, 0)
	z_0 = U.T @ x_0
	dz_0 = U.T @ dx_0

	z_t = z_0 * np.cos(np.sqrt(m * evs) * t) + dz_0 * np.sin(np.sqrt(m * evs) * t)
	return U @ z_t

# Velocity given position
def v(x):
	return -1/m * L @ x

# Estimate velocity autocorrelation
def chi(tau, T, M=1000):
	est = 0
	for _ in range(M):
		est += v(x(T))[0] * v(x(T - tau))[0]
	est /= M
	return est

# Estimate diffusion constant
def diff(T, M=1000):
	est = 0
	for _ in range(M):
		x_t = x(T)
		est += x_t[0] * v(x_t)[0]
	est /= M
	return est

# Plot results
diff_ests = []
for t in tqdm(Tspan):
	diff_ests.append(diff(t))

plt.plot(Tspan, diff_ests)
plt.show()

