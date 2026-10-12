import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import time


plt.style.use("science")
plt.rcParams.update({"xtick.labelsize": 20, "ytick.labelsize": 20, "figure.figsize":(10,6), "axes.labelsize":20})


s = np.sqrt(3) / 2


def u_exact(x, y):
    return ((2*y - np.sqrt(3))**2 - 3*(2*x - 1)**2) * np.sin(y)


def f(x, y):
    return -24*np.sin(y) + 3 *(2*x-1)**2*np.sin(y) + 8*np.sin(y) + \
            4*(2*y-np.sqrt(3)) *np.cos(y) \
            -(2*y-np.sqrt(3))**2*np.sin(y) + \
             4*(2*y-np.sqrt(3)) *np.cos(y)


def node_xy(i, j, h):
    return h*(i + j/2), h*s*j


def stencil_weights(h):
    A = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [0, h, -h/2, -h/2, -h, h/2, h/2],
        [0, 0, -h*s, h*s, 0, h*s, -h*s],
        [0, h**2/2, h**2/8, h**2/8, h**2/2, h**2/8, h**2/8],
        [0, 0, s*h**2/2, -s*h**2/2, 0, s*h**2/2, -s*h**2/2],
        [0, 0, s**2*h**2/2, s**2*h**2/2, 0, s**2*h**2/2, s**2*h**2/2],
        [0, 1, 0, 0, -1, 0, 0]
    ])
    b = np.array([0, 0, 0, 1, 0, 1, 0])
    return np.linalg.solve(A, b)


offsets = [(0, 0), (1, 0), (0,-1), (-1,1), (-1,0), (0,1), (1, -1)]


def prepare_nodes(n):
    idx = {}
    nodes = []
    for j in range(1, n - 1):
        for i in range(1, n - j):
            idx[(i, j)] = len(nodes)
            nodes.append((i, j))
    return idx, nodes


def solve(n, show_spy=False):
    h = 1 / n
    weights = stencil_weights(h)
    rows, cols, vals = [], [], []
    node_ids, nodes = prepare_nodes(n)
    n_dofs = len(node_ids)
    b = np.zeros(n_dofs)

    for k, (i, j) in enumerate(nodes):
        x, y = node_xy(i, j, h)
        b[k] = f(x, y)

        for (di, dj), w in zip(offsets, weights):
            ii, jj = i + di, j + dj
            if (ii, jj) in node_ids:
                rows.append(k)
                cols.append(node_ids[(ii, jj)])
                vals.append(w)

    M = coo_matrix((vals, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
    if show_spy:
        plt.figure()
        plt.spy(M, markersize=3)
        plt.title(f"Sparsity pattern, n = {n}")

    U = spsolve(M, b)

    xy = np.array([node_xy(i, j, h) for (i, j) in nodes])
    err = U - u_exact(xy[:, 0], xy[:, 1])
    l2 = np.sqrt(s / (2 * n**2) * np.sum(err**2))
    return h, l2, xy, U


def power_func(x, a, b):
    return a * (x ** b)


ns = np.array([10, 20, 40, 80, 160, 320])
wall_time = np.zeros(len(ns))

for index, n in enumerate(ns):
    start_time = time.perf_counter()
    solve(n)
    wall_time[index] = time.perf_counter() - start_time

popt, pcov = curve_fit(power_func, ns, wall_time)
a, b = popt
print(f"a = {a} b = {b}")

plt.semilogy(ns, wall_time,  marker='*', label="Numerical", color='red')
plt.semilogy(ns, a*ns**b,  marker= 'o', linestyle='--', label =rf"Fitting: ${a:.3e}\times n^{{{b:.3f}}}$")
plt.legend(fontsize=20)
plt.title("Problem 1e", fontsize=20)
plt.xlabel("n", fontsize=20)
plt.ylabel("Wall Time", fontsize=25)
plt.savefig("1e.pdf")
