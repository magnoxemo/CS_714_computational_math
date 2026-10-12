import time
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


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


def build_coo_matrix(n, stencil_offsets, show_spy=False):
    h = 1 / n
    weights = stencil_weights(h)
    rows, cols, vals = [], [], []
    node_ids, nodes = prepare_nodes(n)
    n_dofs = len(node_ids)
    b = np.zeros(n_dofs)

    for k, (i, j) in enumerate(nodes):
        x, y = node_xy(i, j, h)
        b[k] = f(x, y)

        for (di, dj), w in zip(stencil_offsets, weights):
            # converting local stencil index to global stencil index
            ii, jj = i + di, j + dj

            # checking if the stencil is inside of the domain or at the boundary
            if (ii, jj) in node_ids:
                rows.append(k)
                cols.append(node_ids[(ii, jj)])
                vals.append(w)

    if show_spy:
        M = coo_matrix((vals, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
        plt.figure()
        plt.spy(M, markersize=3)
        plt.title(f"Sparsity pattern, n = {n}")

    return rows, cols, vals, b


class CSRMatrix:
    def __init__(self, rows, cols, values, n_rows):
        self.n_rows = n_rows
        self.calculate_row_off_set(rows)
        self.cols = np.asarray(cols)
        self.values = np.asarray(values, dtype=float)

    def calculate_row_off_set(self, rows):
        counts = np.zeros(self.n_rows, dtype=int)
        for row in rows:
            counts[row] += 1

        self.row_offsets = np.zeros(self.n_rows + 1, dtype=int)
        self.row_offsets[1:] = np.cumsum(counts)

    def multiply(self, x):
        products = self.values * x[self.cols]
        return np.add.reduceat(products, self.row_offsets[:-1])


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
    n = len(b)
    if max_iter is None:
        max_iter = 10 * n
    x = np.zeros(n) if x0 is None else x0.copy()

    r = b - A.multiply(x)
    p = r.copy()
    rr = r @ r
    b_norm = np.sqrt(b @ b)
    residuals = [np.sqrt(rr) / b_norm]

    for _ in range(max_iter):
        Ap = A.multiply(p)
        alpha = rr / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = r @ r
        residuals.append(np.sqrt(rr_new) / b_norm)
        if residuals[-1] < tol:
            break
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new

    return x, residuals


def solve_cg(n, tol=1e-10):
    h = 1 / n
    rows, cols, values, b = build_coo_matrix(n, stencil_offsets=offsets)
    n_dofs = len(b)
    A = CSRMatrix(rows=rows, cols=cols, values=-np.asarray(values), n_rows=n_dofs)

    U, residuals = conjugate_gradient(A, -b, tol=tol)

    _, nodes = prepare_nodes(n)
    xy = np.array([node_xy(i, j, h) for (i, j) in nodes])
    err = U - u_exact(xy[:, 0], xy[:, 1])
    l2 = np.sqrt(s / (2 * n**2) * np.sum(err**2))
    return h, l2, U, residuals, rows, cols, values, b


def power_func(x, a, b):
    return a * (x ** b)


ns_e = np.array([10, 20, 40, 80, 160, 320])
wall_time = np.zeros(len(ns_e))

for index, n in enumerate(ns_e):
    start_time = time.perf_counter()
    solve(n)
    wall_time[index] = time.perf_counter() - start_time

popt_e, pcov = curve_fit(power_func, ns_e, wall_time)

ns = np.arange(10, 161, 10, dtype=int)
wall_time_cg = np.zeros(len(ns))

for index, n in enumerate(ns):
    start_time = time.perf_counter()
    solve_cg(n)
    wall_time_cg[index] = time.perf_counter() - start_time

ns = ns.astype(float)
b, log_a = np.polyfit(np.log(ns), np.log(wall_time_cg), 1)
a = np.exp(log_a)

print(f"a = {a:.6e}")
print(f"b = {b:.6f}")
print(f"T(n) = {a:.6e} * n^{b:.6f}")

plt.figure(figsize=(8,5))
plt.semilogy(ns, wall_time_cg, marker="o", label="1g: CG")
plt.semilogy(ns, a * (ns ** b), marker="s", label=rf"1g: ${a:.2e}\times n^{{{b:.3f}}}$")
plt.semilogy(ns, popt_e[0] * (ns ** popt_e[1]), marker="*", label=rf"1e: ${popt_e[0]:.2e}\times n^{{{popt_e[1]:.3f}}}$")
plt.xlabel("n", fontsize=20)
plt.ylabel(r"$\texttt{Wall Time}$", fontsize=20)
plt.legend(fontsize=20)
plt.savefig("1h_wall_time.pdf")
plt.show()
