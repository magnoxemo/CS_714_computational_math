import numpy as np
import matplotlib.pyplot as plt
import scienceplots
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


# okay now that my coo_matrix works I need to implement GEMM function
# most the codes the CSR matrix and cg solver is taken from my CS 557
# i took last spring. As that class was is in c++ I just had to translate
# it. As per CSR multiplication I asked Yue and she said we would be fine using
# scipy csr or coo matrix multiplication capability but
# I went ahead and implement one in this class anyway

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


# comparing manual CG against spsolve
n = 40
h, l2, U_cg, residuals, rows, cols, values, b = solve_cg(n)
n_dofs = len(b)

M = coo_matrix((values, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
U_direct = spsolve(M, b)

print("CG iterations:", len(residuals) - 1)
print("max |U_cg - U_direct|:", np.max(np.abs(U_cg - U_direct)))
print("l_2 norm of cg", l2)

plt.figure()
plt.semilogy(residuals)
plt.xlabel("iteration", fontsize=20)
plt.ylabel(r"Residual = $\frac{\|r_k\|_2}{\|b\|_2}$", fontsize=20)
plt.title(f"CG convergence, n = {n}", fontsize=20)
plt.savefig("1f_residual.pdf")
plt.show()
