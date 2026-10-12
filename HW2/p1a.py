import numpy as np
import matplotlib.pyplot as plt
import scienceplots


plt.style.use("science")
plt.rcParams.update({"xtick.labelsize": 20, "ytick.labelsize": 20, "figure.figsize":(10,6), "axes.labelsize":20})


def f(x:list):
    return x[0]**3 # as it is only a function of x

def x(i,j, h, s):
    return [h*(i+j/2), h*s*j]

def laplacian_f(i,j, alpha, beta, gamma, delta):
    return  alpha*f(x(i,j, h, s)) + \
            beta*f( x(i+1,j, h, s)) + \
            gamma*f(x(i,j-1, h, s)) + \
            delta*f(x(i-1,j+1, h, s))


s = np.sqrt(3)/2
H = []
norm = []

plt.figure()
plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10)*6,marker='o',  label='analytical')
for n in [10, 100, 1000, 10000]:
    h = 1/n
    A = np.array([
            [1, 1, 1, 1],
            [0, h**2/2, h**2/8, h**2/8],
            [0, 0, h**2*s**2/2, h**2*s**2/2],
            [0, 0, h**2*s/2, -h**2*s/2]
        ])
    b = np.array([0, 1, 1, 0])

    alpha, beta, gamma, delta = np.linalg.solve(A,b)

    mesh = np.linspace(0, 1, n)
    analytical = 6*mesh
    index = np.arange(0, n, 1)
    fdm = laplacian_f(i=index, j=0, alpha=alpha, beta=beta, gamma=gamma, delta=delta)

    plt.plot(mesh, fdm, label=f"h={h:.1e}")

    H.append(h)
    norm.append(np.linalg.norm(analytical-fdm))

plt.ylabel(r"$\nabla^2 f $", fontsize=20)
plt.xlabel("$x$", fontsize=20)
plt.legend(fontsize=20)
plt.savefig("problem_1_a.pdf", dpi=1000)
plt.show()

plt.figure()
plt.plot(H, norm)
plt.ylabel(r"$L_2$ norm", fontsize=25)
plt.xlabel("$h$", fontsize=25)
plt.savefig("problem_1_linear_convergence.pdf", dpi=1000)
plt.show()
