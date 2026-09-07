import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scienceplots

# A note for the grader:
# I use this matplotlib wrapper for getting LaTeX style plots
# https://github.com/garrettj403/SciencePlots should be easy to install with pip

plt.style.use("science")
plt.rcParams.update({"xtick.labelsize": 20, "ytick.labelsize": 15})

# helper elfs
def f(x):
    return np.exp(-x) * np.tan(x)

def regression(H, C, p):
    return C*H**p

def laplacian_f(x):
    sec2_x = 1 / np.cos(x)**2
    return np.exp(-x) * (np.tan(x) - 2 * sec2_x + 2 * sec2_x * np.tan(x))

def step_size(k):
    return 10**(-k/100)

# def fdm_approximation_3_node(function, step_size):
#     x1 = 0
#     x3 = step_size
#     x2 = np.random.uniform(low=x1, high=x3)  # single random interior point
#
#     w1 = 2/((x1-x2)*(x1-x3))
#     w2 = 2/((x2-x1)*(x2-x3))
#     w3 = 2/((x3-x1)*(x3-x2))
#
#     laplacian_fdm_approx = w1*function(x1) + w2*function(x2) + w3*function(x3)
#     return laplacian_fdm_approx, x2

def fdm_approximation_4_node(function, step_size):
    """
    :param function: function of which to approximate f''(x)
    :param step_size: finite difference step size
    :return: approximate value of f''(x) on [0, step_size], and the point at which it was evaluated
    """
    x1 = 0
    x4 = step_size
    # x2 and x3 are uniformly distributed within [x1=0, x4=step_size]
    x2, x3 = np.sort(np.random.uniform(low=x1, high=x4, size=2))

    w1 = 2/((x1-x2)*(x1-x3)) + 2/((x1-x2)*(x1-x4)) + 2/((x1-x3)*(x1-x4))
    w2 = 2/((x2-x1)*(x2-x3)) + 2/((x2-x1)*(x2-x4)) + 2/((x2-x3)*(x2-x4))
    w3 = 2/((x3-x1)*(x3-x2)) + 2/((x3-x1)*(x3-x4)) + 2/((x3-x2)*(x3-x4))
    w4 = 2/((x4-x1)*(x4-x2)) + 2/((x4-x1)*(x4-x3)) + 2/((x4-x2)*(x4-x3))

    laplacian_fdm_approx = 0
    for x, w in zip([x1, x2, x3, x4], [w1, w2, w3, w4]):
        laplacian_fdm_approx += w * function(x)
    return laplacian_fdm_approx, x2


def main():
    H = step_size(np.linspace(101, 300, 1000))
    approximations = []
    evaluated_points = []
    for h in H:
        approx, x2 = fdm_approximation_4_node(function=f, step_size=h)
        approximations.append(approx)
        evaluated_points.append(x2)

    approximations = np.array(approximations)
    evaluated_points = np.array(evaluated_points)
    exact_values = laplacian_f(evaluated_points)
    errors = np.abs(approximations - exact_values)

    # Now we need to do the curve fit
    prop, _ = curve_fit(regression, H, errors)
    print (f"C = {prop[0]:4f}, p = {prop[1]:4f}")

    # now let's plot stuffs
    plt.figure(figsize=(12, 6))
    plt.loglog(H, errors, 'x')
    plt.loglog(H, prop[0]*H**prop[1])
    plt.xlabel("Step size h", fontsize=20)
    plt.ylabel("Absolute error",  fontsize=20)
    plt.title("4 node FDM Laplacian approximation error vs step size", fontsize=20)
    plt.show()


if __name__ == '__main__':
    main()