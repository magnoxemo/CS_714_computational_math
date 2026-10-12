import numpy as np
import matplotlib.pyplot as plt
import scienceplots


plt.style.use("science")
plt.rcParams.update({"xtick.labelsize": 20, "ytick.labelsize": 20, "figure.figsize":(10,6), "axes.labelsize":20})


def y(t):
    return np.exp(-t) * (np.cos(5*t) + 1/5 * np.sin(5*t))

def y_prime(t):
    return  - np.exp(-t) * (np.cos(5*t) + 1/5 * np.sin(5*t)) + np.exp(-t) * (-5*np.sin(5*t) +  np.cos(5*t))


T = 3

h = 0.02
time_steps = np.arange(0, T + 0.0001, h)
exact_solution = y(time_steps)
numerical_solution = y(time_steps-h) + h/3 *(y_prime(time_steps-h) + 4*y_prime(time_steps) + y_prime(time_steps+h))

plt.figure()
plt.plot(time_steps, exact_solution, label = 'exact')
plt.plot(time_steps, numerical_solution, label = 'numerical')
plt.xlabel(r"$T$")
plt.ylabel(r"$y(t)$")
plt.legend(fontsize= 25)
plt.title("2c", fontsize=30)
plt.savefig("2c.pdf")

hs = np.linspace(1e-3, 1e-1, 10)
abs_error = []
for h in hs:
    time_steps = np.arange(0, T + 0.0001, h)
    exact_solution = y(time_steps+h)
    numerical_solution = y(time_steps-h) + h/3 *(y_prime(time_steps-h) + 4*y_prime(time_steps) + y_prime(time_steps+h))
    abs_error.append(np.linalg.norm(exact_solution - numerical_solution))

b, log_a = np.polyfit(np.log(hs), np.log(abs_error), 1)
a = np.exp(log_a)

plt.figure()
plt.loglog(hs, abs_error, marker='o', color='red', label="simulation")
plt.loglog(hs, a*hs**b, marker='*', label = r"\texttt{exp fit}  = "+rf"$ {a:.2f}h^{{{b:.2f}}}$")
plt.xlabel(r"$h$")
plt.ylabel(r"$|y(t) - y_{\texttt{numerical}}(t)|$")
plt.legend(fontsize=20)
plt.title("2C", fontsize=30)
plt.savefig("2c_part_2.pdf")
