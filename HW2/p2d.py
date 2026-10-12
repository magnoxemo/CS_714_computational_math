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
hs = np.linspace(1e-3, 1e-1, 10)
abs_error = []
for h in hs:
    t = np.arange(0, T + 0.0001, h)[-1]
    abs_error.append(np.abs(y(t+h) - (y(t) + h*y_prime(t))))

b, log_a = np.polyfit(np.log(hs), np.log(abs_error), 1)
a = np.exp(log_a)

plt.loglog(hs, abs_error, marker='o', color='red', label="simulation")
plt.loglog(hs, a*hs**b, marker='*', label = r"\texttt{exp fit}  = "+rf"$ {a:.2f}h^{{{b:.2f}}}$")
plt.xlabel(r"$h$")
plt.ylabel(r"$|y(t_{30}) - y_{\texttt{numerical}}(t_{30})|$")
plt.legend(fontsize=20)
plt.title("2D", fontsize=30)
plt.savefig("2D.pdf")
