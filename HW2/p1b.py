from pprint import pprint
import sympy as sm


h, s = sm.symbols('h s')

A = sm.Matrix([
    [1, 1, 1, 1, 1, 1, 1],
    [0, h, -h/2, -h/2, -h, h/2, h/2],
    [0, 0, -h*s, h*s, 0, h*s, -h*s],
    [0, h**2/2, h**2/8, h**2/8, h**2/2, h**2/8, h**2/8],
    [0, 0, s*h**2/2, -s*h**2/2, 0, s*h**2/2, -s*h**2/2],
    [0, 0, s**2*h**2/2, s**2*h**2/2, 0, s**2*h**2/2, s**2*h**2/2],
    [0, 1, 0, 0, -1, 0, 0]
])

b = sm.Matrix([0, 0, 0, 1, 0, 1, 0])

weights = A.inv() * b
pprint(sm.latex(weights))
