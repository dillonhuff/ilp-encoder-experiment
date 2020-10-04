import os
from sympy import *

glpk_path = '/usr/local/Cellar/glpk/4.65/bin/glpsol'
glpk_flags = '--model'

def sp(c):
    return sympify(c)

def run_cmd(cmd):
    res = os.system(cmd)
    return res

# run_cmd('{0} {1} {2}'.format(glpk_path, glpk_flags, 'assignment.mod'))

def print_var(name):
    return 'printf "{0}: %d\\n", {0};'.format(name)

def add_int_var(name, model):
    return model + 'var {0}, integer;\n'.format(name)

def add_constraint(name, cst, model):
    return model + 's.t. {0} : {1};\n'.format(name, cst)

# sp = sympify("ii_p*p + d_p");
# sc = sympify("ii_c*c + d_c");

# polytope = []
# polytope.append(sympify('ii_p > 0'))
# polytope.append(sympify('ii_c > 0'))

# polytope.append(sympify('d_p >= 0'))
# polytope.append(sympify('d_c >= 0'))

# farkas_in = sc - sp >= 0

# resource_p = sympify('R_a == R_b >> S_p(p) != S_c(c)')
# print(resource_p)

# print('Farkas in:', farkas_in)
# print(polytope)

h = sympify("3*ii_x + ii_y - 12 - 1 >= 0")
# print(h)
# h = simplify(h)
# print('Simplified:', h)
# print('collected ;', collect(h, sympify('x')))

h0 = sympify("ii_x >= 1")
h1 = sympify("ii_y >= 1")

objective = 'ii_x + ii_y'

problem = ''
problem = add_int_var("ii_x", problem)
problem = add_int_var("ii_y", problem)

problem = add_constraint('c1', h, problem)
problem = add_constraint('c2', h0, problem)
problem = add_constraint('c3', h1, problem)

problem += 'solve;\n'

problem += print_var('ii_x')
problem += print_var('ii_y')

print(problem)
problem_file = open('prob.mod', 'w').write(problem)

run_cmd('{0} {1} {2}'.format(glpk_path, glpk_flags, 'prob.mod'))


