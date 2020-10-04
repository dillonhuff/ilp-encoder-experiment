import os
from sympy import *

glpk_path = '/usr/local/Cellar/glpk/4.65/bin/glpsol'
glpk_flags = '--model'

def sp(c):
    return sympify(c)

def run_cmd(cmd):
    res = os.system(cmd)
    return res

def print_var(name):
    return 'printf "{0}: %d\\n", {0};\n'.format(name)

def add_int_var(name, model):
    return model + 'var {0}, integer;\n'.format(name)

def add_constraint(name, cst, model):
    return model + 's.t. {0} : {1};\n'.format(name, cst)

class ILPBuilder:

    def __init__(self):
        self.constraints = []
        self.variables = []
        self.variable_bounds = {}
        self.objective = None

    def add_synonym(self, name, expr):
        self.add_int_var(name)
        self.add_constraint(name + ' = ' + expr)
        
    def add_int_var(self, name, lower=None, upper=None):
        self.variables.append(name)
        self.variable_bounds[name] = (lower, upper)
        if (upper != None):
            self.add_constraint(name + ' <= ' + str(upper))
        if (lower != None):
            self.add_constraint(name + ' >= ' + str(lower))

    def add_constraint(self, cst):
        self.constraints.append(cst)

    def set_objective(self, obj):
        self.objective = obj

    def solve(self):
        problem = ''
        
        i = 0
        for v in self.variables:
            problem = add_int_var(v, problem)
            i += 1

        i = 0
        for c in self.constraints:
            problem = add_constraint('c{0}'.format(i), c, problem)
            i += 1

        problem += 'minimize obj : {0};\n'.format(self.objective)
        problem += 'solve;\n'

        i = 0
        for v in self.variables:
            problem += print_var(v)
            i += 1
        problem_file = open('prob.mod', 'w').write(problem)
        run_cmd('{0} {1} {2}'.format(glpk_path, glpk_flags, 'prob.mod'))


builder = ILPBuilder()

# Resource assignment constraints
builder.add_int_var("unit_p", 0, 1)
builder.add_int_var("unit_c", 0, 1)

builder.add_synonym("neg_p_c_share", "unit_p - unit_c")

builder.add_int_var("ii_p", 1, 100000)
builder.add_int_var("ii_c", 1, 100000)

builder.add_int_var("d_p", 0, 100000)
builder.add_int_var("d_c", 0, 100000)

h = sympify("3*ii_p + ii_c - 12 - 1 >= 0")
h0 = sympify("ii_p >= 1")
h1 = sympify("ii_c >= 1")

builder.set_objective('ii_p + ii_c')

builder.add_constraint(h)
builder.add_constraint(h0)
builder.add_constraint(h1)

builder.solve()

