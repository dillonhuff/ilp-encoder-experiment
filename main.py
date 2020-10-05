import os
from sympy import *

glpk_path = '/usr/local/Cellar/glpk/4.65/bin/glpsol'
glpk_flags = '--model'

def parens(s):
    return '(' + s + ')'


def sp(c):
    return sympify(c)

def run_cmd(cmd):
    res = os.system(cmd)
    return res

def print_var(name):
    return 'printf "<SOL>,{0},%d\\n", {0};\n'.format(name)

def add_int_var(name, model):
    return model + 'var {0}, integer;\n'.format(name)

def add_constraint(name, cst, model):
    return model + 's.t. {0} : {1};\n'.format(name, cst)

class Monomial:

    def __init__(self, components):
        self.components = components 

class Polynomial:

    def __init__(self, monomials):
        self.monomials = monomials

class Constraint:

    def __init__(self, expr, cmp):
        self.expr = expr
        self.cmp = cmp

    def __repr__(self):
        return self.expr + ' ' + self.cmp + ' 0'

def parse_constraint(cst):

    assert(False)

class ILPBuilder:

    def __init__(self):
        self.unique_num = 0
        self.outer_foralls = []
        self.constraints = []
        self.variables = []
        self.variable_bounds = {}
        self.objective = None

    def unique_name(self, prefix='U_'):
        un = self.unique_num
        self.unique_num += 1
        print('unique num =', self.unique_num)
        return prefix + str(un);

    def add_outer_forall(self, var, lb, ub):
        self.variable_bounds[var] = (lb, ub)

    def lb(self, name):
        return self.variable_bounds[name][0]

    def ub(self, name):
        return self.variable_bounds[name][1]

    def add_synonym(self, name, expr):
        self.add_int_var(name)
        self.add_constraint_eqz(name + ' - ' + parens(expr))
        # self.add_constraint(name + ' = ' + expr)
        
    def add_int_var(self, name, lower=None, upper=None):
        assert(not name in self.variables)
        self.variables.append(name)
        self.variable_bounds[name] = (lower, upper)
        if (upper != None):
            self.add_constraint_lez(name + ' - ' + parens(str(upper)))
            # self.add_constraint(name + ' <= ' + str(upper))
        if (lower != None):
            self.add_constraint_gez(name + ' - ' + parens(str(lower)))
            # self.add_constraint(name + ' >= ' + str(lower))

    def add_constraint_eqz(self, cst):
        self.constraints.append(Constraint(cst, '='))

    def add_constraint_gez(self, cst):
        self.constraints.append(Constraint(cst, '>='))

    def add_constraint_lez(self, cst):
        self.constraints.append(Constraint(cst, '<='))

    def set_objective(self, obj):
        self.objective = obj

    def is_zero_one_var(self, name):
        return self.variable_bounds[name][0] == 0 and self.variable_bounds[name][1] == 1

    def fresh_zero_one_var(self, prefix='zo_'):
        name = self.unique_name('zo_')
        self.add_int_var(name, 0, 1)
        return name

    def negate(self, var):
        assert(self.is_zero_one_var(var))
        neg = self.fresh_zero_one_var('neg_')
        self.add_constraint_eqz('{0} - 1 + {1}'.format(neg, var))
        return neg

    def add_indicator(self, target, ub):
        name = 'I_' + target
        self.add_int_var(name, 0, 1)
        self.add_constraint_lez('{0} - {1}*{2}'.format(target, ub, name))

    def solve(self):
        problem = ''

        i = 0
        for v in self.variables:
            problem = add_int_var(v, problem)
            i += 1

        i = 0
        for c in self.constraints:
            problem = add_constraint('c{0}'.format(i), str(c), problem)
            i += 1

        if self.objective != None:
            problem += 'minimize obj : {0};\n'.format(self.objective)
        problem += 'solve;\n'

        i = 0
        for v in self.variables:
            problem += print_var(v)
            i += 1
        problem_file = open('prob.mod', 'w').write(problem)
        run_cmd('{0} {1} {2} >& sol.txt'.format(glpk_path, glpk_flags, 'prob.mod'))
        sol_lines = open('sol.txt', 'r').readlines()
        val_map = {}
        for l in sol_lines:
            fields = l.split(',')
            if (len(fields) == 3 and fields[0] == '<SOL>'):
                val_map[fields[1]] = int(fields[2])
        return val_map

builder = ILPBuilder()
builder.add_int_var('a', 0, 1)
builder.add_constraint_eqz('a - 1')
na = builder.negate('a')
sol = builder.solve()
for s in sol:
    print('\t', s, '=', sol[s])
assert(sol['a'] == 1)
assert(sol[na] == 0)


builder = ILPBuilder()
builder.add_int_var('a', 0, 1)
builder.add_constraint_eqz('a')
na = builder.negate('a')
sol = builder.solve()
for s in sol:
    print('\t', s, '=', sol[s])
assert(sol['a'] == 0)
assert(sol[na] == 1)

# builder = ILPBuilder()
# # Schedule parameters
# builder.add_int_var("ii_p", 1, 100000)
# builder.add_int_var("ii_c", 1, 100000)

# builder.add_int_var("d_p", 0, 100000)
# builder.add_int_var("d_c", 0, 100000)

# # Resource assignment constraints
# builder.add_int_var("unit_p", 0, 1)
# builder.add_int_var("unit_c", 0, 1)

# # These indicator variables sum to zero iff
# # p and c are using the same functional unit
# builder.add_synonym("neg_p_c_share", "unit_p - unit_c")
# ub = 1
# lb = -1
# builder.add_indicator("neg_p_c_share", ub)

# builder.add_synonym("neg_c_p_share", "-1*neg_p_c_share")
# builder.add_indicator("neg_c_p_share", -1*lb)

# # These indicator variables sum to zero
# # iff p and c are scheduled at the same time
# # builder.add_synonym("neg_p_c_time", "ii_c*c + d_c - ii_p*p - d_p")
# builder.add_synonym("neg_p_c_time", "ii_c + d_c - ii_p - d_p")
# ub = builder.ub("ii_c")*builder.ub("d_c")
# lb = -1*builder.lb("ii_p")*builder.ub("d_p")

# builder.add_indicator("neg_p_c_time", ub)

# builder.add_synonym("neg_c_p_time", "-1*neg_p_c_time")
# builder.add_indicator("neg_c_p_time", -1*lb)

# builder.set_objective('ii_p + ii_c')

# # Iteration domains
# builder.add_outer_forall("p", 1, 10)
# builder.add_outer_forall("c", 1, 10)

# builder.add_constraint_gez("I_neg_p_c_share + I_neg_c_p_share + I_neg_p_c_time + I_neg_c_p_time - 1")

# sol = builder.solve()
# for s in sol:
    # print('\t', s, '=', sol[s])

