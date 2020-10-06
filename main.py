import os
from sympy import *

glpk_path = '/usr/local/Cellar/glpk/4.65/bin/glpsol'
glpk_flags = '--model'

def parens(s):
    return '(' + s + ')'

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

    def __init__(self, coeff, components):
        self.coeff = coeff
        self.components = components 

    def __repr__(self):
        css = [str(self.coeff)]
        for c in self.components:
            css.append(str(c))
        return '*'.join(css)

class Polynomial:

    def __init__(self, monomials):
        self.monomials = monomials

    def __repr__(self):
        mss = []
        for m in self.monomials:
            mss.append(str(m))
        return ' + '.join(mss)

def parse_mono(txt):
    normed = txt.strip().replace(' ', '')
    terms = normed.split('*')
    return Monomial(terms[0], terms[1:])

def parse_poly(txt):
    normed = txt.strip().replace(' ', '')
    monos = normed.split('+')
    print('monos:', monos)
    monomials = []
    for p in monos:
        monomials.append(parse_mono(p))
    return Polynomial(monomials)

class Constraint:

    def __init__(self, expr, cmp):
        self.expr = expr
        self.cmp = cmp

    def __repr__(self):
        return str(self.expr) + ' ' + self.cmp + ' 0'

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

    def abs_max(self, name):
        return max(abs(self.ub(name)), abs(self.lb(name)))

    def add_synonym(self, name, expr):
        self.add_int_var(name)
        self.add_constraint_eqz(name + ' - ' + parens(expr))
        
    def add_int_var(self, name, lower=None, upper=None):
        assert(not name in self.variables)
        self.variables.append(name)
        self.variable_bounds[name] = (lower, upper)
        if (upper != None):
            self.add_constraint_lez(name + ' - ' + parens(str(upper)))
        if (lower != None):
            self.add_constraint_gez(name + ' - ' + parens(str(lower)))

    def add_constraint(self, cst):
        self.constraints.append(cst)

    def add_constraint_eqz(self, cst):
        self.constraints.append(Constraint((cst), '='))

    def add_constraint_gez(self, cst):
        self.constraints.append(Constraint((cst), '>='))

    def add_constraint_lez(self, cst):
        self.constraints.append(Constraint((cst), '<='))

    def set_objective(self, obj):
        self.objective = obj

    def is_zero_one_var(self, name):
        return self.variable_bounds[name][0] == 0 and self.variable_bounds[name][1] == 1

    def fresh_zero_one_var(self, prefix='zo_'):
        name = self.unique_name('zo_')
        self.add_int_var(name, 0, 1)
        return name

    def lte_var(self, a, b):
        return negate(gt_var(a, b))

    def eq_var(self, a, b):
        return conjoin([lte_var(a, b), lte_var(b, a)])
    
    def gt_var(self, a, b):
        k = max(self.abs_max(a), self.abs_max(b))
        K = 2*k + 1
        name = self.fresh_zero_one_var('gt_')
        self.add_constraint_gez('{0} - {1} + {2}*{3}'.format(b, a, K, name))
        self.add_constraint_lez('{0} - {1} + {2}*{3} - {2} + 1'.format(b, a, K, name))
        return name

    def conjoin(self, vars):
        assert(len(vars) > 0)

        for v in vars:
            print('v =', v)
            assert(self.is_zero_one_var(v))
        conj = self.fresh_zero_one_var('neg_')
        n = len(vars)
        sum_str = ' + '.join(vars)
        top = '(-2*{0})'.format(conj)
        self.add_constraint_gez('{0} + {1}'.format(sum_str, top))
        self.add_constraint_lez('{0} + {1} - 1'.format(sum_str, top))
        return conj

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

and_tests = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
for t in and_tests:
    builder = ILPBuilder()
    builder.add_int_var('a', 0, 1)
    builder.add_int_var('b', 0, 1)
    builder.add_constraint_eqz('a - {0}'.format(t[0]))
    builder.add_constraint_eqz('b - {0}'.format(t[1]))
    na = builder.conjoin(['a', 'b'])
    sol = builder.solve()
    print('solution...')
    for s in sol:
        print('\t', s, '=', sol[s])
    assert(sol['a'] == t[0])
    assert(sol['b'] == t[1])
    assert(sol[na] == t[2])

gt_tests = [[0, 0, 0], [0, 1, 0], [9, 7, 1], [1, -3, 1], [-3, -5, 1]]
for t in gt_tests:
    builder = ILPBuilder()
    builder.add_int_var('a', -1000, 1000)
    builder.add_int_var('b', -500, 250)
    builder.add_constraint_eqz('a - {0}'.format(t[0]))
    builder.add_constraint_eqz('b - {0}'.format(t[1]))
    na = builder.gt_var('a', 'b')
    sol = builder.solve()
    print('expected...')
    for s in range(3):
        print('\t', s, '=', t[s])
    print('solution...')
    for s in sol:
        print('\t', s, '=', sol[s])
    assert(sol['a'] == t[0])
    assert(sol['b'] == t[1])
    assert(sol[na] == t[2])

class Polyhedron:

    def __init__(self):
        self.A = []
        self.b = []

    def add_constraint(self, a, b):
        self.A.append(a);
        self.b.append(b)

    def __repr__(self):
        s = ''
        for i in range(len(self.A)):
            a = self.A[i]
            cs = []
            for v in a:
                cs.append(str(a[v]) + '*' + v)
            s += ' + '.join(cs)
            s += ' + ' + str(self.b[i]) + ' >= 0'
            s += '\n'
        return s

    def num_constraints(self):
        return len(self.A)

    def coeff(self, row, colname):
        r = self.A[row]
        if colname in r:
            return self.A[row][colname]
        else:
            return 0

def add_farkas_constraints(fs, fc, domain, build):
    # for c in build.constraints:
        # print(c)
        # for v in domain.A[0]:
            # if c.expr.find(sympify(v)):
                # print('\tContains variable:', v)
                # s = collect(c.expr, sympify(v))
                # print('\t', srepr(s))
                # for v in s.args:
                    # print('\t\t', v)
                    # print('\t\tdeg:', v.degree(sympify(v)))

    # assert(False)
    num_multipliers = domain.num_constraints()
    fms = []

    fm0 = build.unique_name('fm_')
    build.add_int_var(fm0, 0)

    for j in range(num_multipliers):
        fms.append(build.unique_name('fm_'))

    for v in fms:
        build.add_int_var(v, 0)

    constraints = []
    for v in fs:
        expr = fs[v]
        cexpr = []
        for j in range(num_multipliers):
            fmj = fms[j]
            Aji = domain.coeff(j, v)
            cexpr.append(str(Aji) + '*' + fmj)
        build.add_constraint_eqz(expr + ' - ' + parens(' + '.join(cexpr)))

    csts = []
    for j in range(num_multipliers):
        csts.append(fms[j] + '*' + str(domain.b[j]))
    cst = '{0} - {1} - {2}'.format(fc, parens(' + '.join(csts)), fm0)
    build.add_constraint_eqz(cst)
    return constraints

# builder = ILPBuilder()
# builder.add_int_var('ii_c', 1, 100)
# builder.add_int_var('d_c', 0, 100)

# builder.add_constraint(Constraint(parse_poly('1*ii_c*c + 1*d_c + -1*20'), '>='))

# deps = Polyhedron()
# deps.add_constraint({'c' : 1}, 0)
# deps.add_constraint({'c' : -1}, 10)

# fs = { 'c' : 'ii_c'}
# fc = 'd_c + -20'

# add_farkas_constraints(fs, fc, deps, builder)
# sol = builder.solve()
# assert(len(sol) > 0)

# Checking farkas constraints
builder = ILPBuilder()
builder.add_int_var('ii_c', 1, 100)
builder.add_int_var('d_c', 0, 100)

builder.objective = 'ii_c + d_c'

fs = { 'c' : 'ii_c'}
fc = 'd_c + -20'

deps = Polyhedron()
deps.add_constraint({'c' : 1}, 0)
deps.add_constraint({'c' : -1}, 10)

add_farkas_constraints(fs, fc, deps, builder)

sol = builder.solve()
print('II solution...')
for s in sol:
    print('\t', s, '=', sol[s])
assert(sol['ii_c'] == 1)
assert(sol['d_c'] == 20)


builder = ILPBuilder()
builder.add_int_var('ii_c', 1, 100)
builder.add_int_var('d_c', 0, 100)
builder.add_int_var('ii_p', 1, 100)
builder.add_int_var('d_p', 0, 100)

builder.objective = 'ii_c + d_c + ii_p + d_c'

fs = { 'c' : 'ii_c', 'p' : '-ii_p'}
fc = 'd_c - d_p - 1'

deps = Polyhedron()
deps.add_constraint({'c' : 1}, 0)
deps.add_constraint({'c' : -1}, 10)

deps.add_constraint({'p' : 1}, 0)
deps.add_constraint({'p' : -1}, 10)

deps.add_constraint({'c' : 1, 'p' : -1}, 0)
deps.add_constraint({'c' : -1, 'p' : 1}, 0)
add_farkas_constraints(fs, fc, deps, builder)

sol = builder.solve()
print('II solution...')
for s in sol:
    print('\t', s, '=', sol[s])

assert(sol['ii_c'] == 1)
assert(sol['d_c'] == 1)
assert(sol['ii_p'] == 1)
assert(sol['d_p'] == 0)

class LinearForm:

    def __init__(self, args):
        self.coeffs = args

    def __repr__(self):
        mms = []
        for c in self.coeffs:
            mms.append(str(self.coeffs[c]) + '*' + str(c))
        return ' + '.join(mms)

class AffineForm:

    def __init__(self, lexpr, d):
        self.expr = lexpr
        self.d = d

    def __repr__(self):
        return str(self.expr) + ' + ' + str(self.d)

class QuadraticForm:

    def __init__(self, args):
        self.coeffs = args

    def __repr__(self):
        s = '0'
        for c in self.coeffs:
            s += ' + ' + str(self.coeffs[c]) + '*' + c[0] + '*' + c[1]
        return s

class DConstraint:

    def __init__(self, expr, d, comp):
        self.expr = expr
        self.d = d
        self.comp = comp

    def __repr__(self):
        return str(self.expr) + ' + ' + str(self.d) + ' ' + self.comp + ' 0'

class Connective:

    def __init__(self, name, args):
        self.name = name
        self.args = args

class ForallInPolyhedron:

    def __init__(self, polyhedron, formula):
        self.polyhedron = polyhedron
        self.formula = formula

    def __repr__(self):
        s = 'forall ' + str(self.polyhedron) + ' . ' + str(self.formula)
        return s

qf = QuadraticForm({('c', 'ii_c') : 1, ('p', 'ii_p') : -1})
dc = DConstraint(qf, AffineForm(LinearForm({'d_c' : 1, 'd_p' : -1}), 1), '>=')

df = ForallInPolyhedron(deps, dc)
print(df)
