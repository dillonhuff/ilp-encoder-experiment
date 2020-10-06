import os
from sympy import *
import copy

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

    def all_vars(self):
        vs = []
        for c in self.A:
            for v in c:
                if not v in vs:
                    vs.append(v)
        return vs

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
        for c in args:
            assert(isinstance(c, str))
        self.coeffs = args

    def __add__(self, other):
        assert(isinstance(other, LinearForm))

        cfs = copy.deepcopy(self.coeffs)
        for c in copy.deepcopy(other.coeffs):
            if c in cfs:
                cfs[c] = cfs[c] + other.coeffs[c]
            else:
                cfs[c] = other.coeffs[c]
        return LinearForm(cfs)

    def smul(self, k):
        cfs = {}
        for c in copy.deepcopy(self.coeffs):
            # print('coeff:', c)
            cfs[c] = k*self.coeffs[c]
        return LinearForm(cfs)

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

    def smul(self, k):
        cfs = {}
        for c in copy.deepcopy(self.coeffs):
            print(c)
            cfs[c] = k*self.coeffs[c]
        return QuadraticForm(cfs)

    def __add__(self, other):
        cfs = copy.deepcopy(self.coeffs)
        for c in copy.deepcopy(other.coeffs):
            if c in cfs:
                cfs[c] = cfs[c] + other.coeffs[c]
            else:
                cfs[c] = other.coeffs[c]
        return QuadraticForm(cfs)

    def __repr__(self):
        mms = []
        for c in self.coeffs:
            mms.append(str(self.coeffs[c]) + '*' + c[0] + '*' + c[1])
        if len(mms) == 0:
            return '0'
        return ' + '.join(mms)

def zero_qf():
    return QuadraticForm({})

def zero_lf():
    return LinearForm({})

class DLHS:

    def __init__(self, qf, lf, d):
        assert(isinstance(lf, LinearForm))

        self.qf = qf
        self.lf = lf
        self.d = d

    def all_vars(self):
        vs = []
        for e in self.qf.coeffs:
            if not e[0] in vs:
                vs.append(e[0])
            if not e[1] in vs:
                vs.append(e[1])
        for e in self.lf.coeffs:
            if not e in vs:
                vs.append(e)
        return vs

    def __sub__(self, other):
        return self + dsmul(-1, other)

    def __add__(self, other):
        return DLHS(self.qf + other.qf, self.lf + other.lf, self.d + other.d)

    def __repr__(self):
        ss = []
        if self.qf:
            ss.append(str(self.qf))
        if self.lf:
            ss.append(str(self.lf))
        if self.d:
            ss.append(str(self.d))
        if len(ss) == 0:
            return '0'

        return ' + '.join(ss)

class DConstraint:

    def __init__(self, lhs, comp):
        self.lhs = lhs
        self.comp = comp

    def __repr__(self):
        return str(self.lhs) + ' ' + self.comp + ' 0'

    def all_vars(self):
        return self.lhs.all_vars()

def gtc(expr):
    return DConstraint(expr, '>')

def eqc(expr):
    return DConstraint(expr, '=')

def lte(expr):
    return DConstraint(expr, '<=')

def gte(expr):
    return DConstraint(expr, '>=')

class Connective:

    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __repr__(self):
        mms = []
        for m in self.args:
            mms.append(str(m))
        return parens((' ' + self.name + ' ').join(mms))


def linf_lhs(v):
    return DLHS(zero_qf(), LinearForm(v), 0)

def lin_lhs(v):
    return DLHS(zero_qf(), LinearForm({ v : 1}), 0)

def const_lhs(v):
    return DLHS(zero_qf(), zero_lf(), v)

def dsmul(k, ss):
    return DLHS(ss.qf.smul(k), ss.lf.smul(k), ss.d*k)

def implies_constraint(rc_ne, dc):
    return Connective('->', [rc_ne, dc])

class ForallInPolyhedron:

    def __init__(self, polyhedron, formula):
        self.polyhedron = polyhedron
        self.formula = formula

    def __repr__(self):
        s = 'forall ' + str(self.polyhedron) + ' . ' + str(self.formula)
        return s

qf = QuadraticForm({('c', 'ii_c') : 1, ('p', 'ii_p') : -1})
dc = DConstraint(DLHS(qf, LinearForm({'d_c' : 1, 'd_p' : -1}), 1), '>=')
df = ForallInPolyhedron(deps, dc)

print('Data dependencies...')
print(df)

global unum
unum = 0

def uvar(prefix='v_'):
    global unum
    i = unum
    unum = unum + 1
    return prefix + str(i)

UPPER_BOUND = 99999

class FormulaBuilder:

    def __init__(self, orig):
        self.orig_c = orig
        self.ilp_constraints = []
        self.expr_vars = {}
        self.fm_vars = {}

    def indicator_uvar(self):
        v = uvar('I_')
        self.ilp_constraints.append(lte(lin_lhs(v) - const_lhs(1)))
        self.ilp_constraints.append(gte(lin_lhs(v)))
        return v

    def add_gte(self, a, b):
        self.ilp_constraints.append(gte((a) + dsmul(-1, (b))))

    def add_lte(self, a, b):
        self.ilp_constraints.append(lte((a) + dsmul(-1, (b))))

    def add_eqc(self, a, b):
        assert(isinstance(a, DLHS))
        assert(isinstance(b, DLHS))
        self.ilp_constraints.append(eqc((a) + dsmul(-1, (b))))

    def add_not(self, to_neg):
        vname = self.indicator_uvar()
        v = lin_lhs(vname)
        cs = eqc(v - (const_lhs(1) - lin_lhs(to_neg)))
        self.ilp_constraints.append(cs)
        return vname

    def add_and(self, av, bv):
        ae = lin_lhs(av)
        be = lin_lhs(bv)
        one = const_lhs(1)

        varname = self.indicator_uvar()
        v = lin_lhs(varname)

        self.ilp_constraints.append(lte(ae + be + dsmul(-2, v) - one))
        self.ilp_constraints.append(gte(ae + be + dsmul(-2, v)))
        return varname

    def add_or(self, av, bv):
        ae = lin_lhs(av)
        be = lin_lhs(bv)
        one = const_lhs(1)

        varname = self.indicator_uvar()
        v = lin_lhs(varname)

        self.ilp_constraints.append(gte(ae + be + dsmul(-2, v) + one))
        self.ilp_constraints.append(lte(ae + be + dsmul(-2, v)))
        return varname


    def add_cmp_var(self, var, comparator):
        if comparator == '=':
            res = self.add_and(self.add_cmp_var(var, '>='), self.add_cmp_var(var, '<='))
            return res
        elif comparator == '!=':
            return self.add_not(self.add_cmp_var(var, '='))
        elif comparator == '>=':
            return self.add_not(self.add_cmp_var(var, '<'))
        elif comparator == '<=':
            return self.add_not(self.add_cmp_var(var, '>'))
        elif comparator == '>':
            resname = self.indicator_uvar()
            be = const_lhs(0)
            ae = lin_lhs(var)
            self.add_gte(be - ae + dsmul(UPPER_BOUND, lin_lhs(resname)), const_lhs(0))
            self.add_lte(be - ae + dsmul(UPPER_BOUND, lin_lhs(resname)), const_lhs(UPPER_BOUND - 1))
            return resname
        elif comparator == '<':
            fresh_var = uvar()
            self.add_eqc(dsmul(-1, lin_lhs(var)), lin_lhs(fresh_var))
            return self.add_cmp_var(fresh_var, '>')
        else:
            print('Error: Unsupported comparator', comparator)
            assert(False)

    def build_equivalent_ilp(self, formula):
        print('\t', formula)
        if isinstance(formula, Connective):
            for subf in formula.args:
                self.build_equivalent_ilp(subf)

            self.fm_vars[formula] = uvar('FM_')
        else:
            assert(isinstance(formula, DConstraint))
            self.expr_vars[formula.lhs] = uvar('EX_')
            self.fm_vars[formula] = uvar('FM_')

    def build_boolean_constraints(self, formula):
        print('\t', formula)
        if isinstance(formula, Connective):
            for subf in formula.args:
                self.build_boolean_constraints(subf)
            if formula.name == '->':
                assert(len(formula.args) == 2)
                res = self.add_or(self.add_not(self.fm_vars[formula.args[0]]), self.fm_vars[formula.args[1]])
                self.ilp_constraints.append(eqc(lin_lhs(res) - lin_lhs(self.fm_vars[formula])))
            else:
                print('Error: Unrecognized connective in:', formula)
                assert(False)
        else:
            assert(isinstance(formula, DConstraint))
            fv = self.fm_vars[formula]
            atom_true = self.add_cmp_var(self.expr_vars[formula.lhs], formula.comp)
            self.ilp_constraints.append(eqc(lin_lhs(fv) - lin_lhs(atom_true)))

    def populate_ilp_constraints(self):
        self.build_equivalent_ilp(self.orig_c)
        self.build_boolean_constraints(self.orig_c)
        for e in self.expr_vars:
            self.ilp_constraints.append(eqc(e - lin_lhs(self.expr_vars[e])))

    def solve(self):
        self.populate_ilp_constraints()

        builder = ILPBuilder()
        print('ILP constraints..')
        for c in fb.ilp_constraints:
            print(c)
            for v in c.all_vars():
                if not v in builder.variables:
                    builder.add_int_var(v)
            builder.add_constraint(str(c))


        sol = builder.solve()
        return sol

dc = gtc(lin_lhs('a') - const_lhs(1))
fb = FormulaBuilder(dc)
fb.ilp_constraints.append(eqc(lin_lhs('a') - const_lhs(3)))
sol = fb.solve()
print('II solution...')
for s in sol:
    print('\t', s, '=', sol[s])

assert(sol['a'] > 1)
assert(sol[fb.fm_vars[dc]] == 1)

dc = gtc(lin_lhs('a') - const_lhs(1))
fb = FormulaBuilder(dc)
fb.ilp_constraints.append(eqc(lin_lhs('a') - const_lhs(-7)))
sol = fb.solve()
print('II solution...')
for s in sol:
    print('\t', s, '=', sol[s])

assert(sol['a'] == -7)
assert(sol[fb.fm_vars[dc]] == 0)

dc = gtc(lin_lhs('a') - const_lhs(1))
fb = FormulaBuilder(dc)
fb.ilp_constraints.append(eqc(lin_lhs('a') - const_lhs(-1)))
sol = fb.solve()
print('II solution...')
for s in sol:
    print('\t', s, '=', sol[s])

assert(sol['a'] == -1)
assert(sol[fb.fm_vars[dc]] == 0)

dc = gtc(lin_lhs('a') - const_lhs(1))
fb = FormulaBuilder(dc)
fb.ilp_constraints.append(eqc(lin_lhs('a') - const_lhs(1)))
sol = fb.solve()
print('II solution...')
for s in sol:
    print('\t', s, '=', sol[s])

assert(sol['a'] == 1)
assert(sol[fb.fm_vars[dc]] == 0)


dc = gte(lin_lhs('a') - const_lhs(1))
fb = FormulaBuilder(dc)
fb.ilp_constraints.append(eqc(lin_lhs('a') - const_lhs(1)))
sol = fb.solve()
print('II solution...')
for s in sol:
    print('\t', s, '=', sol[s])

assert(sol['a'] == 1)
assert(sol[fb.fm_vars[dc]] == 1)

dc = eqc(lin_lhs('a') - const_lhs(1))
fb = FormulaBuilder(dc)
fb.ilp_constraints.append(eqc(lin_lhs('a') - const_lhs(1)))
sol = fb.solve()
print('II solution...')
for s in sol:
    print('\t', s, '=', sol[s])

assert(sol['a'] == 1)
assert(sol[fb.fm_vars[dc]] == 1)

dc = eqc(lin_lhs('a') - const_lhs(1))
fb = FormulaBuilder(dc)
fb.ilp_constraints.append(eqc(lin_lhs('a') - const_lhs(7)))
sol = fb.solve()
print('II solution...')
for s in sol:
    print('\t', s, '=', sol[s])

assert(sol['a'] == 7)
assert(sol[fb.fm_vars[dc]] == 0)


deps = Polyhedron()
deps.add_constraint({'c' : 1}, 0)
deps.add_constraint({'c' : -1}, 10)

deps.add_constraint({'p' : 1}, 0)
deps.add_constraint({'p' : -1}, 10)

qf = QuadraticForm({('c', 'ii_c') : 1, ('p', 'ii_p') : -1})
dc = DConstraint(DLHS(qf, LinearForm({'d_c' : 1, 'd_p' : -1}), 0), '!=')

rc_ne = DConstraint(DLHS(zero_qf(), LinearForm({'r_c' : 1, 'r_p' : -1}), 0), '=')

dc = implies_constraint(rc_ne, dc)
df = ForallInPolyhedron(deps, dc)

print('Resource constraints...')
print(df)

fb = FormulaBuilder(dc)
fb.populate_ilp_constraints()
# The formula must be true
fb.ilp_constraints.append(eqc(lin_lhs(fb.fm_vars[dc]) - const_lhs(1)))
df.formula = Connective('^', fb.ilp_constraints)

print('Pre-farkas constraints...')
farkas_vars = df.polyhedron.all_vars()
for c in df.formula.args:
    apply_farkas = False
    print('\t', c)
    for v in c.lhs.all_vars():
        print('\t\t', v)
        if v in farkas_vars:
            print('\t\tMust have farkas applied')
            apply_farkas = True
            break
    if apply_farkas:
        fcs = []
        fm0 = uvar('fm_')
        fcs.append(gte(lin_lhs(fm0)))

        num_multipliers = df.polyhedron.num_constraints()
        fms = []
        for i in range(num_multipliers):
            fm = uvar('fm_')
            fms.append(fm)
            fcs.append(gte(lin_lhs(fm)))

        qf = c.lhs.qf

        lf = c.lhs.lf
        d = c.lhs.d

        lamb_dot = {}
        for i in range(num_multipliers):
            lamb_dot[fms[i]] = df.polyhedron.b[i]
        cst = linf_lhs(lamb_dot) + lin_lhs(fm0)
        fcs.append(eqc(DLHS(zero_qf(), lf, 0) + const_lhs(d) - cst))

        print('Constraints...')
        for c in fcs:
            print('\t', c)
        assert(False)


