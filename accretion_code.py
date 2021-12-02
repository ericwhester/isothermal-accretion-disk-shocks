import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as de
import file_tools as flt
import glob as glob
import interpolation as ip
import sys
import time

from dedalus.tools import post
from scipy.integrate import solve_ivp, solve_bvp
from scipy.interpolate import interp1d, UnivariateSpline, splrep, sproot
from scipy.ndimage import gaussian_filter1d
from numpy.polynomial import Polynomial

d = de.operators.differentiate
integ = de.operators.integrate

def mag(x): return np.log10(np.abs(x)+1e-16)

# ideal/non dissipative problem
class Accretion:
    
    def __init__(self, l, g, rlim=None):
        if rlim == None: rlim = (1.5*g, 10)
        self.l = l
        self.g = g
        self.rlim = rlim

        self.sonic_points()
        self.directions()
        self.terminal_events()
        self.trajectories()
        self.find_projections()
        self.create_splines()
        self.find_shocks()
#         self.create_shock_splines()
    
    def energy(self, r, u):
        l, g = self.l, self.g
        return 0.5*(u**2+(l/r)**2) - 2/(r-g) - np.log(r*np.abs(u))
    
    def sonic_points(self):
        l, g = self.l, self.g
        coeff_list = [1, -2*(1+g), l**2 + g**2, -2*l**2*g, (l*g)**2]
        self.roots = np.roots(coeff_list).astype(complex)
        return self.roots

    def drdt(self, t, r, u):
        return 1

    def dudt(self, t, r, u):
        l, g = self.l, self.g
        return (1/r - 2/(r-g)**2 + l**2/r**3)/(u - 1/u)

    def RHS_inviscid(self,t,X):
        r, u = X
        return [self.drdt(t,r,u),self.dudt(t,r,u)]

    def Jacobian(self, r, u):
        l, g = self.l, self.g
        return np.array([[0 , 1+u**(-2)],
                [-3*l**2/r**4 + 4/(r-g)**3 - 1/r**2 , 0]])
        
    def directions(self):
        l, g = self.l, self.g
        self.sonics = np.sort([root.real for root in self.roots if root.real > g and np.abs(root.imag) < 1e-15])
        jacobians = [self.Jacobian(sonic, -1) for sonic in self.sonics]
        eigs = [np.linalg.eig(j) for j in jacobians]
        self.saddles = [sonic for sonic,eig in zip(self.sonics,eigs) if np.isreal(eig[0][0])]
        self.eigvals = [{i: eigval for i, eigval in enumerate(eig[0])} for eig in eigs if np.isreal(eig[0][0])]
        self.eigvecs = [{int(np.sign(ev[0]*ev[1])): ev*np.sign(ev[1]) for ev in eig[1].T} for eig in eigs if np.isreal(eig[0][0])]
    
    def terminal_events(self):
        self.events = {}
        for i, sonic in enumerate(self.saddles):
            def func(t, y):
                r, u = y
                return (r - sonic)**2 + (u + 1)**2 > 1e-8
            func.terminal = True
            self.events[i] = func
    
    def trajectories(self):
        eps = 1e-3
        self.paths = {}
        self.splines = {}
        for i, (sonic, eigpair) in enumerate(zip(self.saddles, self.eigvecs)):
            for j in [-1,1]:
                # in/out integration
                for k in [-1,1]:
                    # sub/super integration
                    r = sonic
                    u = -1
                    dr, du = eps*eigpair[j]*k*(-1)
                    if np.sign(dr) == 1: rend = self.rlim[1]
                    elif np.sign(dr) == -1: rend = self.rlim[0]
                    self.paths[i,j,k] = solve_ivp(self.RHS_inviscid,
                                                 (r+dr, rend),
                                                 [r+dr, u+du],
                                                 events=list(self.events.values()),
                                                 rtol=1e-10, atol=1e-10)

    def find_projections(self):
        self.projections = {}
        self.projection_splines = {}
        for i in range(len(self.saddles)):
            for j in [-1,1]:
                r, u = self.paths[i,j,1]['y']
                self.projections[i,j] = x, y = np.array([r, 1/u])
                self.projection_splines[i,j] = interp1d(x,y)
    
    def create_splines(self):
        self.splines = {}
        for i in range(len(self.saddles)):
            for j in [-1,1]:
                rs = np.concatenate([self.paths[i,j,k]['y'][0] for k in [-1,1][::-j]])
                us = np.concatenate([self.paths[i,j,k]['y'][1] for k in [-1,1][::-j]])
                self.splines[i,j] = interp1d(rs, us, fill_value=np.nan,bounds_error=False)
    
    def find_shocks(self):
        if len(self.saddles) > 1:
            projection = self.projection_splines[1,1]
            canard = self.splines[0,1]
            x1s, x2s = projection.x, canard.x
            xs = np.logspace(np.log10(max(x1s[0],x2s[0])+1e-10),np.log10(min(x1s[-1],x2s[-1])-1e-10))
            spl = splrep(xs, canard(xs)-projection(xs))
            self.shocks = sproot(spl)
        else: self.shocks = []
    
    def create_shock_splines(self):
        if self.shocks:
            self.shock_splines = {}
            self.shock_splines[0] = self.splines[1,1]
            for i, shock in enumerate(self.shocks):

                self.shock_splines[i+1] = lambda r, shock=shock:  np.piecewise(r, [r < shock, r>=shock], [self.splines[0,1],self.splines[1,1]])

    def stability_eigenvalue(self):
        r1 = self.shocks[0]
        u1 = self.shock_splines[0](r1)
        a1 = -1/r1**2
        b1 = - 4/(r1 - self.g)**3 + 3*self.l**2/r1**4
        c1 = [(b1-a1)*(u1-1/u1)**2, 0, (-2*b1 + a1*(u1**2 + 1/u1**2)), a1*(u1+1/u1)]

        r2 = self.shocks[1]
        u2 = self.shock_splines[0](r2)
        a2 = -1/r2**2
        b2 = - 4/(r2 - self.g)**3 + 3*self.l**2/r2**4
        c2 = [(b2-a2)*(u2-1/u2)**2, 0, (-2*b2 + a2*(u2**2 + 1/u2**2)), a2*(u2+1/u2)]

        return c1, c2
    
    def plot(self,urange=(-.001,-4),clean=False,trajectories=True):
        self.colors = colors = {
            (0,1,1):'C0',
            (0,1,-1):'C0',
            (0,-1,1):'C1',
            (0,-1,-1):'C1',
            (1,1,1):'C2',
            (1,1,-1):'C2',
            (1,-1,1):'C4',
            (1,-1,-1):'C4',}

        self.rs = rs = np.logspace(np.log(1.1*self.g), np.log(10), 1000)
        self.us = us = np.linspace(urange[0],urange[1],1000)
        self.energies = energies = self.energy(rs[:,None], us[None,:])
        self.sonic_energies = self.es = es = self.energy(np.array(self.sonics), -1)
        de = max(es) - min(es)
        if de < .1: de = max(es)
        self.contours = contours = sorted(list(max(es) + np.arange(-2,2,.1)*de))
        
        fig, ax = plt.subplots()

        ax.contourf(np.log(rs), (-us)**(1/2), energies.T,contours,cmap='RdBu_r')
        for p in self.paths:
            i, j, k = p
            ax.plot(np.log(self.paths[p]['y'][0]), (-self.paths[p]['y'][1])**(1/2),
                    color=colors[p],
                    label=f'Sonic point {i}, { {-1:"Inward",1:"Outward"}[j]} { {-1:"sub",1:"super"}[k]}sonic canard')

        ax.set(xlim=[np.log(1.1*self.g),np.log(10)],
               ylim=[0,np.sqrt(-urange[1])],
               xlabel='$\log r$',ylabel='$|u|^{1/2}$') 

        ax.set_facecolor('k')
        return fig, ax 


# def solve_steady(l,g,ϵ,n,boundaries,initial,
#                  angular_dissipation=False,maxits=150,minpert=1e-10):
#     r0, r1 = boundaries[0], boundaries[1]
#     u0, u1 = initial['u'](np.array([r0,r1]))
#     l1 = l
#     if len(boundaries) == 2:
#         rbasis = de.Chebyshev('r',n,interval=(r0,r1))
#     elif len(boundaries) == 3:
#         rb1 = de.Chebyshev('r1',n,interval=(boundaries[0:2]))
#         rb2 = de.Chebyshev('r2',n,interval=(boundaries[1:3]))
#         rbasis = de.Compound('r',[rb1,rb2])
#     domain = de.Domain([rbasis],grid_dtype=np.float64)
#     r, = domain.grids()

#      # Zeroth first order formulation
#     angular_dissipation = False
#     problem = de.NLBVP(domain, variables=['u','m']+angular_dissipation*['l','n'])
#     problem.meta[:]['r']['dirichlet'] = True
#     problem.parameters['ε'] = ϵ
#     problem.parameters['γ'] = g
#     problem.parameters['u0'] = u0
#     problem.parameters['u1'] = u1
#     problem.parameters['l1'] = l

#     if not angular_dissipation:
#         problem.substitutions['l'] = 'l1'

#     problem.substitutions['ρ'] = '-1/(r*u)'
#     problem.substitutions['p'] = 'ρ'
#     problem.substitutions['mop'] = 'p + ρ*u**2 - 2*ϵ*ρ*dr(r*u)/r'
#     problem.substitutions['nop'] = 'dr(l)/r'
#     problem.substitutions['res1'] = 'dr(m) + m/r + (r**2 - 2*r**3/(r-γ)**2 + l**2)/(r**4*u)'
#     problem.substitutions['res2'] = 'ϵ*(r*dr(u) + u) - (r/2)*(1 + u**2 - m/ρ)'

#     problem.add_equation('r*dr(m) + m = -(r**2 - 2*r**3/(r-γ)**2 + l**2)/(r**3*u)')
#     problem.add_equation('2*ϵ*(r*dr(u) + u) = r*(1 + u**2 - m/ρ)') 
#     problem.add_bc('left(u) = u0')
#     problem.add_bc('right(u) = u1')

#     if angular_dissipation:
#         problem.add_equation('r*n - dr(l) = 0')
#         problem.add_equation('ϵ*dr(n) = u*n + ϵ*(1/r + dr(u)/u)*(n - 2*l/r**2)')
#         problem.add_bc('left(n) = 0')
#         problem.add_bc('right(l) = l1')

#     solver = problem.build_solver()

#     if angular_dissipation:
#         fields = u,m,l,n = [solver.state[a] for a in problem.variables]
#         l['g'] = l1
#         n['g'] = solver.evaluator.vars['nop'].evaluate()['g']

#     elif not angular_dissipation:
#         fields = u,m = [solver.state[a] for a in problem.variables]
#     for f in fields: f.set_scales(domain.dealias)
#     u['g'] = initial['u'](r)
#     u['g'] = gaussian_filter1d(u['g'], 1)
#     if 'm' in initial: m['g'] = initial['m'](r)
#     else: m['g'] = -(1/r)*(u['g']+1/u['g'])

#     its = 0
#     perturbations = solver.perturbations.data
#     perturbations.fill(1+1e-10)
#     pert = np.sum(np.abs(solver.perturbations.data))
#     while its < maxits and pert > minpert:
#         if pert > 1e-2: damping = .5
#         else: damping = 1
#         solver.newton_iteration(damping=damping)
#         its += 1
#         pert = np.sum(np.abs(solver.perturbations.data))
#         print(pert)
#         if pert > 1e5: break
#     return {'domain':domain,'r':r,'u':u,'m':m}

def replace(string, replacements):
    for r in replacements:
        s, e = r
        string = string.replace(s, e)
    return string

def solve_steady_compound(params,nr,boundaries,initial,
                          l_visc=False,maxits=150,minpert=1e-10, maxpert=1e6, start_damping=.5, thresh_damping=1, end_damping=1, flatten_inner=False):

    sbasis = de.Chebyshev('s',nr,interval=(0,1))
    domain = de.Domain([sbasis],grid_dtype=np.float64)
    s, = domain.grids()
    rs = {}
    N = len(boundaries) - 1
    for i in range(N):
        r_1, r_2 = boundaries[i:i+2]
        rs[i] = r_1 + (r_2 - r_1)*s

    # Zeroth first order formulation
    variables = ['u','m'] + ['l','n']*l_visc
    problem = de.NLBVP(domain, variables=[f + str(i) for f in variables for i in range(N)])
    problem.meta[:]['s']['dirichlet'] = True
    for param in params: problem.parameters[param] = params[param]
    for i in range(N+1): problem.parameters[f'r_{i}'] = boundaries[i]
    for i in range(N):
        reps = [(f'{f}', f'{f}{i}') for f in 'lmnpruδρ']
        problem.substitutions[f'r{i}'] = f'r_{i} + (r_{i+1} - r_{i})*s'
        problem.substitutions[f'c{i}'] = f'1/(r_{i+1}-r_{i})'
        problem.substitutions[f'dr{i}(A)'] = f'c{i}*ds(A)'
        problem.substitutions[f'ρ{i}'] = f'-1/(r{i}*u{i})'
        problem.substitutions[f'p{i}'] = f'ρ{i}'
        problem.substitutions[f'dr{i}δ{i}'] = f'-(1/r{i} + dr{i}(u{i})/u{i})'
        problem.substitutions[f'mop{i}'] = replace('p + ρ*u**2 - 2*ϵ*ρ*dr(r*u)/r',reps)
        problem.substitutions[f'res_m{i}'] = replace('dr(m) + m/r + (r**2 - 2*r**3/(r-γ)**2 + l**2)/(r**4*u)',reps)
        problem.substitutions[f'res_u{i}'] = replace('ϵ*(r*dr(u) + u) - (r/2)*(1 + u**2 - m/ρ)',reps)
        if l_visc:
            problem.substitutions[f'nop{i}'] = replace('l + ϵ*ρ*r**3*dr(l/r**2)',reps)
            problem.substitutions[f'res_l{i}'] = replace('ϵ*(r*dr(l) - 2*l) - r*u*(l-n)',reps)
            problem.substitutions[f'res_n{i}'] = replace('dr(n)',reps)
#             problem.substitutions[f'res_l{i}'] = replace('ϵ*(r*dr(n) - n + (r*n-2*l)*drδ) - r*n*u',reps)
#             problem.substitutions[f'res_n{i}'] = replace('n - dr(l)',reps)
    for i in range(N):
        reps = [(f'{f}', f'{f}{i}') for f in 'lmnpruδρ']
#        problem.add_equation(replace('dr(l) = 0',reps))
        problem.add_equation(replace('2*ϵ*(r*dr(u) + u) = r*(1 + u**2 - m/ρ)',reps)) 
        problem.add_equation(replace('r*dr(m) + m = -(r**2 - 2*r**3/(r-γ)**2 + l**2)/(r**3*u)',reps))
        if l_visc:
            problem.add_equation(replace('ϵ*(r*dr(l) - 2*l) = r*u*(l-n)',reps))
            problem.add_equation(replace('dr(n) = 0',reps))
#             problem.add_equation(replace('n - dr(l) = 0',reps))
#             problem.add_equation(replace('  ϵ*(r*dr(n) - n) = r*n*u + ϵ*(2*l-r*n)*drδ',reps))
#            problem.add_equation(replace('  ϵ*(r**2*dr(n) - 2*r*n + 2*l) = r**2*n*u + ϵ*r*(r*n - 2*l)*dr(u)/u',reps))
        
    problem.add_bc('left(u0) = u_l')
    problem.add_bc(f'right(u{N-1}) = u_r')
    for i in range(N-1):
        problem.add_bc(f'right(u{i}) - left(u{i+1}) = 0')
        problem.add_bc(f'right(m{i}) - left(m{i+1}) = 0')
    if l_visc:
        for i in range(N-1):
            problem.add_bc(f'right(l{i}) - left(l{i+1}) = 0')

            problem.add_bc(f'right(n{i}) - left(n{i+1}) = 0')
        problem.add_bc(f'left(l0) = l_l')
        problem.add_bc(f'right(l{N-1}) = l_r')
    
    solver = problem.build_solver()

    fields = {fname: solver.state[fname] for fname in problem.variables}
    for f in fields: fields[f].set_scales(domain.dealias)
    for i in range(N):
#         print(i, rs[i][0], rs[i][-1], initial[]'u'))
        fields[f'u{i}']['g'] = initial['u'](rs[i])
        fields[f'm{i}']['g'] = solver.evaluator.vars[f'mop{i}']['g']
        if l_visc:
            fields[f'l{i}']['g'] = initial['l'](rs[i])
            fields[f'n{i}']['g'] = solver.evaluator.vars[f'nop{i}']['g']
            if flatten_inner:
                fields['m1']['g'] = params['m_inner']
                fields['n1']['g'] = params['n_inner']

    residuals = {f'res_{a}': solver.evaluator.vars[f'res_{a}'] for a in problem.variables}

    its = 0
    perturbations = solver.perturbations.data
    perturbations.fill(1+1e-10)
    pert = np.sum(np.abs(solver.perturbations.data))
    while its < maxits and pert > minpert:
        if pert > thresh_damping: damping = start_damping
        else: damping = end_damping
        solver.newton_iteration(damping=damping)
        its += 1
        pert = np.sum(np.abs(solver.perturbations.data))
        print(pert)
        if pert > maxpert: break
    output = {'domain':domain,'boundaries':boundaries,'s':s}
    output['grid'] = {i: rs[i] for i in rs}
    output['fields'] = {f: fields[f] for f in problem.variables}
    splines = {f: Spline(boundaries, [fields[f'{f}{i}'] for i in range(N)]) for f in 'ulmn'}
    return {'splines':splines, 'residuals':residuals, 'params':params, 'initial':initial, **output}


def make_intervals(r, boundaries):
    rs = boundaries
    N = len(rs) - 1
    return [(r >= rs[i]) & (r < rs[i+1]) for i in range(N-1)] + [(r >= rs[N-1]) & (r <= rs[N])]

def make_interpolants(dictionary,variables='um'):
    s = dictionary
    rs = boundaries = dictionary['boundaries']
    fields = dictionary['fields']
    N = len(rs) - 1
    interp_s = lambda r: np.piecewise(r,make_intervals(r,rs),
                             [lambda r, i=i, r_1=rs[i], r_2=rs[i+1]: (r - r_1)/(r_2 - r_1) for i in range(N)])
    interps = {}
    for f in variables:
        interps[f] = lambda r, f=f: np.piecewise(r,make_intervals(r,rs),
                             [lambda r, i=i: ip.interp(fields[f'{f}{i}'],interp_s(r)) for i in range(N)]) 
    interp_δ = lambda r: np.log(-1/(r*interps['u'](r)))
    return {'sf':interp_s, 'δ':interp_δ, **interps}

class Spline:
    
    def __init__(self, boundaries, fields):
        self.boundaries = bs = boundaries
        self.N = N = len(bs) - 1
        self.fields = fields
        self.domain = fields[0].domain
        self.local_grid = s = self.domain.grids()[0]
        self.global_grids = [(bs[i+1] - bs[i])*s + bs[i] for i in range(N)]
        self.ddxs = [1/(bs[i+1] - bs[i]) for i in range(N)]
    
    def differentiate(self):
        self.derivatives = [(ddx*field.differentiate('s')).evaluate() for ddx, field in zip(self.ddxs, self.fields)]
        return Spline(self.boundaries, self.derivatives)
    
    def to_local(self, x):
        bs = self.boundaries
        N = self.N
#         import ipdb; ipdb.set_trace()
        slices = [x < bs[0]] + [(x >= bs[i]) & (x < bs[i+1] if i < N-1 else x <= bs[i+1]) for i in range(N)] + [x > bs[N]]
        interp_funcs = [lambda x: np.nan*x] + [lambda x, j=i: (x - bs[j])/(bs[j+1] - bs[j]) for i in range(N)] + [lambda x: np.nan*x]
        local_grids = [interp_funcs[i](x[slices[i]]) for i in range(len(slices))]
        return local_grids[1:-1]

    def interp(self, x):
        local_grids = self.to_local(x)
        return np.concatenate([ip.interp(field, local_grid) for field, local_grid in zip(self.fields, local_grids)])
    
    def __getitem__(self,index):
        return [field[index] for field in self.fields]
    
    def __call__(self, x):
        if np.isscalar(x): return self.interp(np.array(x)).item()
        else: return self.interp(x)
        
class FuncSpline:
    
    def __init__(self, boundaries, funcs):
        self.boundaries = bs = boundaries
        self.N = N = len(bs) - 1
        self.funcs = funcs
    
    def cond_list(self, x):
        bs, N = self.boundaries, self.N
        return [(x >= bs[i]) & (x < bs[i+1] if i < N-1 else x <= bs[i+1]) for i in range(N)]
    
    def __call__(self, x):
        return np.piecewise(x, self.cond_list(x), self.funcs)
#         if np.isscalar(x): return self.interp(np.array(x)).item()
        

def save_steady(dictionary, save_dir, sim_name):
    flt.makedir(save_dir)
    N = len(dictionary['boundaries']) - 1
    for dname in ['l','g','ϵ','boundaries','s'] + [f'r{i}' for i in range(N)]:
        flt.save_data(f'{save_dir}/data-{sim_name}.h5', dictionary[dname], dname)
    for fname in [f + str(i) for f in 'um' for i in range(N)]:
        flt.save_data(f'{save_dir}/data-{sim_name}.h5', dictionary[fname]['g'], fname)
    flt.save_domain(f'{save_dir}/domain-{sim_name}.h5', dictionary['domain'])    

def load_steady(save_dir, sim_name):
    domain = flt.load_domain(f'{save_dir}/domain-{sim_name}.h5')
    output = {}
    boundaries, = flt.load_data(f'{save_dir}/data-{sim_name}.h5', 'boundaries')
    N = len(boundaries) - 1
    for dname in ['l','g','ϵ','boundaries','s']+[f'r{i}' for i in range(N)]:
        output[dname], = flt.load_data(f'{save_dir}/data-{sim_name}.h5',dname)
    field_names = [f + str(i) for f in 'um' for i in range(N)]
    for fname in field_names:
        output[fname] = domain.new_field()
        output[fname]['g'], = flt.load_data(f'{save_dir}/data-{sim_name}.h5',fname,)
    splines = make_interpolants(output)
    output = {**output,**splines}
    return output
    
def solve_initial(params):
    l,g,ϵ,r_1,r_2,n,dt,initial = [params[_] for _ in ['l','g','ϵ','r_1','r_2','n','dt','initial']]
    sim_name,save_dir,print_freq,max_sim_time,timestepper,max_wall_time,max_iteration,save_step,save_max,dealias = [params[_] for _ in ['sim_name','save_dir','print_freq','max_sim_time','timestepper','max_wall_time','max_iteration','save_step','save_max','dealias']]
    boundaries = (r_1, r_2)
    rbasis = de.Chebyshev('r',n,interval=boundaries,dealias=dealias)
    u_1, u_2 = initial['u'](np.array([r_1,r_2]))
    l_1, l_2 = initial['l'](np.array([r_1,r_2]))
    δ_2 = np.log(-1/(r_2*u_2))
    domain = de.Domain([rbasis],grid_dtype=np.float64)
    r, = domain.grids()

    # Zeroth first order formulation
    problem = de.IVP(domain, variables=['u','δ','q'])
    problem.meta[:]['r']['dirichlet'] = True
    problem.parameters['ε'] = ϵ
    problem.parameters['g'] = g
    problem.parameters['l'] = l
    problem.parameters['u_1'] = u_1
    problem.parameters['u_2'] = u_2
    problem.parameters['l_1'] = u_1
    problem.parameters['l_2'] = u_2
    problem.parameters['δ_2'] = δ_2

    problem.substitutions['ρ'] = 'exp(δ)'
    problem.substitutions['p'] = 'ρ'
    problem.substitutions['n_op'] = 'dr(l)'
    problem.substitutions['q_op'] = 'dr(r*u)/r'
    problem.substitutions['res_δ'] = '- q - u*dr(δ)'
    problem.substitutions['res_u'] = '-dr(δ) - u*q + u**2/r + l**2/r**3 - 2/(r-g)**2 + ϵ*2*(dr(q) + dr(u)*dr(δ))'
    problem.substitutions['res_l'] = '-u*n + ϵ*(dr(n) - n/r + (n - 2*l/r)*dr(δ))'

    problem.add_equation('dt(δ) + q - dr(δ) = -(u+1)*dr(δ)')
    problem.add_equation('r*q - dr(r*u) = 0')
    problem.add_equation('dt(u) + dr(δ) - ϵ*2*dr(q) = -u*q + u**2/r + l**2/r**3 - 2/(r-g)**2 + ϵ*2*dr(u)*dr(δ)')
    problem.add_equation('n - dr(l) = 0')
    problem.add_equation('dt(l) - ϵ*dr(n) = - u*n + ϵ*(-n/r + (n - 2*l/r)*dr(δ))')
    
    problem.add_bc('left(u) = u_1')
    problem.add_bc('left(l) = l_1')
    # problem.add_bc('left(q) = 0')
    #problem.add_bc('left(dr(q) - dr(u)) = left(-(1 + dr(δ))*dr(u))')
    problem.add_bc('right(u) = u_2')
    problem.add_bc('right(l) = l_2')
    problem.add_bc('right(δ) = δ_2')

    solver = problem.build_solver(getattr(de.timesteppers,timestepper))
    solver.stop_sim_time = max_sim_time
    solver.stop_wall_time = max_wall_time
    solver.stop_iteration = max_iteration
    save_name = f'analysis-{sim_name}'
    
    fs = {}
    for f in problem.variables:
        fs[f] = solver.state[f]
        fs[f].set_scales(domain.dealias)        
    
    reload = False
    if reload:
        save_file = sorted(glob.glob(f'{save_dir}/{save_name}/*.h5'))[-1]
        write, _ = solver.load_state(save_file,-1)
    else:
        u0 = fs['u']['g'] = initial['u'](r)
        l0 = fs['l']['g'] = initial['l'](r)
        δ0 = fs['δ']['g'] = - np.log(-r*fs['u']['g'])
        q0 = fs['q']['g'] = solver.evaluator.vars['q_op']['g']
        n0 = fs['n']['g'] = solver.evaluator.vars['n_op']['g']
    
    dtu = solver.evaluator.vars['res_u']
    dtl = solver.evaluator.vars['res_l']
    dtδ = solver.evaluator.vars['res_δ']

    analysis = solver.evaluator.add_file_handler(f'{save_dir}/{save_name}',sim_dt=save_step,max_writes=save_max,mode='overwrite' if not reload else 'append')
    for task in problem.variables: analysis.add_task(task)
    analysis.add_task("dtu")
    analysis.add_task("dtl")
    analysis.add_task("dtδ")

    start_time = time.time()

    while solver.ok:
        solver.step(dt)
        if solver.iteration % print_freq == 0: 
            print(solver.iteration, 
                  f'{solver.sim_time:.3f}', 
                  f'{(fs["u"]["g"]-u0).max():2f}', 
                  f'{(fs["u"]["g"]-u0).min():2f}')
            if np.any(np.isnan(fs["u"]['g'])): break
            sys.stdout.flush()
    solver.step(dt)    

def solve_initial_perturbation(params):
    l,g,ϵ,r_1,r_2,A,n,dt,initial = [params[_] for _ in ['l','g','ϵ','r_1','r_2','A','n','dt','initial']]
    sim_name,save_dir,print_freq,max_sim_time,timestepper,max_wall_time,max_iteration,save_step,save_max,dealias = [params[_] for _ in ['sim_name','save_dir','print_freq','max_sim_time','timestepper','max_wall_time','max_iteration','save_step','save_max','dealias']]
    boundaries = (r_1, r_2)
    rbasis = de.Chebyshev('r',n,interval=boundaries,dealias=dealias)
    domain = de.Domain([rbasis],grid_dtype=np.float64)
    r, = domain.grids(domain.dealias)
    
    fs = {f+'0': domain.new_field() for f in 'δuqln'}
    for f in fs: fs[f].set_scales(domain.dealias)
    fs['u0']['g'] = initial['u0'](r)
    fs['l0']['g'] = initial['l0'](r)
    fs['δ0']['g'] = -np.log(-r*fs['u0']['g'])
    fs['q0']['g'] = fs['u0'].differentiate('r')['g'] + fs['u0']['g']/r
    fs['n0']['g'] = fs['l0'].differentiate('r')['g']    

    # Zeroth first order formulation
    problem = de.IVP(domain, variables=['δ1','u1','q1','l1','n1'])
    problem.meta[:]['r']['dirichlet'] = True
    problem.parameters['ε'] = ϵ
    problem.parameters['g'] = g
    for f in fs: problem.parameters[f] = fs[f]
    # problem.substitutions['n_op(l)'] = 'dr(l)'
    # problem.substitutions['q_op(u)'] = 'dr(r*u)/r'
    problem.substitutions['res_δ1'] = '-(u0*dr(δ1) + dr(δ0)*u1 + q1 + u1*dr(δ1))'
    problem.substitutions['res_u1'] = '-(dr(δ1) + u0*q1 + q0*u1 - ϵ*2*(dr(q1) + dr(u0)*dr(δ1) + dr(u1)*dr(δ0)) - (2*u0+u1)*u1/r - (2*l0+l1)*l1/r**3 + u1*q1 - 2*ϵ*dr(u1)*dr(δ1))'
    problem.substitutions['res_l1'] = '-(u0*n1 + u1*n0 - ϵ*(dr(n1) + n1*dr(δ0) + n0*dr(δ1)) + n1*u1 + ϵ*(n1 + 2*(l1*dr(δ0) + l0*dr(δ1)) - (r*n1-2*l1)*dr(δ1))/r)'

    problem.add_equation('dt(δ1) + u0*dr(δ1) + u1*dr(δ0) + q1 = -u1*dr(δ1)')
    problem.add_equation('r*q1 - dr(r*u1) = 0')
    problem.add_equation('dt(u1) + dr(δ1) + u0*q1 + u1*q0 - ϵ*2*(dr(q1) + dr(u0)*dr(δ1) + dr(u1)*dr(δ0)) = (2*u0+u1)*u1/r + (2*l0+l1)*l1/r**3 - u1*q1 + 2*ϵ*dr(u1)*dr(δ1)')
    problem.add_equation('n1 - dr(l1) = 0')
    problem.add_equation('dt(l1) + u0*n1 + u1*n0 - ϵ*(dr(n1) + n1*dr(δ0) + n0*dr(δ1)) = -n1*u1 + ϵ*(-n1 - 2*(l1*dr(δ0) + l0*dr(δ1)) + (r*n1-2*l1)*dr(δ1))/r')

    problem.add_bc('right(δ1) = 0')
    problem.add_bc('right(u1) = 0')
    problem.add_bc('right(l1) = 0')
    problem.add_bc('left(u1) = 0')
    problem.add_bc('left(l1) = 0')

    solver = problem.build_solver(getattr(de.timesteppers,timestepper))
    solver.stop_wall_time = max_wall_time
    solver.stop_iteration = max_iteration
    solver.stop_sim_time = max_sim_time

    for f in problem.variables:
        fs[f] = solver.state[f]
        fs[f].set_scales(domain.dealias)

    dtδ = solver.evaluator.vars['res_δ1']
    dtu = solver.evaluator.vars['res_u1']
    dtl = solver.evaluator.vars['res_l1']

    for f1 in problem.variables:
        f = f1[0]
        fs[f1]['g'] = A*fs[f+'0'].differentiate('r')['g']

    flt.makedir(save_dir)
    analysis = solver.evaluator.add_file_handler(f'{save_dir}/analysis-{sim_name}',sim_dt=save_step,max_writes=save_max,mode='overwrite')
    for task in problem.variables: analysis.add_task(task)

    parameters = solver.evaluator.add_file_handler(f'{save_dir}/parameters-{sim_name}',sim_dt=np.inf,max_writes=save_max,mode='overwrite')
    for task in problem.variables + ['u0','δ0','q0']: parameters.add_task(task)

    start_time = time.time()

    while solver.ok:
        solver.step(dt)
        if solver.iteration % print_freq == 0: 
            print(solver.iteration, 
                  f'{solver.sim_time:.3f}', 
                  f'{(fs["u1"]["g"]).max():2f}', 
                  f'{(fs["u1"]["g"]).min():2f}')
            if np.any(np.isnan(fs["u1"]['g'])): break
            sys.stdout.flush()
    solver.step(dt)    

    
def get_saves(save_dir, sim_name):
    post.merge_analysis(f'{save_dir}/{sim_name}')
    return sorted(glob.glob(f'{save_dir}/{sim_name}/*.h5'))


# asymptotic solutions

def solve_outer_zeroth(l, g, n, boundaries, initial,
                       maxits=50, minpert=1e-13, start_damping=.1, thresh_damping=10):

    bs = boundaries
    Ns = {side: len(bs[side]) - 1 for side in bs}
    r_l = bs['l'][0]
    r_r = bs['r'][-1]
    u_l = initial['ul'](r_l).item()
    u_r = initial['ur'](r_r).item()
    δ_l = initial['δl'](r_l)
    δ_r = initial['δr'](r_r)

    sbasis = de.Chebyshev('s',n,interval=(0,1))
    domain = de.Domain([sbasis],grid_dtype=np.float64)
    s, = domain.grids()
    rs = {}
    for side in 'lr':
        b = bs[side]
        N = Ns[side]
        for i in range(N):
            rs[f'{side}{i}'] = b[i] + (b[i+1] - b[i])*s

    problem = de.NLBVP(domain, variables=[f'{f}{side}{i}' for f in 'uδ' for side in 'lr' for i in range(Ns[side])])
    problem.meta[:]['s']['dirichlet'] = True
    problem.parameters['γ'] = g
    problem.parameters['δ_l'] = δ_l
    problem.parameters['u_l'] = u_l
    problem.parameters['δ_r'] = δ_r
    problem.parameters['u_r'] = u_r
    problem.parameters['l'] = l
    for j in ['l','r']:
        for i in range(Ns[j]+1):
            problem.parameters[f'r_{j}{i}'] = bs[j][i]

        for i in range(Ns[j]):
            problem.substitutions[f'r{j}{i}'] = f'r_{j}{i} + (r_{j}{i+1} - r_{j}{i})*s'
            problem.substitutions[f'c{j}{i}'] = f'1/(r_{j}{i+1}-r_{j}{i})'
            problem.substitutions[f'dr{j}{i}(A)'] = f'c{j}{i}*ds(A)'
            problem.substitutions[f'ρ{j}{i}'] = f'-1/(r{j}{i}*u{j}{i})'
            problem.substitutions[f'p{j}{i}'] = f'ρ{j}{i}'
            problem.substitutions[f'ρ{j}{i}']
            problem.substitutions[f'res_δ{j}{i}'] = f'dr{j}{i}(r{j}{i}*u{j}{i})/r{j}{i} + u{j}{i}*dr{j}{i}(δ{j}{i})'
            problem.substitutions[f'res_u{j}{i}'] = f'dr{j}{i}(δ{j}{i}) + u{j}{i}*dr{j}{i}(u{j}{i}) + (2/(r{j}{i}-γ)**2 - l**2/r{j}{i}**3)'

    for j in ['l','r']:
        for i in range(Ns[j]):
        #     problem.add_equation(f'2*ϵ*(r{j}{i}*dr{j}{i}(u{j}{i}) + u{j}{i}) = r{j}{i}*(1 + u{j}{i}**2 - m{i}/ρ{j}{i})') 
        #     problem.add_equation(f'r{j}{i}*dr{j}{i}(m{i}) + m{i} = -(r{j}{i}**2 - 2*r{j}{i}**3/(r{j}{i}-γ)**2 + l**2)/(r{j}{i}**3*u{j}{i})')
            problem.add_equation(f'r{j}{i}*dr{j}{i}(u{j}{i}) + u{j}{i} = - r{j}{i}*u{j}{i}*dr{j}{i}(δ{j}{i})')
            problem.add_equation(f'dr{j}{i}(δ{j}{i}) = - u{j}{i}*dr{j}{i}(u{j}{i}) - (2/(r{j}{i}-γ)**2 - l**2/r{j}{i}**3)')

    for j in 'lr':
        for i in range(Ns[j]-1):
            for f in 'uδ':
                problem.add_bc(f'right({f}{j}{i}) - left({f}{j}{i+1}) = 0')
    problem.add_bc(f'left(δl0) = left(-log(-rl0*ul0))')
    problem.add_bc(f'right(ul0) = -1')
    Nr = Ns['r']
    problem.add_bc(f'right(δr{Nr-1}) = right(-log(-rr{Nr-1}*ur{Nr-1}))')
    problem.add_bc(f'left(ur{Nr-1}) = -1')

    solver = problem.build_solver()

    fields_zero = {fname: solver.state[fname] for fname in problem.variables}
    for f in fields_zero: fields_zero[f].set_scales(domain.dealias)
    for j in ['l','r']:
        for i in range(Ns[j]): 
            fields_zero[f'u{j}{i}']['g'] = initial[f'u{j}'](rs[f'{j}{i}'])
            fields_zero[f'δ{j}{i}']['g'] = initial[f'δ{j}'](rs[f'{j}{i}'])

    residuals = {f'res_{a}': solver.evaluator.vars[f'res_{a}'] for a in problem.variables}

    its = 0
    perturbations = solver.perturbations.data
    perturbations.fill(.1+1e-10)
    pert = np.sum(np.abs(solver.perturbations.data))

    while its < maxits and pert > minpert:
        if pert > thresh_damping: damping = start_damping
        else: damping = 1
        solver.newton_iteration(damping=damping)
        its += 1
        pert = np.sum(np.abs(solver.perturbations.data))
        print(pert)
        if pert > 1e5: break
    
    splines = {}
    for j in 'lr':
        for f in 'uδ':
            subfields = [fields_zero[f'{f}{j}{i}'] for i in range(Ns[j])]
            splines[f'{f}{j}0'] = Spline(bs[j], subfields)
    
    return {'l':l,'g':g,'initial':initial,'rs':rs,'fields':fields_zero,'residuals':residuals,'splines':splines}
            
def find_shock(r0, bs, us):
    bl, br = bs['l'], bs['r']
    ulf, urf = us['l'], us['r']
    cl = 1/(bl[1] - bl[0])
    cr = 1/(br[1] - br[0])
    ul_rf = (cl*ulf.differentiate('s')).evaluate()
    ur_rf = (cr*urf.differentiate('s')).evaluate()

    def diff(r):
        sl = (r-bl[0])/(bl[1] - bl[0])
        sr = (r-br[0])/(br[1] - br[0])
        ul = ulf.interpolate(s=sl)['g'][0]
        ur = urf.interpolate(s=sr)['g'][0]
        ul_r = ul_rf.interpolate(s=sl)['g'][0]
        ur_r = ur_rf.interpolate(s=sr)['g'][0]
        return ul - 1/ur, ul_r + ur_r/ur**2

    def shock_iteration(r0, maxits=10):
        it = 0
        r = r0
        while it < maxits:
            f, fr = diff(r)
            r -= f/fr
            it += 1
        return r

    r_s = shock_iteration(r0)
    sl_s = (r_s - bl[0])/(bl[1] - bl[0])
    sr_s = (r_s - br[0])/(br[1] - br[0])
    return r_s, sl_s, sr_s    

def interp_u0_shocks(r, r_s, splines):
    return np.piecewise(r,[r<=r_s,r>r_s],[splines['l'],splines['r']])

def solve_outer_first(l, g, n, boundaries, u0_func, l1_l, l1_r):
    sbasis = de.Chebyshev('s',n,interval=(0,1))
    domain = de.Domain([sbasis],grid_dtype=np.float64)
    s, = domain.grids()
    rs1 = {}
    Ns = {side: len(boundaries[side])-1 for side in boundaries}
    for side in ['l','r']:
        for i in range(Ns[side]):
            r_0, r_1 = boundaries[side][i], boundaries[side][i+1]
            rs1[side+str(i)] = r_0 + (r_1 - r_0)*s

    zeroth = {}
    for side in ['l', 'r']:
        for i in range(Ns[side]):
            zeroth[f'u{side}0{i}'] = domain.new_field()
            zeroth[f'u{side}0{i}']['g'] = u0_func[side](rs1[side+str(i)])
            zeroth[f'δ{side}0{i}'] = domain.new_field()
            zeroth[f'δ{side}0{i}']['g'] = -np.log(-rs1[side+str(i)] * zeroth[f'u{side}0{i}']['g'])     
            zeroth[f'λ{side}0{i}'] = domain.new_field()
            zeroth[f'λ{side}0{i}']['g'] = l
    
    problem = de.LBVP(domain, variables=[f'{f}{side}1{i}' for f in 'uδλ' for side in 'lr' for i in range(Ns[side])])
    problem.meta[:]['s']['dirichlet'] = True
    problem.parameters['γ'] = g
    for side in 'lr':
        for i in range(Ns[side]):
            for f in 'uδ':
                key = f'{f}{side}0{i}'
                problem.parameters[key] = zeroth[key]
            problem.parameters[f'λ{side}0{i}'] = l
    problem.parameters['λ1_l'] = l1_l
    problem.parameters['λ1_r'] = l1_r

    for j in ['l','r']:
        for i in range(Ns[j]+1): problem.parameters[f'r_{j}{i}'] = boundaries[j][i]
        for i in range(Ns[j]): 
            problem.substitutions[f'r{j}{i}'] = f'r_{j}{i} + (r_{j}{i+1} - r_{j}{i})*s'
            problem.substitutions[f'c{j}{i}'] = f'1/(r_{j}{i+1}-r_{j}{i})'
            problem.substitutions[f'dr{j}{i}(A)'] = f'c{j}{i}*ds(A)'
            problem.substitutions[f'q{j}0{i}'] = f'dr{j}{i}(r{j}{i}*u{j}0{i})/r{j}{i}'
    for side in 'lr':
        for i in range(Ns[side]):
            reps = [('r',f'r{side}{i}')] + [(f'{f}{order}', f'{f}{side}{order}{i}') for f in 'uδqλ' for order in '01']
            eq1 = 'r*(u0*dr(δ1) + dr(δ0)*u1 + dr(u1)) + u1 = 0'
            eq2 = 'u0*dr(u1) + dr(u0)*u1 + dr(δ1) - 2*λ0*λ1/r**3 = 2*(dr(q0) + dr(u0)*dr(δ0))'
            eq3 = 'dr(λ1) = (2*λ0/(r*u0))*(1/r + dr(u0)/u0)'
    #         eq3 = 'dr(λ1) = 0' # no angular dissipation
            problem.add_equation(replace(eq1, reps))
            problem.add_equation(replace(eq2, reps))
            problem.add_equation(replace(eq3, reps))

    # continuity at boundaries
    for side in 'lr':
        for i in range(Ns[side]-1):
            for f in 'uλδ':
                problem.add_bc(f'left({f}{side}1{i+1}) - right({f}{side}1{i}) = 0')
    for side in 'lr':
        i = Ns[side] - 1
        reps = [('r',f'r{side}{i}'),('intp','left')] + [(f'{f}{order}', f'{f}{side}{order}{i}') for f in 'uδqλ' for order in '01']
        problem.add_bc(replace(f'intp(dr(u0)*u1 - (λ0/r**3)*λ1) = intp(dr(dr(u0)) + dr(u0)**2 + 1/r**2)',reps))
        problem.add_bc(replace(f'intp(δ1 - u1) = 0',reps))
    problem.add_bc('left(λl10) = λ1_l')
    problem.add_bc(f'right(λr1{Ns["r"]-1}) = λ1_r')

    solver = problem.build_solver()

    solver.solve()
    first = {var: solver.state[var] for var in problem.variables}
    fields = {**zeroth, **first}
    splines = {}
    for f in 'uδλ':
        for order in [0,1]:
            for side in 'lr':
                subfields = [fields[f'{f}{side}{order}{j}'] for j in range(Ns[side])]
                splines[f'{f}{side}{order}'] = Spline(boundaries[side], subfields)

    return {'l':l, 'g':g, 'u0_func':u0_func, 'rs':rs1, 'fields':fields, 'splines':splines}


def analytic_inner_zeroth(x, u0_r):
    a = np.log(-u0_r)
    c, d = np.cosh(a), np.sinh(a)
    return -c - d*np.tanh(x*d/2)

def sech(x): return np.cosh(x)**(-1)

import mpmath as mp
li2_obj = np.frompyfunc(lambda x: float(mp.polylog(2,x)),1,1)
li2 = lambda y: li2_obj(y).astype(float)

def analytic_inner_first(x, l, g, r_s, u0_r, l1_r):
    a = np.log(-u0_r)
    c, d = np.cosh(a), np.sinh(a)
    u0 = -c - d*np.tanh(x*d/2)
    u11 = - (d**2/2)*sech(d*x/2)**2
    u12 = u0 + x*u11 + 1/c
    c1 = (c/d**2)*(
        2*x +.5*np.exp(-a)*x**2
        -2*x*np.log(np.exp(-d*x-2*a)+1)
        -4*sech(a)*np.log(np.cosh(.5*d*x+a))
        +np.cosh(x*d)/(d**2 * c)
        +(2/d)*li2(-np.exp(-2*a-d*x)))
    c2 = (c/d**2)*(2*np.log(np.cosh(.5*d*x+a)) - c*x)
    gs = -.5*(1/r_s - 2/(r_s-g)**2 + l**2/r_s**3)
    ui = gs*(c1*u11 + c2*u12)
    
    l1 = (l/r_s)*(1/c + np.cosh(a-d*x))*sech(d*x/2)**2
    l1 += l1_r - l1[-1]
    return {'x':x,'u0':u0, 'u11':u11, 'u12':u12, 'u1i':ui, 'gs':gs, 'l1':l1}

def solve_inner_zeroth(boundaries,parameters,mid=0, maxits=50, minpert=1e-13, start_damping=.1, thresh_damping=1):
    u0_r, δ0_l, n = [parameters[a] for a in ['u0_r','δ0_l','n']]
    u0_l = 1/u0_r
    x_l, x_r = boundaries[0], boundaries[-1]
    L = x_r - x_l
    u0_int = u0_l*(mid-x_l) + u0_r*(x_r-mid)

    xbasis = de.Chebyshev('x', n, interval=boundaries)
    domain = de.Domain([xbasis],grid_dtype=np.float64)
    x, = domain.grids()

    problem = de.NLBVP(domain, variables=['u0','q0','δ0','v'])#[f'{f}0{i}' for f in 'uqδ' for i in range(N)])
    problem.meta[:]['x']['dirichlet'] = True

    problem.parameters['L'] = L
    problem.parameters['u0_int'] = u0_int
    problem.parameters['u0_l'] = u0_l
    problem.parameters['u0_r'] = u0_r
    problem.parameters['δ0_l'] = δ0_l
#     problem.parameters['u0_0'] = u_mid
    problem.substitutions['c'] = 'u0_l + u0_r'
    problem.substitutions[f'res_u0'] = '2*dx(q0) + c*q0 - 2*u0*q0'
    problem.substitutions[f'res_δ0'] = 'u0*dx(δ0) + q0'

    problem.add_equation('2*dx(q0) + c*q0 = 2*u0*q0')
    problem.add_equation('dx(u0) - q0 = 0')
    problem.add_equation('dx(δ0) = -q0/u0')
    problem.add_equation('dx(v) - u0 = 0')

    problem.add_bc(' left(u0) = u0_l')
    problem.add_bc(' left(v) = 0')
    problem.add_bc('right(v) = u0_int') # integral constraint
    problem.add_bc(' left(δ0) = δ0_l')

    solver = problem.build_solver()    

    fields = {var: solver.state[var] for var in problem.variables}

    u0, q0, δ0, v = [fields[a] for a in problem.variables]
    u0['g'] = analytic_inner_zeroth(x,u0_r)
    q0['g'] = u0.differentiate('x')['g'].copy()
    δ0['g'] = - np.log(-u0['g'])
    v['g'] = u0.antidifferentiate('x',('left',0))['g']
    res_u0 = solver.evaluator.vars['res_u0']
    res_δ0 = solver.evaluator.vars['res_δ0']


    its = 0
    perturbations = solver.perturbations.data
    perturbations.fill(.1+1e-10)
    pert = np.sum(np.abs(solver.perturbations.data))

    while its < maxits and pert > minpert:
        if pert > thresh_damping: damping = start_damping
        else: damping = .9
        solver.newton_iteration(damping=damping)
        its += 1
        pert = np.sum(np.abs(solver.perturbations.data))
        print(pert)
        if pert > 1e5: break    
    
    return {'domain':domain, 'fields':fields, 'boundaries':boundaries, 'parameters':parameters, 'grid':x}

def solve_inner_first(boundaries, parameters):
    n, u0_r, u0_l, ul01_l, ur01_r, ul10_l, ur10_r, g_s = [parameters[a] for a in ['n', 'u0_r', 'u0_l', 'ul01_l', 'ur01_r', 'ul10_l', 'ur10_r', 'g_s']]

    x_l, x_r = boundaries[0], boundaries[-1]
    L = x_r - x_l

    xbasis = de.Chebyshev('x', n, interval=boundaries)
    domain = de.Domain([xbasis],grid_dtype=np.float64)
    x, = domain.grids()
    u0 = domain.new_field()
    u0['g'] = analytic_inner_zeroth(x, u0_r)

    problem = de.LBVP(domain, variables=['u10', 'u11', 'v1'], ncc_cutoff=1e-16)
    problem.meta[:]['x']['dirichlet'] = True
    problem.parameters['L'] = L
    problem.parameters['c'] = u0_l + u0_r
    problem.parameters['ul01_l'] = ul01_l
    problem.parameters['ur01_r'] = ur01_r
    problem.parameters['ul10_l'] = ul10_l
    problem.parameters['ur10_r'] = ur10_r
    problem.parameters['g_s'] = g_s
    problem.parameters['u0'] = u0
    problem.substitutions[f'res_u1'] = 'dx(u11) - (1/2)*(u0-1/u0+4*dx(u0)/u0)*u11 - (1/2)*(c*dx(u0)/u0)*u10 - g_s'

    problem.add_equation('dx(u11) - (1/2)*(u0-1/u0+4*dx(u0)/u0)*u11 - (1/2)*(c*dx(u0)/u0)*u10 = g_s')
    problem.add_equation('u11 - dx(u10) = 0')
    problem.add_equation('dx(v1) - dx(u0)*u10 = 0')

    problem.add_bc('left(u10) = left(x*ul01_l + ul10_l)')
    problem.add_bc('left(v1) = 0')
    problem.add_bc('right(v1) = 0')

    solver = problem.build_solver()

    fields = {var: solver.state[var] for var in problem.variables}
    solver.solve()
    return {'domain':domain, 'fields':fields, 'boundaries':boundaries, 'parameters':parameters, 'grid':x}

def calc_shift(x, ui, u12, ul01_l, ur01_r, ul10_l, ur10_r):
    print(ul01_l,ur01_r,ul10_l,ur10_r)
    u1i1_l = ul01_l#(ui[1] - ui[0])/(x[1] - x[0])
    u1i1_r = ur01_r#(ui[-1] - ui[-2])/(x[-1] - x[-2])
    u12_l = u12[0]
    u12_r = u12[-1]

    L = np.array([[-u1i1_l, u12_l],
                  [-u1i1_r, u12_r]])
    R = np.array([ul10_l + x[0]*ul01_l - ui[0], 
                  ur10_r + x[-1]*ur01_r - ui[-1]])

    x0, amp = np.linalg.solve(L,R)
    print(x0,amp, (ui[-1] - x0*ur01_r + amp*u12[-1]) - (ur10_r + x[-1]*ur01_r))
    return x0, amp

def split(r_split, f1, f2):
    return lambda r: np.piecewise(r, [r<r_split,r>=r_split], [f1,f2])

def asymptotic_analysis(l, g, n, L=60, rlim=None):
    accretion = Accretion(l, g, rlim=rlim)
    accretion.plot()
    
    out = {}
    out['accretion'] = accretion
    out['boundaries'] = {}
    out['sims'] = {}
    
    # zeroth order outer problem, refined
    initial_0 = {'ul':accretion.splines[0,1], 'ur':accretion.splines[1,1]}
    initial_0['δl'] = lambda r: -np.log(-r*initial_0['ul'](r))
    initial_0['δr'] = lambda r: -np.log(-r*initial_0['ur'](r))
    bl = np.array((1.5*g, accretion.sonics[0], accretion.shocks[1]*1.1))
    br = np.array((1.5*g, accretion.sonics[0], accretion.shocks[1]*0.9, accretion.sonics[-1], accretion.rlim[-1]))
    out['boundaries']['outer-0'] = bs = {'l': bl, 'r':br}
    out['sims']['outer-0'] = solve_outer_zeroth(l, g, n, bs, initial_0, maxits=10)
    
    # shock location analysis
    il, ir = 1, 2
    r_s_1 = find_shock(accretion.shocks[0], 
                      {'l':bs['l'][1:],
                       'r':bs['r'][il:il+2]}, 
                      {'l':out['sims']['outer-0']['fields']['ul1'], 
                       'r':out['sims']['outer-0']['fields'][f'ur{il}']})[0]
    r_s_2 = find_shock(accretion.shocks[1], 
                      {'l':bs['l'][1:],
                       'r':bs['r'][ir:ir+2]}, 
                      {'l':out['sims']['outer-0']['fields']['ul1'], 
                       'r':out['sims']['outer-0']['fields'][f'ur{ir}']})[0]    
    out['shocks'] = {1:r_s_1, 2:r_s_2}
    
    # first order outer asymptotics
    u0_func = {s: out['sims']['outer-0']['splines'][f'u{s}0'] for s in 'lr'}
    r_l, r_r = bs['l'][0], bs['r'][-1]
    u_l, u_r = u0_func['l'](r_l), u0_func['r'](r_r)
    e2 = accretion.sonic_energies[1]
    ρinf = np.exp(e2)
    ρ_l, ρ_r = -1/(r_l*u_l), -1/(r_r*u_r)
    l1_l, l1_r = 2*l*(ρ_l - ρinf), 2*l*(ρ_r - ρinf)
    out['sims']['outer-1'] = solve_outer_first(l, g, n, bs, u0_func, l1_l, l1_r)
    ul0, ur0, ul1, ur1, λl0, λr0, λl1, λr1 = [out['sims']['outer-1']['splines'][f'{f}{s}{i}'] for f in 'uλ' for i in range(2) for s in 'lr']
    # zero/first order inner asymptotics (analytical solution)
    for shock, r_s in out['shocks'].items():
        ur0_r, lr1_r = ur0(r_s), λr1(r_s)
        out['sims'][f'inner-{shock}'] = {sp: (lambda x, name=sp, r_s=r_s,ur0_r=ur0_r, lr1_r=lr1_r: analytic_inner_first(x, l, g, r_s, ur0_r, lr1_r)[name]) for sp in ['u0','u11','u12','u1i','l1']}
        
        # calculate shift
        x = np.array([-L/2,-L/2+1e-5, L/2-1e-5,L/2])
        u1i, u12 = [out['sims'][f'inner-{shock}'][a] for a in ['u1i','u12']]
        ul01_l, ur01_r, ul10_l, ur10_r = ul0.differentiate()(r_s), ur0.differentiate()(r_s), ul1(r_s), ur1(r_s)
        x0, amp = calc_shift(x, u1i(x), u12(x), ul01_l, ur01_r, ul10_l, ur10_r)
        out['sims'][f'inner-{shock}']['x0'], out['sims'][f'inner-{shock}']['amp'] = x0, amp
        out['sims'][f'inner-{shock}']['u1'] = lambda x,u1i=u1i,amp=amp: u1i(x) + amp*u12(x)
        out['sims'][f'inner-{shock}']['l0'] = lambda x: l + 0*x
    return out

def approximations(dic, ϵ, L=60):
    # get spline of each asymptotic order in each region
    splines = {}
    for f in 'uλ':
        for s in 'lr':
            for i in range(2):
                g = 'l' if f=='λ' else f
                splines[f'{g}{s}{i}'] = dic['sims']['outer-1']['splines'][f'{f}{s}{i}']
    for shock, r_s in dic['shocks'].items():
        for f in 'ul':
            for i in range(2):
                splines[f'{f}{i}-{shock}'] = dic['sims'][f'inner-{shock}'][f'{f}{i}']

    out = {'smooth':{}, 'inner':{}, 'outer':{}}
    for f in 'ul':
        out['smooth'][f'{f}0'] = splines[f'{f}r0']

        out['smooth'][f'{f}1'] = lambda r, f=f, ϵ=ϵ: splines[f'{f}r0'](r) + ϵ*splines[f'{f}r1'](r)
    for shock, name in zip([1,2],['inner','outer']):
        x0 = dic['sims'][f'inner-{shock}']['x0']
        r_s = dic['shocks'][shock]
        r_s_shift = r_s + ϵ*x0
        for f in 'ul':
            bs_0 = [splines[f'{f}l0'].boundaries[0],
                    r_s-ϵ*L/2,
                    r_s+ϵ*L/2,
                    splines[f'{f}r0'].boundaries[-1]]
            bs_1 = [splines[f'{f}l0'].boundaries[0],
                    r_s_shift-ϵ*L/2,
                    r_s_shift+ϵ*L/2,
                    splines[f'{f}r0'].boundaries[-1]]
            out[name][f'{f}0'] = FuncSpline(bs_0, [
                splines[f'{f}l0'], 
                lambda r, f=f, r_s=r_s, ϵ=ϵ, sh=shock: splines[f'{f}0-{sh}']((r-r_s)/ϵ), 
                splines[f'{f}r0']])
            out[name][f'{f}1'] = FuncSpline(bs_1, 
                [lambda r, f=f,ϵ=ϵ: splines[f'{f}l0'](r) + ϵ*splines[f'{f}l1'](r), 
                 lambda r, f=f,r_s_shift=r_s_shift,ϵ=ϵ,sh=shock: (lambda x: splines[f'{f}0-{sh}'](x) + ϵ*splines[f'{f}1-{sh}'](x))((r-r_s_shift)/ϵ), 
                 lambda r, f=f,ϵ=ϵ: splines[f'{f}r0'](r) + ϵ*splines[f'{f}r1'](r)])
            
#             lambda r, r_s=r_s, f=f, s=shock, ϵ=ϵ: np.piecewise(r,
#                 [r<r_s-ϵ*L/2, 
#                 (r>=r_s-ϵ*L/2)&(r<r_s+ϵ*L/2), 
#                 r>r_s+ϵ*L/2],                                           
#                 [splines[f'{f}l0'], 
#                 lambda r: splines[f'{f}0-{s}']((r-r_s)/ϵ), 
#                 splines[f'{f}r0']])
#             out[name][f'{f}1'] = lambda r, r_s_shift=r_s_shift, f=f, s=shock, ϵ=ϵ: np.piecewise(r, 
#                 [r<r_s_shift-ϵ*L/2, 
#                 (r>=r_s_shift-ϵ*L/2)&(r<r_s_shift+ϵ*L/2),
#                  r>r_s_shift+ϵ*L/2],
#                 [lambda r: splines[f'{f}l0'](r) + ϵ*splines[f'{f}l1'](r), 
#                  lambda r, r_s_shift=r_s_shift: (lambda x: splines[f'{f}0-{s}'](x) + ϵ*splines[f'{f}1-{s}'](x))((r-r_s_shift)/ϵ), 
#                  lambda r: splines[f'{f}r0'](r) + ϵ*splines[f'{f}r1'](r)])

    out['splines'] = splines
    
    return out

# analytical shock calculations
def newton(ff,dff,x,
           initial_damping=1, 
           maxits=40, 
           xatol=1e-10,
           xrtol=1e-10,
           fatol=1e-10,
           frtol=1e-10,
           f_thresh=1e-1, 
           x_thresh=1e-1, 
           x_symb='x',
           f_symb='f',
           out=False,
           bounds=(-np.inf,np.inf),
           mindamping=1e-6):
    if out: print(out, bounds)
    it = 0
    f, df = ff(x),dff(x)
    dx = -f/df
    while (it < 40 and 
           ((np.abs(dx) > xatol) or
           (np.abs(dx)/(np.abs(x) + 1e-12) > xrtol) or
           (np.abs(f) > fatol))):
#            (np.abs(df*dx*damping)/(np.abs(f) + 1e-12) > frtol))):
        damping = initial_damping
        if out: print(f'{out}, {it}, {x_symb} {x:.4e}, {f_symb} {f:.4e}, d{f_symb} {df:.4e}, Δ{x_symb} {dx:.4e}')
        while ((np.abs(ff(x+damping*dx)) > np.abs(f)) or # iteration is worse
               (np.isnan(ff(x+damping*dx))) or # step outside domain
               (x + damping*dx > bounds[1]) or # step outside boundaries
               (x + damping*dx < bounds[0])):
            if out: print('  ', out, f'|{f_symb}({x_symb}+d*Δ{x_symb})| = {np.abs(ff(x+damping*dx)):.3e}, |{f_symb}({x_symb})| = {np.abs(f):.3e}, damping = {damping:.3e}, {x_symb} + d*Δ{x_symb} = {x + damping*dx:.3e}')
            if damping < mindamping: 
                raise ValueError('The damping has gotten too small')
            damping *= 1/2
        x += dx*damping
        f = ff(x)
        df = dff(x)
        dx = -f/df
        it += 1
    return x

def secant(f,x0, x1,stopdx=1e-10,f_thresh=1e-1,x_thresh=1e-1,out=False,bounds=(-np.inf,np.inf),mindamping=1e-5):
    it = 0
    dx = 1
    while it < 40 and (np.abs(dx) > stopdx):
        damping = 1.
        dx = -f(x1)*(x1-x0)/(f(x1) - f(x0))
        while ((np.abs(f(x1+damping*dx)) > np.abs(f(x1))) or # iteration is worse
               (np.isnan(f(x1+damping*dx))) or # step outside domain
               (x1+damping*dx > bounds[1]) or # step outside boundaries
               (x1 + damping*dx < bounds[0])):
            if out: print(f'    conditions |f(x+d*Δx)| = {np.abs(f(x1+damping*dx)):.3e}, |f(x)| = {np.abs(f(x1)):.3e}, damping = {damping:.3e}, x + d*Δx = {x1 + damping*dx:.3e}')
            if damping < mindamping: raise ValueError('    The damping has gotten too small')
            damping *= 1/2
        x0 = x1
        x1 += dx
        it += 1
        if out: print(f'{it}, x {x1:.4e}, f {f(x1):.4e}, df {f(x1)-f(x0):.4e}, Δx {dx:.4e}, d {damping}')
#         print(it, x0, x1, f(x1))
    return x1

def f(r, u, e, l, g):
    return e - (0.5*(u**2 + (l/r)**2) - 2/(r-g) - np.log(r*np.abs(u)))

def fr(r,u,e,l,g):
    return l**2/r**3 - 2/(r-g)**2 + 1/r

def frr(r,u,e,l,g):
    return -3*l**2/r**4 + 4/(r-g)**3 - 1/r**2

def fu(r,u,e,l,g):
    return -u + 1/u

def fuu(r,u,e,l,g):
    return -1 - 1/u**2

def fe(r,u,e,l,g):
    return 1

def fl(r,u,e,l,g):
    return -l/r**2

def fg(r,u,e,l,g):
    return -2/(r-g)**2

def stability(l, g,out=False):
    dic = {}
#    a = Accretion(l,g)
    coeff_list = [1, -2*(1+g), l**2 + g**2, -2*l**2*g, (l*g)**2]
    roots = np.roots(coeff_list).astype(complex)
    sonics = np.sort([root.real for root in roots if root.real > g and np.abs(root.imag) < 1e-15])    
    dic['l'] = l
    dic['g'] = g
    dic['r_2'] = newton(lambda r: fr(r, 0, 0, l, g), 
                        lambda r: frr(r, 0, 0, l, g), sonics[2])
    dic['e_2'] = newton(lambda e: f(dic['r_2'], -1, e, l, g), 
                        lambda e: fe(dic['r_2'], -1, e, l, g), 0)
    dic['r_1'] = newton(lambda r: fr(r, 0, 0, l, g), 
                        lambda r: frr(r, 0, 0, l, g), sonics[0])
    dic['e_1'] = newton(lambda e: f(dic['r_1'], -1, e, l, g), 
                        lambda e: fe(dic['r_1'], -1, e, l, g), 0)
    dic['u_2'] = lambda r: newton(lambda u: f(r, u, dic['e_2'], l, g), 
                                  lambda u: fu(r, u, dic['e_2'], l, g), 
                                  -1.1 if r < dic['r_2'] else -.1)
    dic['u_2_proj'] = lambda r: 1/dic['u_2'](r)
    dic['u_1'] = lambda r, initial_damping=.5, f_thresh=1e-2,x_thresh=1e-2: newton(lambda u: f(r, u, dic['e_1'], l, g), 
                                  lambda u: fu(r, u, dic['e_1'], l, g), 
                                  -1.1 if r < dic['r_1'] else -.1,initial_damping=initial_damping,f_thresh=f_thresh,x_thresh=x_thresh)
    dic['r_0'] = newton(lambda r: fr(r, 0, 0, l, g),
                        lambda r: frr(r, 0, 0, l, g), sonics[1],)
    dic['r_1_crit'] = newton(lambda r: f(r, -1, dic['e_1'], l, g),
                             lambda r: fr(r, -1, dic['e_1'], l, g),
                             0.5*(dic['r_0']+dic['r_2']), initial_damping=1,out='r_1_crit'*out, bounds=[dic['r_0'],dic['r_2']])
    def dr_ugap(r):
        u2 = dic['u_2'](r)
        u1 = dic['u_1'](r)
        diff = u1 - 1/u2
        dru2 = -fr(r, u2, dic['e_2'], l, g)/fu(r, u2, dic['e_2'], l, g)
        dru1 = -fr(r, u1, dic['e_1'], l, g)/fu(r, u1, dic['e_1'], l, g)
        grad = dru1 + dru2/u2**2
        return grad

    dic['r_s2'] = newton(lambda r: dic['u_1'](r) - 1/dic['u_2'](r),
                         dr_ugap, dic['r_1_crit']*.98, 
                         out='r_s2'*out, 
                         bounds=[dic['r_0'],dic['r_1_crit']], 
                         initial_damping=1)

    dic['r_s1'] = newton(lambda r: dic['u_1'](r) - 1/dic['u_2'](r),
                         dr_ugap, 
                         dic['r_1']*1.02,
                         out='r_s1'*out,
                         bounds=[dic['r_1'],dic['r_0']])
    dic['up_s2'] = dic['u_2'](dic['r_s2'])
    dic['um_s2'] = dic['u_1'](dic['r_s2'])
    dic['up_s1'] = dic['u_2'](dic['r_s1'])
    dic['um_s1'] = dic['u_1'](dic['r_s1'])
    dic['a'] = np.log(-dic['up_s2'])
    r_s1, r_s2 = dic['r_s1'], dic['r_s2']
    dic['g1'] =  -(1/2) * (1/r_s1 - 2/(r_s1 - g)**2 + l**2/r_s1**3)
    dic['g2'] =  -(1/2) * (1/r_s2 - 2/(r_s2 - g)**2 + l**2/r_s2**3)
    dic['λ_s1'] = -dic['g1']/np.cosh(dic['a'])
    dic['λ_s2'] = -dic['g2']/np.cosh(dic['a'])
    return dic

# using dedalus
def solvability(a,n=512,L=30):
    dic = {}
    dic['a'] = a
    dic['L'] = L
    n = 512
    xbasis = de.Chebyshev('x', n, interval=(-L/2, L/2))
    domain = de.Domain([xbasis], grid_dtype=np.float64)
    x, = domain.grids()
    dic['x'] = x
    u0_r2 = -np.exp(a)
    c = - 2*np.cosh(a)
    d = - 2*np.sinh(a)
    dic['u0'] = u0f = domain.new_field()
    u0f['g'] = -np.cosh(a) - np.sinh(a) * np.tanh(np.sinh(a)*x/2)
    dic['u11'] = u11 = domain.new_field()
    u11['g'] = - .5* np.sinh(a)**2 * np.cosh(np.sinh(a)*x/2)**(-2)
    dic['u12'] = u12 = domain.new_field()
    u12['g'] = -np.cosh(a) + 1/np.cosh(a) -  np.sinh(a) * np.tanh(np.sinh(a)*x/2) - .5 * x * np.sinh(a)**2 * np.cosh(np.sinh(a)*x/2)**(-2)
    dic['c1'] = c1 = domain.new_field()

    c1['g'] = (2*c/d**2)*(-8*np.sinh(d*x/2)/d**3 
                          + 4*x/d**2 
                          + (4/d)*li2(-np.exp(d*x/2 - 2*a)) 
                          + 2*x*np.log(np.exp(d*x/2 - 2*a)+1) 
                          + x**2/(2*u0_r2)
                          - (4/(c*d**3))*(4*np.sinh(a + d*x/2) + x*np.sinh(4*a))
                          - (8/c)*np.log(np.cosh(a - d*x/4)))
    dic['c2'] = c2 = domain.new_field()
    c2['g'] = (2/(np.sinh(a)*np.tanh(a)))*np.log(2*np.cosh(np.sinh(a)*x/2 + a)) -x/np.tanh(a)**2
    dic['ui'] = ui = domain.new_field()
    ui['g'] = c1['g']*u11['g'] + c2['g']*u12['g']
    dic['u1'] = dic['ui']# - 1000*dic['u11'] - 10*dic['u12']).evaluate() # these parts don't matter

    u0 = dic['u0']
    u0x = u0.differentiate('x')
    u1 = dic['u1']
    u1x = u1.differentiate('x')
    dic['integrand'] = integrand = ( (-1/(2*d*c)) * (u0x/u0**3) * (c*(u0**2 - 1)*u1 + 2*(3*u0**2 - 1)*u1x) ).evaluate()
    dic['solvability'] = integrand.integrate('x')['g'][0]
    
    return dic    