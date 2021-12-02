import numpy as np
import matplotlib.pyplot as plt
import accretion_code as ac
import file_tools as flt
from scipy.interpolate import interp1d
import dedalus.public as de
import file_tools as flt

def mag(x): return np.log10(np.abs(x)+1e-16)
import mpmath as mp
li2_obj = np.frompyfunc(lambda x: float(mp.polylog(2,x)),1,1)
li2 = lambda y: li2_obj(y).astype(float)

# stability diagrams
filename = 'regime-curves.h5'
curves = {}
for curve in flt.get_keys(filename):
    curves[curve] = {'l':flt.load_data(filename,'l',group=curve)[0],
                     'g':flt.load_data(filename,'g',group=curve)[0]}
    
curve_splines = {curve: interp1d(curves[curve]['l'], curves[curve]['g']) for curve in curves}
fracbasis = de.Chebyshev('s',12,interval=(0,1))
fracs = fracbasis.grid()

c0 = curve_splines['equal-shock']
c1 = curve_splines['tangent-shock']
ls = np.linspace(0.2, 1.3, 20)
gs0 = c0(ls)
gs1 = c1(ls)
gs = gs0[:,None] + (gs1 - gs0)[:,None]*fracs[None,:]

# shock location and magnitude

dics = {}
ur0_rs = {}
for i in range(len(ls)):
    for j in range(gs.shape[1]):
        print(i,j)
        li = ls[i]
        gij = gs[i,j]
        dics[i,j] = ac.stability(li,gij,out=False)
        
# growth rate calculation

i, j = 1,1
dic = dics[i, j]
λ1s = np.zeros(gs.shape)
λ2s = np.zeros(gs.shape)
avals = np.zeros(gs.shape)
for i in range(gs.shape[0]):
    for j in range(gs.shape[1]):
        l, g = ls[i], gs[i,j]
        λ1s[i,j] = dics[i,j]['λ_s1']
        λ2s[i,j] = dics[i,j]['λ_s2']

from scipy.interpolate import RectBivariateSpline
λ1_spline = RectBivariateSpline(ls, fracs, λ1s)
λ2_spline = RectBivariateSpline(ls, fracs, λ2s)

ls_high = np.linspace(.2,1.3,100)
fracs_high =  np.linspace(.005,.995,100)
λ1s_high = λ1_spline(ls_high, fracs_high)
λ2s_high = λ2_spline(ls_high, fracs_high)

import matplotlib.colors as colors

frac = np.linspace(0,1,gs.shape[1],endpoint=False)
fig, ax = plt.subplots(1,2,gridspec_kw={'wspace':0},figsize=(6,2.5))
p1 = ax[0].pcolormesh(ls_high, fracs_high, λ1s_high.T,
                      norm=colors.SymLogNorm(linthresh=0.1, linscale=1.,
                                              vmin=-2000, vmax=2000, base=10),
                      shading='nearest',cmap='RdBu_r')
ax[0].contour(ls_high, fracs_high, np.log10(np.abs(λ1s_high.T)),[-1,0,1,2,3],colors='k',linestyles='-')
p2 = ax[1].pcolormesh(ls_high, fracs_high, λ2s_high.T,
                      norm=colors.SymLogNorm(linthresh=0.1, linscale=1.,
                                              vmin=-2000, vmax=2000, base=10),
                      shading='nearest',cmap='RdBu_r')
ax[1].contour(ls_high, fracs_high, np.log10(np.abs(λ2s_high.T)),[-1,0,1,2,3],colors='k',linestyles='-')
ax[0].set(xlabel='$\ell$',title='Inner shock')
ax[0].set_ylabel('$\\frac{r_h - r_{h,1}(\ell)}{r_{h,2}(\ell) - r_{h,1}(\ell)}$',fontsize=15)
ax[1].set(xlabel='$\ell$',yticks=[],title='Outer shock')
fig.suptitle('Asymptotic growth/decay rate $\lambda(\ell, r_h)$',y=1.08)
plt.colorbar(p2,ax=ax)
plt.savefig('figures/black-hole-shock-stability-regimes.png',bbox_inches='tight',dpi=500)


# finite eps regimes

def discriminant(l, g):
    return 32 * l**6 * g**3 - 32 * l**8 * g**3 - 432 * l**4 * g**4 \
           + 560* l**6 * g**4 - 1440 * l**4 * g**5 - 96* l**6*g**5 \
           - 1184*l**4*g**6 - 96*l**4*g**7 - 16*l**2*g**8 - 32*l**2*g**9

def sonic_points(l, g):
    coeff_list = [1, -2*(1+g), l**2 + g**2, -2*l**2*g, (l*g)**2]
    return np.roots(coeff_list).astype(complex).real[:-1][::-1]

def sonic_energy(l, g, rs):
    return ac.newton(lambda e: ac.f(rs,-1,e,l,g), lambda e:ac.fe(rs,-1,e,l,g), 0)


def log_min_u1_estimate(l, g, r0, e):
    return .5*(l/r0)**2 - 2/(r0-g) - np.log(r0) - e

def min_u1(l, g, r0, e1, u1):
    return ac.newton(lambda u: ac.f(r0, u, e1, l, g), 
                     lambda u: ac.fu(r0, u, e1, l, g),
                     u1,
                     bounds=[-1, -0],)

def max_u2(l, g, r0, e2):
    return ac.newton(lambda u: ac.f(r0, u, e2, l, g), 
                     lambda u: ac.fu(r0, u, e2, l, g),
                     -1.1,
                     bounds=[-np.inf, -1],)

def r_crit_u1(l, g, e1, r0, r2):
    return ac.newton(lambda r: ac.f(r, -1, e1, l, g),
                     lambda r: ac.fr(r, -1, e1, l, g),
                     .5*(r0+r2),
                     bounds=[r0,r2])

def find_shock(l, g, e1, e2, r0, rcrit, out=False):
    u10, u20 = -.9, -1.1
    u1f = lambda r: ac.newton(lambda u: ac.f(r, u, e1, l, g),
                              lambda u: ac.fu(r, u, e1, l, g),
                              u10,
                              bounds=[-1, 0], out='    u1' if out else None, x_symb='u1')
    u2f = lambda r: ac.newton(lambda u: ac.f(r, u, e2, l, g),
                              lambda u: ac.fu(r, u, e2, l, g),
                              u20,
                              bounds=[-np.inf, -1], out='    u2' if out else None, x_symb='u2')
    u10 = u1f(rcrit*.99)
    u20 = u2f(rcrit*.99)
    def dr_gap(r):
        nonlocal u10
        nonlocal u20
        u1, u2 = u10, u20 = u1f(r), u2f(r)
        diff = u1 - 1/u2
        dru1 = -ac.fr(r, u1, e1, l, g)/ac.fu(r, u1, e1, l, g)
        dru2 = -ac.fr(r, u2, e2, l, g)/ac.fu(r, u2, e2, l, g)
        grad = dru1 + dru2/u2**2
        return grad
    
    return ac.newton(lambda r: u1f(r) - 1/u2f(r),
                     dr_gap,
                     rcrit*(1-1e-5),
                     bounds=[r0, rcrit], out=out, x_symb='r', f_symb='Δu')

def u0_vec(r, e1, l, g, out=False):
    u10 = -.99
    us = np.zeros(r.shape)
    def u1f(r):
        nonlocal u10
        return ac.newton(lambda u: ac.f(r, u, e1, l, g),
                         lambda u: ac.fu(r, u, e1, l, g),
                         u10,
                         bounds=[-1, 0], out='    u1' if out else None, x_symb='u1',
                         xatol=1e-14, fatol=1e-14, xrtol=1e-14)
    
    for i, ri in enumerate(r):
        us[i] = u10 = u1f(ri)
    return us

def u1_r0(l, g, r1, r0, rs2, e1, e2, nr=128, out=False):
    rbasis = de.Chebyshev('r',nr,interval=(r1, rs2))
    domain = de.Domain([rbasis], grid_dtype=np.float64)
    r, = domain.grids()
    u0s = u0_vec(r, e1, l, g)
    u0, l1, rf = domain.new_fields(3)
    rf['g'] = r
    u0['g'] = u0s
    ρinf = np.exp(e2)
    ρ0s = -1/(r*u0s)
    l1['g'] = 2*l*(ρ0s - ρinf)

    problem = de.LBVP(domain, variables=['u1'])
    problem.parameters['l'] = l
    problem.parameters['g'] = g
    problem.parameters['l1'] = l1
    problem.parameters['u0'] = u0
    problem.parameters['e1'] = e1
    problem.substitutions['res_u0'] = '(u0**2 + (l/r)**2)/2 - 2/(r-g) - log(-r*u0) - e1'
    problem.substitutions['res_u1'] = 'dr((u0-1/u0)*u1)/2 - (dr(dr(u0)) - dr(u0)**2/u0 - u0/r**2 + l*l1/r**3)'
#     problem.substitutions['res'] = '((u0-1/u0)*dr(u1) + (1 + 1/u0**2)*dr(u0)*u1)/2 - (dr(dr(u0)) - dr(u0)**2/u0 - u0/r**2 + l*l1/r**3)'
    problem.substitutions['rhs'] = 'dr(dr(u0)) - dr(u0)**2/u0 - u0/r**2 + l*l1/r**3'
    problem.add_equation('dr((u0-1/u0)*u1)/2 = dr(dr(u0)) - dr(u0)**2/u0 - u0/r**2 + l*l1/r**3')
#     problem.add_equation('((u0-1/u0)*dr(u1) + (1 + 1/u0**2)*dr(u0)*u1)/2 = dr(dr(u0)) - dr(u0)**2/u0 - u0/r**2 + l*l1/r**3')
    problem.add_bc('left(dr(u0))*left(u1) = left(dr(dr(u0)) + dr(u0)**2 + 1/r**2 + l*l1/r**3)')

    solver = problem.build_solver()
    solver.solve()
    u1 = solver.state['u1']
    ratio = u1.interpolate(r='right')['g'][0]/u0.interpolate(r='right')['g'][0]
    if out: 
        rhs = solver.evaluator.vars['rhs'].evaluate()
        res_u0 = solver.evaluator.vars['res_u0'].evaluate()
        res_u1 = solver.evaluator.vars['res_u1'].evaluate()
        return {'r':rf, 'u0':u0, 'l1':l1, 'ρ0':ρ0s, 'u1':u1, 'rhs':rhs, 'res_u0':res_u0, 'res_u1':res_u1, 'ratio':ratio}
    else: return u1.interpolate(r='right')['g'][0]/u0.interpolate(r='right')['g'][0]

from scipy.optimize import brentq

def find_equal_energy(g):
    
    ls = np.linspace(0, .3)
    discs = discriminant(ls,g)
    leftmost = ls[np.where(discs < 0)[0][-1]]
    
    def energy_gap(l):
        r1, r0, r2 = sonic_points(l, g)
        e1, e2 = sonic_energy(l, g, r1), sonic_energy(l, g, r2)
        return e1 - e2
    
    return brentq(energy_gap, leftmost, .3)

def check_crossings(l, g,out=False, nr=128):
    dic = {}
    dic['disc'] = disc = discriminant(l, g)
    if disc < 0: return dic
    dic['r1'],dic['r0'],dic['r2'] = r1, r0, r2 = sonic_points(l, g)
    dic['e1'] = e1 = sonic_energy(l, g, r1)
    dic['e2'] = e2 = sonic_energy(l, g, r2)
    dic['e0'] = e0 = sonic_energy(l, g, r0)
    if e1 > e2: return dic
    dic['log_u1_min_0'] = log_u1_min_0 = log_min_u1_estimate(l, g, r0, e1)
    if log_u1_min_0 > -20: dic['u1_min'] = u1_min = min_u1(l, g, r0, e1, -np.exp(log_u1_min_0))
    else: dic['u1_min'] = u1_min = -np.exp(log_u1_min_0)
    dic['u2_max'] = u2_max = max_u2(l, g, r0, e2)
    dic['r_crit_u1'] = rcrit = r_crit_u1(l, g, e1, r0, r2)
    dic['crossing'] = u1_min - 1/u2_max
    if dic['crossing'] > 0:
        dic['rs2'] = rs2 = find_shock(l, g, e1, e2, r0, rcrit)
        try: dic['u1_r0'] = u1 = u1_r0(l, g, r1, r0, rs2, e1, e2, out=out, nr=nr)
        except Exception: pass
            
    return dic

ls = np.linspace(0,.3,501)[1:]
gs = np.linspace(0,5e-3,21)[1:]

# a = ac.Accretion(ls[11],gs[0])
# a.plot()

Δs = discriminant(ls[:,None], gs[None,:])
# g = r_h
# Δs = discriminant(ls, g)

dics = {}
for j, g in enumerate(gs):
    for i, l in enumerate(ls):
        if Δs[i,j] > 0:
#             print(i, j, f'{l:.3f}')
            dics[i,j] = check_crossings(l, g, out=True)

for key, dic in dics.items():
    if dic.get('crossing',-1) > 0:
        if 'u1_r0' in dic:
            print(key, dic['u1_r0']['ratio'])

zeros = np.zeros((len(ls), len(gs)))
shocks = zeros.copy()
ratios = zeros.copy()


for i, l in enumerate(ls):
    for j, g in enumerate(gs):
        if dics.get((i,j)):
            shocks[i,j] = dics[i,j].get('crossing',np.nan)
            if dics[i,j].get('crossing',-1) > 0 and ('u1_r0' in dics[i,j]):
                ratio = dics[i,j]['u1_r0']['ratio']
                if ratio > 0: ratio = np.nan
                ratios[i, j] = ratio

ls_dic = {}
ls_dic['three-sonics'] = [ls[np.where(Δs[:,j] > 0)[0][0]] for j in range(len(gs))]
ls_dic['tangent'] = [ls[np.where((shocks[:,j]>0) & np.isfinite(shocks[:,j]))[0][0]] for j in range(len(gs))]
for mag in range(0, 30, 5):
    ls_dic[f'min-u1-{mag}'] = [ls[(np.where(np.log(-ratios[:,j]) > mag)[0][0])] for j in range(len(gs))]
ls_dic['collision'] = [find_equal_energy(g) for g in gs]

g0 = 1e-4
l0 = brentq(lambda l: discriminant(l, 1e-4), 0.01, .1)
l1 = find_equal_energy(1e-4)


gs2 = np.linspace(0,5e-3,21)
discs2 = discriminant(ls[:,None], gs2[None,:])

fig, ax = plt.subplots(figsize=(6,4))
ax.plot([l0]+ls_dic['three-sonics'], [g0]+list(gs), 'C4', label='Three sonic points',zorder=11)
ax.plot(ls_dic['tangent'], gs, 'C0', label='Shock tangency',zorder=10)
ax.plot(ls_dic['min-u1-15'], gs, 'C2', label='$ε = 10^{-15}$ breakdown',zorder=9)
ax.plot(ls_dic['min-u1-20'], gs, 'C1', label='$ε = 10^{-20}$ breakdown',zorder=8)
ax.plot(ls_dic['min-u1-25'], gs, 'C3', label='$ε = 10^{-25}$ breakdown',zorder=7)
ax.plot([l1]+ls_dic['collision'], [g0]+list(gs), 'C5', label='Shock-sonic collision',zorder=6)
ax.contourf(ls, gs2, discs2.T, np.arange(-2e-10,2e-10,1e-11), cmap='RdBu_r')
ax.set_facecolor('k')
ax.legend(frameon=False)
ax.set(xlim=[0.05,0.28],ylim=[0.000,0.005],
       xlabel='Angular momentum $\ell$',
       ylabel='Horizon scale $r_h$',
       title='Narrow shock regimes for small $r_h$')
plt.savefig('figures/black-hole-small-rh-asymptotic-breakdown-regimes.png',dpi=400)
