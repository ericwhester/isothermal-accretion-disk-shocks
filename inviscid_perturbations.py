import numpy as np
import matplotlib.pyplot as plt
import accretion_code as ac
import file_tools as flt
import dedalus.public as de
from dedalus.tools import post

d = de.operators.differentiate
integ = de.operators.integrate


# inviscid perturbations simulation
l = 1.1
g = .0925

a = ac.Accretion(l, g)

rs = a.sonics[-1]
energy = a.energy(rs, -1)

f = ac.f
fu = ac.fu
u_func = lambda r: ac.newton(lambda u: f(r, u, energy, l, g),
                             lambda u: fu(r, u, energy, l, g),
                             -1.1 if r < rs else -.1, 
                             bounds=[-np.inf,-1] if r < rs else [-1,0])
u_ufunc = np.vectorize(u_func)

boundaries = bs = (g+1e-6,g+1e-2, rs, 200)

nr = 512
sbasis = de.Chebyshev('s',nr, interval=(0,1))
sdomain = de.Domain([sbasis],grid_dtype=np.float64)
s, = sdomain.grids()
N = len(bs)-1
rarrs = [bs[i] + (bs[i+1]-bs[i])*s for i in range(N)]
drs = [bs[i+1]-bs[i] for i in range(N)]
u0s = sdomain.new_fields(N)
for i in range(N): u0s[i]['g'] = u_ufunc(rarrs[i])

dru01 = (u0s[1].differentiate('s')/drs[1]).evaluate().interpolate(s='right')['g'][0]
dru02 = (u0s[2].differentiate('s')/drs[2]).evaluate().interpolate(s='left')['g'][0]

problem = de.IVP(sdomain, variables=[f'{f}{i}' for i in range(N) for f in 'uva' ])
for i in range(N+1): problem.parameters[f'r_{i}'] = bs[i]
for i in range(N):
    problem.substitutions[f'r{i}'] = f'r_{i} + s*(r_{i+1} - r_{i})'
    problem.substitutions[f'dr{i}(A)'] = f'ds(A)/(r_{i+1} - r_{i})'
    problem.parameters[f'u0{i}'] = u0s[i]
    problem.parameters['l'] = l
    problem.parameters['g'] = g
    problem.substitutions[f'f{i}'] = f'(u0{i}-1/u0{i})'
    problem.substitutions[f'e{i}'] = f'(1/2)*(f{i}*a{i}**2 - u0{i}*v{i}**2)'
    problem.substitutions[f'integr{i}(A)'] = f'integ(A)*(r_{i+1} - r_{i})'
    problem.substitutions[f'E{i}'] = f'integr{i}(e{i})'
    problem.substitutions[f'decay{i}'] = f'2*integr{i}((dr{i}(u0{i})/u0{i})*a{i}**2)'
    problem.substitutions[f'boundaries{i}'] = f'right(-f{i}*u0{i}*(v{i}+a{i})*a{i}) - left(-f{i}*u0{i}*(v{i}+a{i})*a{i})'

for i in range(N):
    problem.add_equation(f'a{i} - dt(u{i}) = 0')
    problem.add_equation(f'v{i} - dr{i}((u0{i}-1/u0{i})*u{i}) = 0')
    problem.add_equation(f'dt(a{i}) + dr{i}(u0{i}*(2*a{i} + v{i})) = 0')

problem.add_equation('right(u2) = 0')
problem.add_equation('right(v1) - left(v2) = 0')
problem.add_equation('right(v1 - 2*dr1(u01)*u1) = 0')
problem.add_equation('right(u1) - left(u2) = 0')
problem.add_equation('right(v0) - left(v1) = 0')
problem.add_equation('right(u0) - left(u1) = 0')

solver = problem.build_solver(de.timesteppers.RK443)

fields = {_: solver.state[_] for _ in problem.variables}
fields['u2']['g'] = np.exp(-((rarrs[2]-30)/5)**2)
fields['v2']['g'] = d((u0s[2]-1/u0s[2])*fields['u2'],'s')['g']/drs[2]

savedir = 'data'
simname = "analysis-BH-smooth-pert-coupled"
# analysis = solver.evaluator.add_file_handler(f'{savedir}/{simname}', sim_dt=1)#iter = freq_output)
# analysis.add_system(solver.state, layout='g')
# for task in ['e','E','decay','boundaries']:
#     for i in range(N):
#         analysis.add_task(f'{task}{i}')

# dt = .02
# for i in range(4000): 
#     solver.step(dt)
#     if solver.iteration % 100 == 0: 
#         print(solver.iteration, 
#               f't {solver.sim_time:.1f}',
#               f'|u1| max {np.abs(fields["u1"]["g"]).max():.3e}',
#               f'|u2| max {np.abs(fields["u2"]["g"]).max():.3e}')
# solver.step(dt)

# analysis
post.merge_analysis(f'{savedir}/{simname}')
filename = f'{savedir}/{simname}/{simname}_s1.h5'

ts, = flt.load_data(filename, 'sim_time', group='scales')
Us = flt.load_data(filename,'u0','u1','u2',group='tasks')
es = flt.load_data(filename,'e0','e1','e2',group='tasks')
Es = flt.load_data(filename,'E0','E1','E2',group='tasks')
Vs = flt.load_data(filename,'v0','v1','v2',group='tasks')
As = flt.load_data(filename,'a0','a1','a2',group='tasks')
Ds = flt.load_data(filename,'decay0','decay1','decay2',group='tasks')
Bs = flt.load_data(filename,'boundaries0','boundaries1','boundaries2',group='tasks')

from matplotlib import cm
from matplotlib.collections import LineCollection

c0 = 0.1
c1 = 1
colors = [cm.inferno_r(c0+ (c1-c0)*frac) for frac in ts/ts.max()]

# outer profiles
# fig, ax = plt.subplots(3,3, figsize=(6,4), gridspec_kw={'width_ratios':[2,1,1],'wspace':.1})
# gridspec inside gridspec
from matplotlib import gridspec
fig = plt.figure(figsize=(6,4))
gs0 = gridspec.GridSpec(2, 1, figure=fig, hspace=0.1, height_ratios=[3,2])
gs00 = gs0[0].subgridspec(1, 2, hspace=0, wspace=0.0, width_ratios=[1,2])
gs01 = gs0[1].subgridspec(1, 2, hspace=0, wspace=0.4, width_ratios=[1,2])

ax = {}
ax[0,0] = fig.add_subplot(gs00[0])
ax[0,1] = fig.add_subplot(gs00[1])
ax[1,0] = fig.add_subplot(gs01[0])
ax[1,1] = fig.add_subplot(gs01[1])
ax[0,1].set(yticks=[])

step = 5
for i in range(N):
    for j in range(0,len(ts),5):
        for k in range(2):
            ax[0,k].plot(rarrs[i],Us[i][j],color=colors[j],label=f'$t = {ts[j]:.0f}$' if k == 1 and i == 2 else '')
#            ax[1,k].plot(rarrs[i],es[i][j],color=colors[j],linewidth=1)

Esuper = Es[0]+Es[1]
Esub = Es[2]

pointssuper = np.array([ts,Esuper]).transpose().reshape(-1,1,2)
segssuper = np.concatenate([pointssuper[:-1],pointssuper[1:]],axis=1)
lcsuper = LineCollection(segssuper, cmap=plt.get_cmap('inferno_r'))
lcsuper.set_array(c0+(c1-c0)*ts/ts[-1]) # color the segments by our parameter

pointssub = np.array([ts,Esub]).transpose().reshape(-1,1,2)
segssub = np.concatenate([pointssub[:-1],pointssub[1:]],axis=1)
lcsub = LineCollection(segssub, cmap=plt.get_cmap('inferno_r'))
lcsub.set_array(c0+(c1-c0)*ts/ts[-1]) # color the segments by our parameter

ax[1,0].add_collection(lcsuper)
ax[1,1].add_collection(lcsub)
ax[0,0].plot([rs,rs],[-.5,1.1],'k--',linewidth=1)
# ax[1,0].plot([rs,rs],[-.5,1.1],'k--',linewidth=1)
ax[0,1].legend(frameon=False,bbox_to_anchor=(1.05,1.05),loc='upper left',fontsize=9)

ax[0,0].set(xlim=[0,2.5],ylim=[-.5,1.1],xticks=[])
ax[0,0].set_ylabel('Velocity perturbation\n$\\tilde{u}$',labelpad=0)
ax[0,1].set(xlim=[0,150],ylim=[-.5,1.1],xticks=[])
# ax[1,0].set(xlim=[0,2.5],ylim=[0,.1],xlabel='Radius $r$',ylabel='Local energy\n$e$')
# ax[1,1].set(xlim=[0,150],ylim=[0,.2],xlabel='Radius $r$')
ax[1,0].set(xlim=[0,ts[-1]],ylim=[0,.1],yticks=[0,.05,.1],xlabel='Time $t$',ylabel='Total energy\n$E_1, r< r_*$')
ax[1,1].set(xlim=[0,ts[-1]],ylim=[0,1.5],xlabel='Time $t$',ylabel='Total energy\n$E_2, r> r_*$')
fig.suptitle('Evolution of linear perturbation of smooth accretion flow\n$\ell = 1.1, r_h = 0.0925$')
plt.savefig('figures/black-hole-linear-perturbation-smooth-evolution.pdf',bbox_inches='tight')



# normal mode plots

boundaries = bs = (g+1e-10,g+1e-8,g+1e-6,g+1e-4,g+1e-2, rs-1e-2,rs)

nr = 256
sbasis = de.Chebyshev('s',nr, interval=(0,1))
sdomain = de.Domain([sbasis],grid_dtype=np.float64)
s, = sdomain.grids()
N = len(bs)-1
rarrs = [bs[i] + (bs[i+1]-bs[i])*s for i in range(N)]
drs = [bs[i+1]-bs[i] for i in range(N)]
u0s = sdomain.new_fields(N)
for i in range(N): u0s[i]['g'] = u_ufunc(rarrs[i])

dru01 = (u0s[1].differentiate('s')/drs[1]).evaluate().interpolate(s='right')['g'][0]

def compound_spectrum(λ):
    problem = de.LBVP(sdomain, variables=[f'{f}{i}' for i in range(N) for f in 'uvw' ])
    problem.parameters['λ'] = λ
    problem.substitutions[f'dt(A)'] = 'λ*A'
    problem.parameters['l'] = l
    problem.parameters['g'] = g
    for i in range(N+1): problem.parameters[f'r_{i}'] = bs[i]
    for i in range(N):
        problem.substitutions[f'r{i}'] = f'r_{i} + s*(r_{i+1} - r_{i})'
        problem.substitutions[f'dr{i}(A)'] = f'ds(A)/(r_{i+1} - r_{i})'
        problem.parameters[f'u0{i}'] = u0s[i]
        problem.substitutions[f'a{i}'] = f'λ*u{i}'
        problem.substitutions[f'f{i}'] = f'(u0{i}-1/u0{i})'
        problem.substitutions[f'e{i}'] = f'(1/2)*(f{i}*a{i}**2 - u0{i}*v{i}**2)'
        problem.substitutions[f'integr{i}(A)'] = f'integ(A)*(r_{i+1} - r_{i})'
        problem.substitutions[f'E{i}'] = f'integr{i}(e{i})'
        problem.substitutions[f'decay{i}'] = f'2*integr{i}((dr{i}(u0{i})/u0{i})*a{i}**2)'
        problem.substitutions[f'boundaries{i}'] = f'right(-f{i}*u0{i}*(v{i}+a{i})*a{i}) - left(-f{i}*u0{i}*(v{i}+a{i})*a{i})'
        problem.substitutions[f'res{i}'] = f'dt(a{i}) + dr{i}(u0{i}*(2*a{i} + v{i}))' 

    for i in range(N):
        problem.add_equation(f'v{i} - dr{i}((u0{i}-1/u0{i})*u{i}) = 0')
        problem.add_equation(f'dt(a{i}) + dr{i}(u0{i}*(2*a{i} + v{i})) = 0')
        problem.add_equation(f'dr{i}(w{i}) - u{i} = 0')

    # problem.add_equation('right(u2) = 0')
    problem.add_equation('left(w0) = 0')
    for i in range(N-1):
        problem.add_equation(f'right(w{i}) - left(w{i+1}) = 0')
        problem.add_equation(f'right(v{i}) - left(v{i+1}) = 0')
        problem.add_equation(f'right(u{i}) - left(u{i+1}) = 0')
    problem.add_equation(f'right(v{N-1} - 2*dr{N-1}(u0{N-1})*u{N-1}) = 0')
    problem.add_equation(f'right(w{N-1}) = 1')
    # problem.add_equation(f'right(u{N-1}) = 0')

    solver = problem.build_solver()

    fields = {_: solver.state[_] for _ in problem.variables}
    solver.solve()
    res = {res: solver.evaluator.vars[res].evaluate() for res in [f'res{i}' for i in range(N)]}
    return {**fields,**res,**{'λ':λ}}

λs = np.linspace(-2,2,21)
dics = {}
for i, λi in enumerate(λs):
    dics[i] = compound_spectrum(λi)
    print(i, λi)

colors = ['C5','C7','C0','C9','C2','C8','C1','C3','C6','C4']
colors = colors[::-1] + ['k'] + colors
ordering = list(range(0,11)) + list(range(20,10,-1))

fig, ax = plt.subplots(figsize=(6,4))
for i in ordering:
    λ = dics[i]['λ']
    for k in range(N):
        ax.plot(rarrs[k], dics[i][f'u{k}']['g'], colors[i], 
            label=f'$λ = {dics[i]["λ"]:.1f}$' if k == 0 else None,
            linewidth=2 if λ==0 else 1,linestyle='-' if λ <= 0 else '--')
ax.legend(frameon=False,ncol=2,loc='upper right')
ax.set(xlim=[g,rs],ylim=[0,5.5],xlabel='Radius $r$',ylabel='Perturbation $\\tilde u$',
       title='Normal modes for smooth supersonic perturbations\n$\ell = 1.1, r_h = 0.0925$')
plt.savefig('figures/black-hole-smooth-pertubation-normal-modes.pdf',bbox_inches='tight')