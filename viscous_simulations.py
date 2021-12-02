import numpy as np
import matplotlib.pyplot as plt
import accretion_code as ac
import file_tools as flt
from scipy.interpolate import interp1d
import dedalus.public as de

# inviscid approximation
l = 1.1
g = .0925
n = 128
L = 60
asymptotics = ac.asymptotic_analysis(l, g, n, L=L)

# full viscous simulations
a = asymptotics['accretion']
l, g = a.l, a.g
ϵs = np.array([1e-4,5e-5,2e-5,1e-5])
n = 128

print(ϵs)

outputs = {'smooth':{}, 'inner':{}, 'outer':{}}

for profile in outputs:
    shock = {'inner':1,'outer':2}.get(profile)
    if shock:
        r_s = asymptotics['shocks'][shock]
        x0 = asymptotics['sims'][f'inner-{shock}']['x0']

    for i, ϵ in enumerate(ϵs):
        print(i)
        approxs = ac.approximations(asymptotics,ϵ,L=L)
        if shock:
            bs = (1.5*g,
                  a.saddles[0],#*1.01, 
                  r_s + ϵ*(x0-L), 
                  r_s + ϵ*(x0+L), 
                  a.saddles[1],#*.99)
                  3)
        else: bs = (1.5*g,a.saddles[0],a.saddles[1],3)
        initials = {'u':approxs[profile]['u1'],'l':approxs[profile]['l1']}
        u_l, u_r = initials['u'](np.array([bs[0]+1e-10,bs[-1]-1e-10]))
        l_l, l_r = initials['l'](np.array([bs[0]+1e-10,bs[-1]-1e-10]))
        params = {'γ':g, 'ε':ϵ, 'u_r':u_r, 'u_l':u_l, 'l_r':l_r, 'l_l':l_l,}
        outputs[profile][i] = ac.solve_steady_compound(params, n, bs,initials,
                                              l_visc=True,
                                              start_damping=.1,thresh_damping=1,
                                              maxits=50,
                                              flatten_inner=False,maxpert=1e7)

# empirical asymptotics
def richardson(s, ϵ, toarray=False):
    l = [(s[j+1]/ϵ[j+1] - s[j]/ϵ[j])/(1/ϵ[j+1] - 1/ϵ[j]) for j in range(len(s) - 1)]
    if toarray: l = np.array(l)
    return l

from scipy.interpolate import UnivariateSpline
def find_shift(x, y, y0):
    spl = UnivariateSpline(x, y - y0, s=0)
    return spl.roots()[0]

M = len(ϵs)

arrs = {}
for profile in ['inner','outer']:
    for sim, out in outputs[profile].items():
        shock = {'inner':1,'outer':2}.get(profile)
        r = out['grid'][2]
        r_s = asymptotics['shocks'][shock]
        x0 = asymptotics['sims'][f'inner-{shock}']['x0']
        ϵ = out['params']['ε']
        arrs[profile,sim,'x'] = x = (r - (r_s + ϵ*x0))/ϵ
        for f in 'ulmn':
            arrs[profile,sim,f] = out['fields'][f'{f}2']['g']
rarrs = {profile:{f: np.stack([arrs[profile,sim,f] for sim in outputs[profile]]) for f in 'ulmn'} for profile in ['inner','outer']}

for profile in rarrs:
    shock = {'inner':1,'outer':2}.get(profile)
    r_s = asymptotics['shocks'][shock]
    u0r = asymptotics['sims']['outer-1']['splines']['ur0'](r_s)
    u_mid = (u0r + 1/u0r)/2
    # calculate shock location and richardson extrapolation for empirical convergence
    rarrs[profile]['x0'] = x0 = asymptotics['sims'][f'inner-{shock}']['x0'] + 0*ϵs
    rarrs[profile]['xs'] = xs = x0 + np.array([find_shift(x, u, u_mid) for u in rarrs[profile]['u']])
    rarrs[profile]['x1'] = x1 = richardson((xs - x0)/ϵs, ϵs)
    rarrs[profile]['x2'] = x2 = richardson((xs[1:] - x0[1:] - ϵs[1:]*x1)/ϵs[1:]**2,ϵs[1:])
    # calculate shifted profiles
    rarrs[profile]['x_shift'] = {}
    rarrs[profile]['u_shift'] = us = 0*rarrs[profile]['u']
    rarrs[profile]['l_shift'] = ls = 0*rarrs[profile]['l']
    for sim, out in outputs[profile].items():
        ϵ = ϵs[sim]
        rarrs[profile]['x_shift'][sim] = x_shift = arrs[profile,sim,'x'] + ϵ*np.mean(x1)
        r_shift = r_s + ϵ*(x_shift + x0[sim])
        rarrs[profile]['u_shift'][sim] = out['splines']['u'](r_shift)
        rarrs[profile]['l_shift'][sim] = out['splines']['l'](r_shift)
    # Richardson extrapolation of these velocities\    
    rarrs[profile]['u0'] = u0 = richardson((us), ϵs)#ac.analytic_inner_zeroth(x_shift,u0r)[None,:] + 0*ϵs[:,None]
    rarrs[profile]['l0'] = l0 = richardson((ls), ϵs)#l + 0*u0
    rarrs[profile]['u1'] = u1 = richardson((us[1:] - u0)/ϵs[1:,None], ϵs[1:])
    rarrs[profile]['l1'] = l1 = richardson((ls[1:] - l0)/ϵs[1:,None], ϵs[1:])

# Richardson extrapolation of outer velocities
rarrs['left'], rarrs['right'] = {}, {}
rarrs['left']['r'] = rl = np.linspace(*outputs['outer'][3]['boundaries'][0:3:2], 500)
rarrs['right']['r'] = rr = np.linspace(*outputs['smooth'][0]['boundaries'][0::3], 500)
for side in ['left','right']:
    r = rarrs[side]['r']
    rarrs[side]['us'] = us = 0*(rarrs[side]['r'][None,:] + ϵs[:,None])
    rarrs[side]['ls'] = ls = 0*(rarrs[side]['r'][None,:] + ϵs[:,None])
    profile = 'outer' if side=='left' else 'smooth'
    for sim, ϵ in enumerate(ϵs):
        rarrs[side]['us'][sim] = outputs[profile][sim]['splines']['u'](r)
        rarrs[side]['ls'][sim] = outputs[profile][sim]['splines']['l'](r)
    s = side[0]
    rarrs[side]['u0'] = u0 = richardson(us,ϵs,toarray=True)# = asymptotics['sims']['outer-1']['splines'][f'u{s}0'](r)[None,:] + 0*ϵs[:,None]
    rarrs[side]['l0'] = l0 = richardson(ls,ϵs,toarray=True)# = asymptotics['sims']['outer-1']['splines'][f'λ{s}0'](r)[None,:] + 0*ϵs[:,None]
    rarrs[side]['u1'] = richardson((us[1:] - u0)/ϵs[1:,None], ϵs[1:],toarray=True)
    rarrs[side]['l1'] = richardson((ls[1:] - l0)/ϵs[1:,None], ϵs[1:],toarray=True)    

def ends(x):
    return np.array([x[0],-x[-1]])

# outer profiles
# fig, ax = plt.subplots(3,3, figsize=(6,4), gridspec_kw={'width_ratios':[2,1,1],'wspace':.1})
# gridspec inside gridspec
from matplotlib import gridspec
fig = plt.figure()
gs0 = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25)
gs00 = gs0[0].subgridspec(3, 1, hspace=.2)
gs01 = gs0[1].subgridspec(3, 2, hspace=.2, wspace=0.15)

ax = {}
ax[0,0] = fig.add_subplot(gs00[0])
ax[1,0] = fig.add_subplot(gs00[1])
ax[2,0] = fig.add_subplot(gs00[2])
for i in range(3):
    for j in range(1,3):
        ax[i,j] = fig.add_subplot(gs01[i,j-1])

# plot shocks through outer solutions
for shock in [1,2]:
    r_s = asymptotics['shocks'][shock]
    ul0, ur0 = [asymptotics['sims']['outer-1']['splines'][f] for f in ['ul0','ur0']]
    ul1, ur1 = [asymptotics['sims']['outer-1']['splines'][f] for f in ['ul1','ur1']]
    ll1, lr1 = [asymptotics['sims']['outer-1']['splines'][f] for f in ['λl1','λr1']]
    style = 'k'+'--'*(shock==1)
    label = ('Inner' if shock==1 else 'Outer') + ' shock'
    ax[0,0].plot(np.log(2*[r_s]), [ul0(r_s), ur0(r_s)], style, label=label)
    ax[1,0].plot(np.log(2*[r_s]), [ul1(r_s), ur1(r_s)], style)
    ax[2,0].plot(np.log(2*[r_s]), [ll1(r_s), lr1(r_s)], style)        
        
# plot outer smooth solutions through inner and outer sonic points        
rl = rarrs['left']['r']
rr = rarrs['right']['r']
ax[0,0].plot(np.log(rl), asymptotics['sims']['outer-1']['splines']['ul0'](rl),'C0',label='Inner sonic',zorder=10)
ax[0,0].plot(np.log(rr), asymptotics['sims']['outer-1']['splines']['ur0'](rr),'C2',label='Outer sonic',zorder=10)
ax[1,0].plot(np.log(rl), asymptotics['sims']['outer-1']['splines']['ul1'](rl),'C0',label='$u^-_1$',zorder=10)
ax[1,0].plot(np.log(rr), asymptotics['sims']['outer-1']['splines']['ur1'](rr),'C2',label='$u^+_1$',zorder=10)
ax[2,0].plot(np.log(rl), asymptotics['sims']['outer-1']['splines']['λl1'](rl),'C0',label='$\ell^-_1$',zorder=10)
ax[2,0].plot(np.log(rr), asymptotics['sims']['outer-1']['splines']['λr1'](rr),'C2',label='$\ell^+_1$',zorder=10)

def format10(x):
    l = np.log10(x)
    n = np.floor(l)
    c = 10**(l-n)
    return c, n

# plot asymptotic convergence of empirical sims on outer scale    
colors = ['C8','C1','C3','C4']
for i in range(M):
    if i < M-1:
        coeff, power = format10(ϵs[i+1])
        ax[0,0].plot(np.log(rarrs['left']['r']), rarrs['left']['u0'][i], color=colors[i+1], 
                   label=f'$ϵ = {coeff:.0f}\\times 10^{{{power:.0f}}}$')
        ax[0,0].plot(np.log(rarrs['right']['r']), rarrs['right']['u0'][i], color=colors[i+1],)
    if i < M-2:
        coeff, power = format10(ϵs[i+2])
        ax[1,0].plot(np.log(rarrs['left']['r']), rarrs['left']['u1'][i], color=f'C{i+1}', 
                   label=f'$u_1^-$ at $ϵ = {coeff:.0f}\\times 10^{{{power:.0f}}}$')
        ax[1,0].plot(np.log(rarrs['right']['r']), rarrs['right']['u1'][i], color=f'C{i+3}',
                   label=f'$u_1^+$ at $ϵ = {coeff:.0f}\\times 10^{{{power:.0f}}}$')
        ax[2,0].plot(np.log(rarrs['left']['r']), rarrs['left']['l1'][i], color=f'C{i+1}', 
                   label=f'$ℓ_1^-$ at $ϵ = {coeff:.0f}\\times 10^{{{power:.0f}}}$')
        ax[2,0].plot(np.log(rarrs['right']['r']), rarrs['right']['l1'][i], color=f'C{i+3}',
                   label=f'$ℓ_1^+$ at $ϵ = {coeff:.0f}\\times 10^{{{power:.0f}}}$')

# plot shock scales
x = arrs['inner',0,'x']
for i, (side, sim) in enumerate(zip(['inner','outer'],['inner-1','inner-2'])):
    x0, amp, ui, u12, u0, l1 = [asymptotics['sims'][sim][f] for f in ['x0','amp','u1i','u12','u0','l1']]
    r_s = asymptotics['shocks'][int(sim[-1])]
    ul0,ur0,ul1,ur1 = [asymptotics['sims']['outer-1']['splines'][f'u{s}{o}'] for o in range(2) for s in 'lr']
    ll1, lr1 = [asymptotics['sims']['outer-1']['splines'][_] for _ in ['λl1','λr1']]
    xlong = np.linspace(*sorted([2*x0,0]),10)
    ax[0,1+i].plot(xlong, ul0(r_s) + 0*xlong, 'C0')
    ax[0,1+i].plot(xlong, ur0(r_s) + 0*xlong, 'C2')
    shock_style = 'k' + '--'*(i==0)
    ax[0,1+i].plot(x + x0, u0(x), shock_style,zorder=10)
    for j in range(M-1):
        ax[0,1+i].plot(x + x0, rarrs[side]['u0'][j], color=colors[j+1], )
        if j < M-2:
            ax[1,1+i].plot(x + x0, rarrs[side]['u1'][j], color=colors[j+2], )
            ax[2,1+i].plot(x + x0, rarrs[side]['l1'][j], color=colors[j+2], )
#                    label=f'$u_0^-$ at $ϵ = {coeff:.0f}\\times 10^{{{power:.0f}}}$')
    ax[1,1+i].plot(xlong, (ul1(r_s) + ul0.differentiate()(r_s)*xlong), 'C0')
    ax[1,1+i].plot(xlong, (ur1(r_s) + ur0.differentiate()(r_s)*xlong), 'C2')
    ax[1,1+i].plot(x + x0, (ui(x) + amp*u12(x)), shock_style, zorder=10)
    ax[2,1+i].plot(xlong, ll1(r_s) + 0*xlong, 'C0')
    ax[2,1+i].plot(xlong, lr1(r_s) + 0*xlong, 'C2')
    ax[2,1+i].plot(x + x0, l1(x), shock_style, zorder=10)
        
# set axis labels etc.
ax[0,0].set(yticks=[-2,-1,0],ylim=[-2,0],xticks=[],)
ax[0,0].annotate('Reduced problem',(.14,1.1),xycoords='axes fraction', fontsize=12)
ax[0,0].set_ylabel('$u^\\pm_0$',labelpad=8)
gs0.figure.suptitle(f'              Asymptotic behaviour of isothermal shocks $(\ell = {a.l:.1f}, r_h = {a.g:.4f})$',y=1.01)
ax[1,0].set(yticks=[0,100,200],xticks=[],
            ylabel='$u^\\pm_1$')
ax[2,0].set(yticks=[0,5,10],xticks=[-2,-1,0,1],xlabel='$\log r$')
ax[2,0].set_ylabel('$ℓ^\\pm_1$',labelpad=8)

ax[0,1].set(yticks=[0,-1,-2],xlim=[-150,0],xticks=[],)
#             title='Inner shock')
ax[0,1].annotate('Layer problem',(0.43,1.1),xycoords='axes fraction',fontsize=12)
ax[0,1].set_ylabel('$u_0$',labelpad=0)
ax[0,2].set(yticks=[],ylim=[-2,0],xlim=[0,600],xticks=[],)
#             title='Outer shock')
ax[1,1].set(yticks=[-400,0,400],ylim=[-400,400],xlim=[-150,0],xticks=[])
ax[1,1].set_ylabel('$u_1$',labelpad=-14)
ax[1,2].set(yticks=[],ylim=[-400,400],xlim=[0,600],xticks=[])

ax[2,1].set(yticks=[0,5,10],ylim=[0,10],xlim=[-150,0],
            xlabel='$x$')
ax[2,1].set_ylabel('$ℓ_1$',labelpad=0)
ax[2,2].set(yticks=[],ylim=[0,10],xlim=[0,600],
            xlabel='$x$')

ax[0,0].legend(bbox_to_anchor=(2.95,1.2),frameon=False,fontsize=10,handlelength=1)
# ax[1,0].legend(frameon=False,fontsize=8,labelspacing=0,handlelength=1)
# ax[2,0].legend(frameon=False,fontsize=8,labelspacing=0,handlelength=1)

plt.savefig('figures/black-hole-outer-asymptotics-summary.pdf',bbox_inches='tight')

fig, ax = plt.subplots(figsize=(2.5,4))
for shock in ['inner','outer']:
    style = 'C0' if shock=='inner' else 'C2' #'k' + '--'*(shock=='inner')
    label = ('Inner' if shock=='inner' else 'Outer') + ' shock'
    ax.loglog(ϵs, np.abs(rarrs[shock]['x0'] - rarrs[shock]['xs']), style, label=label)
ax.loglog(ϵs,1e5*ϵs,'k--',label='$1\\times 10^5 ϵ$')
ax.loglog(ϵs,3e5*ϵs,'k',label='$3\\times 10^5 ϵ$')
ax.legend(frameon=False)
# ax.grid(True)
ax.set(ylim=[.8,120],
       xlabel='$ϵ$',
       ylabel='$|x_{mid} - x_0|$',
       title='Asymptotic convergence of \nshock location $x_{mid}$')
plt.savefig('figures/black-hole-shock-location-asymptotics.pdf',bbox_inches='tight')


# time evolution simulations

save_dir = 'data/evolution'
n = 256
dt = 1e-5
save_step = .01
save_max = 500
timestepper = 'SBDF2'
max_sim_time = 1
max_wall_time = 10*60
max_iteration = 100000
dealias = 3/2
print_freq = 1000
params = {}
for profile in ['inner','outer']:
    params[profile] = {}
    for sim in outputs[profile]:
        ϵ = ϵs[sim]
        bs = outputs[profile][sim]['boundaries'][2:4]
        sim_name = f'black-hole-unsteady-perturbation-{profile}-{ϵ:.0e}'
        A = ϵ/10
        out = outputs[profile][sim]
        initial = {}
        initial['u0'] = out['splines']['u']
        initial['l0'] = out['splines']['l']
        params[profile][sim] = {
            'l':a.l, 'g':a.g, 'ϵ':ϵ, 'r_1':bs[0], 'r_2':bs[1], 'A':A,
            'n':n, 'dt':dt, 'initial':initial, 
            'sim_name':sim_name, 'save_dir':save_dir, 'print_freq':print_freq,
            'max_sim_time':max_sim_time, 'timestepper':timestepper, 'max_wall_time':max_wall_time,
            'max_iteration':max_iteration, 'save_step':save_step, 'save_max':save_max, 'dealias':dealias}
        print(sim_name)
        ac.solve_initial_perturbation(params[profile][sim])
        
# analysis
ϵs = np.array([1e-4,5e-5,2e-5,1e-5])

save_names = [f'analysis-black-hole-unsteady-perturbation-{profile}-{ϵ:.0e}' for profile in ['inner','outer'] for ϵ in ϵs]

files = {i: ac.get_saves(save_dir, name) for i, name in enumerate(save_names)}

sims = {}
for i in files:
    file = files[i][-1]
    sims[i] = {}
    sims[i]['file'] = file
    sims[i]['t'],sims[i]['r'] = flt.load_data(file,'sim_time','r/1.0',group='scales')
    for key in flt.get_keys(file,group='tasks'):
        sims[i][key], = flt.load_data(file,key,group='tasks')

l = 1.1
g = .0925
dic = ac.stability(l,g)
λ1 = dic['λ_s1']
λ2 = dic['λ_s2']
t0 = sims[0]['t']

# growth rate
upeaks = {sim: np.abs(sims[sim]['u1']).max(axis=1) for sim in sims}
λs = {sim: np.polyfit(sims[sim]['t'], np.log(upeaks[sim]),1)[0] for sim in sims}

import matplotlib
cmap = matplotlib.cm.get_cmap('Blues')
colors = [cmap(i) for i in np.linspace(.3,.9,len(t0))]

fig, ax = plt.subplots(1,2, figsize=(8,4),gridspec_kw={'width_ratios':[2,1],'wspace':.4})

colors = ['C0','C2','C1','C3']
for sim in range(4):
    c, nn = format10(ϵs[sim])
    ax[0].plot(sims[sim]['t'], np.log(upeaks[sim]) - np.log(upeaks[sim][0]),'--',
            color=colors[sim],
            label=f'$ϵ = {c:.0f} \\times 10^{{{nn:.0f}}}$')
for sim in range(4,8):
    ax[0].plot(sims[sim]['t'], np.log(upeaks[sim]) - np.log(upeaks[sim][0]),color=colors[sim-4])
ax[0].plot(t0, dic['λ_s1']*t0, 'k--',label='$\exp(λ_0 t)$')
ax[0].plot(t0, dic['λ_s2']*t0, 'k')
ax[0].legend(frameon=False)

# ax.annotate(f'$\lambda_{{ac.full}} = {λ00:.4f}$',(.4,-7.2))
# ax.annotate(f'$\lambda_{{pert}} = {λ01:.4f}$',(.4,-7.25))
ax[0].annotate(f'Inner $\lambda_{{0}} = + {dic["λ_s1"]:.4f}$',(.5,.25))
ax[0].annotate(f'Outer $\lambda_{{0}} = {dic["λ_s2"]:.4f}$',(.5,-.05))
ax[0].annotate('$(a)$',(.03,.05),xycoords='axes fraction',fontsize=12)
ax[0].set(xlabel='Time $t$',
          ylabel='Amplitude growth \n $\log \\frac{\max \\tilde{u}(t)}{\max \\tilde{u}(0)}$',
          title='Inner (dashed) and Outer (solid) shocks')
fig.suptitle(f'Perturbation growth/decay rates for $ℓ = {l:.1f}, r_h = {g:.4f}$',y=1.01)

ax[1].loglog((ϵs), ([λs[sim] - λ1 for sim in range(4)]),'.',color='C0',label='Inner shock')
ax[1].loglog((ϵs), (np.abs([λs[sim] - λ2 for sim in range(4,8)])),'.',color='C2',label='Outer shock')
ax[1].loglog((ϵs), (1e3*ϵs),'k--',label='$1000 \ ϵ$')
ax[1].loglog((ϵs), (50*ϵs),'k',label='$50 \ ϵ$')
ax[1].legend(frameon=False,loc='upper left')
ax[1].set(ylim=[3e-4,8e-1],
       xlabel='$ϵ$', 
       ylabel='Growth rate error\n$|λ_{emp} - λ_0|$',
       title='Growth/decay rate error')
ax[1].annotate('$(b)$',(.8,.05),xycoords='axes fraction',fontsize=12)
plt.savefig('figures/black-hole-shock-growth-rate-errors.pdf',bbox_inches='tight')
