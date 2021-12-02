import numpy as np
import matplotlib.pyplot as plt
import accretion_code as ac

def mag(x): return np.log10(np.abs(x)+1e-16)
import mpmath as mp
li2_obj = np.frompyfunc(lambda x: float(mp.polylog(2,x)),1,1)
li2 = lambda y: li2_obj(y).astype(float)


l = 1.1
gs = [.1,
      0.09656129775676174 - 1e-5,
      0.094,
      0.09301766051725471,
      0.0925,
      0.09173120388494756,
      0.08,
      0.048682552176966334 + 1e-4,
      0.04]

accs = []
for g in gs:
    print(g)
    accs.append(ac.Accretion(l,g))

for a in accs:
    a.plot()
    
fig, ax = plt.subplots(3,3,
                       gridspec_kw={'hspace':0,'wspace':0},
                       figsize=(7,4))
axf = ax.T.flatten()
accs[0].colors[0,1,1] = 'C2'
accs[0].colors[0,1,-1] = 'C2'
accs[0].colors[0,-1,1] = 'C4'
accs[0].colors[0,-1,-1] = 'C4'
for i, a in enumerate(accs):
    axf[i].contourf(np.log(a.rs), (-a.us)**(1/2), a.energies.T, a.contours,cmap='RdBu_r')
    axf[i].set_facecolor('k')
    for p in a.paths:
        axf[i].plot(np.log(a.paths[p]['y'][0]),
                    (-a.paths[p]['y'][1])**(1/2),
                    color=a.colors[p])
    for j, s in enumerate(a.shocks):
        if not (j==1 and i==3):
            axf[i].plot(np.log([s,s]), (-np.array([a.splines[0,1](s),a.splines[1,1](s)]))**(1/2),'k')
    axf[i].set(ylim=[0.55,1.45])
    axf[i].annotate(f'$r_h = {gs[i]:.5f}$',(.03,.32),xycoords='axes fraction',rotation='vertical',c='w',fontsize=8)
for i in range(3): axf[i].set(xlim=np.log([1.1*gs[2],10]),yticks=[.75,1.,1.25])
for i in range(3,6): axf[i].set(xlim=np.log([1.1*gs[5],10]),yticks=[])    
for i in range(6,9): axf[i].set(xlim=np.log([np.exp(-3.9),10]),yticks=[])    
for i in range(9): axf[i].annotate(f'$({"abcdefghi"[i]})$',(.01,.06),xycoords='axes fraction',c='w')
    
ax[1,0].set(ylabel='$|u|^{1/2}$')
ax[2,1].set(xlabel='$\log r$')
ax[2,0].set(xticks=[-2,-1,0,1,2])
ax[2,1].set(xticks=[-2,-1,0,1,2])
ax[2,2].set(xticks=[-3,-2,-1,0,1,2])    
ax[0,1].set(title='Black hole accretion diagrams for $ℓ=1.1$')
plt.savefig('figures/black-hole-accretion-diagrams.png',bbox_inches='tight',dpi=500)    