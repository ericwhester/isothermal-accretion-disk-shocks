import numpy as np
import matplotlib.pyplot as plt
import accretion_code as ac
import file_tools as flt

def mag(x): return np.log10(np.abs(x)+1e-16)
import mpmath as mp
li2_obj = np.frompyfunc(lambda x: float(mp.polylog(2,x)),1,1)
li2 = lambda y: li2_obj(y).astype(float)

"""
Script to create regime diagrams
"""

# calculate boundaries of regime diagram

curves = {}

l, g = .6, .025
r10 = 2*g
r00 = 4*g
r20 = 2
e10 = 0
e20 = 0
u10 = -.1
u20 = -1.5

rf = lambda l, g, r0: ac.newton(lambda r: ac.fr(r,0,0,l,g), lambda r:ac.frr(r,0,0,l,g), r0)
ef = lambda l, g, r0, e0: ac.newton(lambda e: ac.f(r0,-1,e,l,g), lambda e:ac.fe(r0,-1,e,l,g), e0)
uf = lambda l, g, r0, e0, u0: ac.newton(lambda u: ac.f(r0, u, e0, l, g), lambda u: ac.fu(r0, u, e0, l, g), u0)
def diff(l, g, r00=r00, r10=r10, r20=r20, e10=0, e20=0, u10=u10, u20=u20):
    r0 = rf(l, g,r00)
    r1 = rf(l, g,r10)
    r2 = rf(l, g,r20)
    e1 = ef(l, g, r1, e10)
    e2 = ef(l, g, r2, e20)
    u1 = uf(l, g, r0, e1, u10)
    u2 = uf(l, g, r0, e2, u20)
    return -(u1 - 1/u2)

# initial points on curve

ts = {}
g0 = .025
ts[0] = {}
ts[0]['l'] = l# = .6
ts[0]['g0'] = ac.secant(lambda g: diff(l,g), g,g+1e-3)
ts[0]['r0'] = rf(l,g0, r00)
ts[0]['r1'] = rf(l,g0, r10)
ts[0]['r2'] = rf(l,g0, r20)
ts[0]['e1'] = ef(l,g0, ts[0]['r1'], e10)
ts[0]['e2'] = ef(l,g0, ts[0]['r2'], e20)
ts[0]['u1'] = uf(l,g0, ts[0]['r0'], ts[0]['e1'], u10)
ts[0]['u2'] = uf(l,g0, ts[0]['r0'], ts[0]['e2'], u20)
ts[1] = {}
g0 = ts[0]['g0']
ts[1]['l'] = l +1e-3# = .601
ts[1]['g0'] = ac.secant(lambda g: diff(l,g), .99*g0,g0)
ts[1]['r0'] = rf(l,g0, r00)
ts[1]['r1'] = rf(l,g0, r10)
ts[1]['r2'] = rf(l,g0, r20)
ts[1]['e1'] = ef(l,g0, ts[1]['r1'], e10)
ts[1]['e2'] = ef(l,g0, ts[1]['r2'], e20)
ts[1]['u1'] = uf(l,g0, ts[1]['r0'], ts[1]['e1'], u10)
ts[1]['u2'] = uf(l,g0, ts[1]['r0'], ts[1]['e2'], u20)

# shoot down
dl = -.001
l += dl
i = 1
passed = True
while l > 0 and i < 100000 and passed:
    ts[-i] = {}
    r00, r10, r20, e10, e20, u10, u20 = [ts[-(i-1)][a] for a in ['r0','r1','r2','e1','e2','u1','u2']]
    g1, g0 = ts[-(i-1)]['g0'], ts[-(i-2)]['g0']
    passed = False
    while dl < -1e-5:
        try:
            l = ts[-(i-1)]['l'] + dl
            g = ac.secant(lambda g: diff(l, g, r00, r10, r20, e10, e20, u10, u20),g0,g1)
            if np.isnan(g) or np.abs(diff(l, g, r00, r10, r20, e10, e20, u10, u20)) > 1e-10: raise ValueError('ac.secant iteration did not converge')
            ts[-i]['l'] = l
            ts[-i]['g0'] = g
            passed = True
            break
        except ValueError:
            dl *= 0.9
    
    if i % 10 == 0: print(i, dl, l, g)
    ts[-i]['r0'] = rf(l,g, r00)
    ts[-i]['r1'] = rf(l,g, r10)
    ts[-i]['r2'] = rf(l,g, r20)
    ts[-i]['e1'] = ef(l,g, ts[-i]['r1'], e10)
    ts[-i]['e2'] = ef(l,g, ts[-i]['r2'], e20)
    ts[-i]['u1'] = uf(l,g, ts[-i]['r0'], ts[-i]['e1'], u10)
    ts[-i]['u2'] = uf(l,g, ts[-i]['r0'], ts[-i]['e2'], u20)
    i += 1
    
# shoot up
dl = .001
l = .6 + dl
i = 2
passed = True
while l < 2 and i < 10000 and passed:
    ts[i] = {}
    r00, r10, r20, e10, e20, u10, u20 = [ts[(i-1)][a] for a in ['r0','r1','r2','e1','e2','u1','u2']]
    g1, g0 = ts[(i-1)]['g0'], ts[(i-2)]['g0']
    passed = False
    while dl > 1e-5:
        try:
            l = ts[(i-1)]['l'] + dl
            g = ac.secant(lambda g: diff(l, g, r00, r10, r20, e10, e20, u10, u20),g0,g1)
            if np.isnan(g) or np.abs(diff(l, g, r00, r10, r20, e10, e20, u10, u20)) > 1e-10: raise ValueError('ac.secant iteration did not converge')
            ts[i]['l'] = l
            ts[i]['g0'] = g
            passed = True
            break
        except ValueError:
            dl *= 0.5
    
    if i % 10 == 0: print(i, dl, l, g)
    ts[i]['r0'] = rf(l,g, r00)
    ts[i]['r1'] = rf(l,g, r10)
    ts[i]['r2'] = rf(l,g, r20)
    ts[i]['e1'] = ef(l,g, ts[i]['r1'], e10)
    ts[i]['e2'] = ef(l,g, ts[i]['r2'], e20)
    ts[i]['u1'] = uf(l,g, ts[i]['r0'], ts[i]['e1'], u10)
    ts[i]['u2'] = uf(l,g, ts[i]['r0'], ts[i]['e2'], u20)
    i += 1    
    
order, ls, gs = np.array([(i, ts[i]['l'], ts[i].get('g0',0)) for i in ts if (np.isfinite(ts[i].get('g0',np.nan)))]).T

ordering = np.argsort(order)
ls, gs = ls[ordering], gs[ordering]

curves['tangent-shock'] = {'l':ls, 'g':gs}    

# equal energy sonic points
l, g = .6, .025
r10 = 2*g
r20 = 2
e10 = 0
e20 = 0

rf = lambda l, g, r0: ac.newton(lambda r: ac.fr(r,0,0,l,g), lambda r:ac.frr(r,0,0,l,g), r0, mindamping=1e-10)
ef = lambda l, g, r0, e0: ac.newton(lambda e: ac.f(r0,-1,e,l,g), lambda e:ac.fe(r0,-1,e,l,g), e0, mindamping=1e-10)

def diff_e(l, g, r10=r10, r20=r20, e10=0, e20=0):
    r1 = rf(l, g,r10)
    r2 = rf(l, g,r20)
    e1 = ef(l, g, r1, e10)
    e2 = ef(l, g, r2, e20)
    return e1 - e2

ts = {}
ts[0] = {}
ts[0]['l'] = l = .6
ts[0]['g0'] = g0 = ac.secant(lambda g: diff_e(l,g), .024,.025)
ts[0]['r1'] = rf(l,g0, r10)
ts[0]['r2'] = rf(l,g0, r20)
ts[0]['e1'] = ef(l,g0, ts[0]['r1'], e10)
ts[0]['e2'] = ef(l,g0, ts[0]['r2'], e20)

ts[1] = {}
g0 = ts[0]['g0']
ts[1]['l'] = l = .601
ts[1]['g0'] = g0 = ac.secant(lambda g: diff_e(l,g), g0,.99*g0)
ts[1]['r0'] = rf(l,g0, r00)
ts[1]['r1'] = rf(l,g0, r10)
ts[1]['r2'] = rf(l,g0, r20)
ts[1]['e1'] = ef(l,g0, ts[1]['r1'], e10)
ts[1]['e2'] = ef(l,g0, ts[1]['r2'], e20)

# shoot to lower l
dl = -.001
l += dl
i = 1
passed = True

while l > 0 and i < 10000 and passed:
    ts[-i] = {}
    r10, r20, e10, e20 = [ts[-(i-1)][a] for a in ['r1','r2','e1','e2']]
    g1, g0 = ts[-(i-1)]['g0'], ts[-(i-2)]['g0']
    passed = False
    while dl < -1e-6:
        try:
            l = ts[-(i-1)]['l'] + dl
            g = ac.secant(lambda g: diff_e(l, g, r10, r20, e10, e20), g0, g1, out=False)#l<.28)

    #             if np.isnan(g) or np.abs(diff_e(l, g, r10, r20, e10, e20)) > 1e-10: raise ValueError('ac.secant iteration did not converge')
            ts[-i]['l'] = l
            ts[-i]['g0'] = g
            passed = True
            break
        except ValueError:
            print(dl)
            dl *= 0.5
    
    if i % 10 == 0: print(i, dl, l, g)
    ts[-i]['r0'] = rf(l,g, r00)
    ts[-i]['r1'] = rf(l,g, r10)
    ts[-i]['r2'] = rf(l,g, r20)
    ts[-i]['e1'] = ef(l,g, ts[-i]['r1'], e10)
    ts[-i]['e2'] = ef(l,g, ts[-i]['r2'], e20)
    i += 1
    
# shoot up
dl = .001
l = .6 + dl
i = 2
passed = True
while l < 2 and i < 10000 and passed:
    ts[i] = {}
    r10, r20, e10, e20 = [ts[(i-1)][a] for a in ['r1','r2','e1','e2']]
    g1, g0 = ts[(i-1)]['g0'], ts[(i-2)]['g0']
    passed = False
    while dl > 1e-5:
        try:
            l = ts[(i-1)]['l'] + dl
            g = ac.secant(lambda g: diff_e(l, g, r10, r20, e10, e20),g0,g1)
            if np.isnan(g) or np.abs(diff_e(l, g, r10, r20, e10, e20)) > 1e-10: raise ValueError('ac.secant iteration did not converge')
            ts[i]['l'] = l
            ts[i]['g0'] = g
            passed = True
            break
        except ValueError:
            dl *= 0.5
    
    if i % 10 == 0: print(i, dl, l, g)
    ts[i]['r1'] = rf(l,g, r10)
    ts[i]['r2'] = rf(l,g, r20)
    ts[i]['e1'] = ef(l,g, ts[i]['r1'], e10)
    ts[i]['e2'] = ef(l,g, ts[i]['r2'], e20)
    i += 1    
    
order, ls, gs = np.array([(i, ts[i]['l'], ts[i].get('g0',0)) for i in ts if (np.isfinite(ts[i].get('g0',np.nan)))]).T
ordering = np.argsort(order)
ls, gs = ls[ordering], gs[ordering]
ls = ls[gs>0]
gs = gs[gs>0]
curves['equal-shock'] = {'l':ls, 'g':gs}    

def Delta(l, g):
    return 32 * l**6 * g**3 - 32 * l**8 * g**3 - 432 * l**4 * g**4 \
           + 560* l**6 * g**4 - 1440 * l**4 * g**5 - 96* l**6*g**5 \
           - 1184*l**4*g**6 - 96*l**4*g**7 - 16*l**2*g**8 - 32*l**2*g**9

# outer sonic collision


ts = {}
ts[0] = {}
ts[0]['l'] = l = 1.02
ts[0]['g'] = g = ac.secant(lambda g: Delta(l, g), 0.012, 0.013 )

ts[1] = {}
ts[1]['l'] = l = 1.021
ts[1]['g'] = g = ac.secant(lambda g: Delta(l, g), 0.012,0.013)

# shoot to lower l
dl = -.001
l += dl
i = 1
passed = True

while l > 1 and i < 10000 and passed:
    ts[-i] = {}
    g1, g0 = ts[-(i-1)]['g'], ts[-(i-2)]['g']
    passed = False
    while dl < -1e-6:
        try:
            l = ts[-(i-1)]['l'] + dl
            g = ac.secant(lambda g: Delta(l,g),g0,g1)
            if np.isnan(g): raise ValueError('ac.secant iteration did not converge')
            ts[-i]['l'] = l
            ts[-i]['g'] = g
            passed = True
            break
        except ValueError:
            dl *= 0.5
    
    if i%10 == 0: print(i, dl, l, g, Delta(l,g))
    i += 1

# shoot up
dl = .001
l = ts[0]['l']
i = 2
passed = True
while l < 2 and i < 10000 and passed:
    ts[i] = {}
    g1, g0 = ts[(i-1)]['g'], ts[(i-2)]['g']
    passed = False
    while dl > 1e-5:
        try:
            l = ts[(i-1)]['l'] + dl
            g = ac.secant(lambda g: Delta(l,g), g0, g1)
            if np.isnan(g) or np.abs(Delta(l,g)) > 1e-10: raise ValueError('ac.secant iteration did not converge')
            ts[i]['l'] = l
            ts[i]['g'] = g
            passed = True
            break
        except ValueError:
            dl *= 0.5
    
    if i%10 == 0: print(i, dl, l, g, Delta(l,g))
    i += 1
    
order, ls, gs = np.array([(i, ts[i]['l'], ts[i].get('g',0)) for i in ts if (np.isfinite(ts[i].get('g',np.nan)))]).T
ordering = np.argsort(order)
ls, gs = ls[ordering], gs[ordering]
ls = ls[gs>0]
gs = gs[gs>0]
curves['outer-sonic-collision'] = {'l':ls, 'g':gs}    

# inner sonic collision
ts = {}
ts[0] = {}
ts[0]['l'] = l = .6
ts[0]['g'] = g = ac.secant(lambda g: Delta(l, g), 0.025, 0.026 )

ts[1] = {}
ts[1]['l'] = l = .61
ts[1]['g'] = g = ac.secant(lambda g: Delta(l, g), 0.025,0.026)

# shoot to lower l
dl = -.01
l += dl
i = 1
passed = True

while l > 0 and i < 10000 and passed:
    ts[-i] = {}
    g1, g0 = ts[-(i-1)]['g'], ts[-(i-2)]['g']
    passed = False
    while dl < -1e-6:
        try:
            l = ts[-(i-1)]['l'] + dl
            g = ac.secant(lambda g: Delta(l,g),g0,g1)
            if np.isnan(g) or np.abs(Delta(l,g)) > 1e-10: raise ValueError('ac.secant iteration did not converge')
            ts[-i]['l'] = l
            ts[-i]['g'] = g
            passed = True
            break
        except ValueError:
            dl *= 0.5
    
    if i % 10 == 0: print(i, dl, l, g, Delta(l,g))
    i += 1
    
# shoot up
dl = .001
l = ts[0]['l']
i = 2
passed = True
while l < 2 and i < 10000 and passed:
    ts[i] = {}
    g1, g0 = ts[(i-1)]['g'], ts[(i-2)]['g']
    passed = False
    while dl > 1e-5:
        try:
            l = ts[(i-1)]['l'] + dl
            g = ac.secant(lambda g: Delta(l,g), g0, g1)
            if np.isnan(g) or np.abs(Delta(l,g)) > 1e-10: raise ValueError('ac.secant iteration did not converge')
            ts[i]['l'] = l
            ts[i]['g'] = g
            passed = True
            break
        except ValueError:
            dl *= 0.5
    
    if i%10==0: print(i, dl, l, g, Delta(l,g))
    i += 1    
    
order, ls, gs = np.array([(i, ts[i]['l'], ts[i].get('g',0)) for i in ts if (np.isfinite(ts[i].get('g',np.nan)))]).T
ordering = np.argsort(order)
ls, gs = ls[ordering], gs[ordering]
ls = ls[gs>0]
gs = gs[gs>0]
curves['inner-sonic-collision'] = {'l':ls, 'g':gs}

ls = np.linspace(0,1.5,1001)
gs = np.linspace(0,.2,1001)
ds = Delta(ls[:,None],gs[None,:])


# plot regime diagram
fig, ax = plt.subplots(figsize=(6,4))
p = ax.contourf(ls,gs,np.arcsinh(100*ds.T),np.linspace(-.1,.1,11),cmap='RdBu_r')
ax.set_facecolor('k')
ax.plot(curves['inner-sonic-collision']['l'],curves['inner-sonic-collision']['g'],
         'C2',linewidth=1.5,label='$r_{h,3}$: Inner-center sonic collision')
ax.plot(curves['tangent-shock']['l'],curves['tangent-shock']['g'],
         'C1',linewidth=1.5,label='$r_{h,2}$: Single shock - projection tangency')
ax.plot(curves['equal-shock']['l'],curves['equal-shock']['g'],
         'C4',linewidth=1.5,label='$r_{h,1}$: Two shocks - equal energy sonics')
ax.plot(curves['outer-sonic-collision']['l'],curves['outer-sonic-collision']['g'],
         'C5',linewidth=1.5,label='$r_{h,0}$: Outer-centre sonic collision')
legend = ax.legend(framealpha=1,loc=2,bbox_to_anchor=(.15,1),frameon=False)
for text in legend.get_texts(): text.set_color('w')
for line in legend.get_lines(): line.set_linewidth(2.0)
ax.set(xlabel='Angular momentum $ℓ$', ylabel='Event horizon $r_h$',title='Black hole accretion regime diagram in $ℓ, r_h$')
ax.set(yticks=[0,0.05,0.1,0.15,0.2])
plt.savefig('figures/black-hole-accretion-regime-diagram.pdf',bbox_inches='tight')

# save the curves
filename = 'regime-curves.h5'
for curve in curves:
    flt.save_data(filename,curves[curve]['l'], 'l',group=curve)
    flt.save_data(filename,curves[curve]['g'], 'g',group=curve)
