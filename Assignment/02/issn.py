
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(42); np.random.seed(42)
OUT = "/Users/Hp/Desktop/outputs"; os.makedirs(OUT, exist_ok=True)


def softplus_np(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))

def softplus_d_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))   # sigmoid

def softplus_dd_np(x):
    s = softplus_d_np(x)
    return s * (1.0 - s)

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_d_np(x):
    s = sigmoid_np(x)
    return s * (1.0 - s)

# generating dataset
def lhs(n, lo, hi, seed):
    raw = qmc.LatinHypercube(d=4, seed=seed).random(n=n)
    return qmc.scale(raw, [lo]*4, [hi]*4)

def dataset1(n_tr=500, n_te=5000):
    def f(X):
        x,y,t,z = X[:,0],X[:,1],X[:,2],X[:,3]
        return np.exp(-0.5*x) + np.log1p(np.exp(0.4*y)) + np.tanh(t) + np.sin(z) - 0.4
    Xtr = lhs(n_tr, 0, 4, 0); Xte = lhs(n_te, 0, 6, 1)
    return Xtr, f(Xtr), Xte, f(Xte), f

def dataset2(n_tr=500, n_te=5000):
    def g(X):
        x,y,t,z = X[:,0],X[:,1],X[:,2],X[:,3]
        return np.exp(-0.3*x)*(0.15*y)**2*np.tanh(0.3*t)*(0.2*np.sin(0.5*z+2)+0.5)
    Xtr = lhs(n_tr, 0,  4, 2); Xte = lhs(n_te, 0, 10, 3)
    return Xtr, g(Xtr), Xte, g(Xte), g

#pytorcg

class PosLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.log_w = nn.Parameter(torch.zeros(out_f, in_f))     # stored as log
        self.b     = nn.Parameter(torch.zeros(out_f)) if bias else None
        nn.init.normal_(self.log_w, mean=-1.0, std=0.3)          # small positive init

    @property
    def weight(self):
        return F.softplus(self.log_w)                            # always ≥ 0

    def forward(self, x):
        out = x @ self.weight.T
        return out + self.b if self.b is not None else out


class ISNN1_PT(nn.Module):
    def __init__(self, n_layers=2, nh=10):
        super().__init__()
        d = 1
        # y branch: NonNeg + softplus activation
        self.y = nn.ModuleList([PosLinear(d if i==0 else nh, nh) for i in range(n_layers)])
        # z branch: free + sigmoid
        self.z = nn.ModuleList([nn.Linear(d if i==0 else nh, nh) for i in range(n_layers)])
        # t branch: NonNeg + sigmoid
        self.t = nn.ModuleList([PosLinear(d if i==0 else nh, nh) for i in range(n_layers)])
        # x branch: first layer cross connections, rest NonNeg
        self.x0_xx = nn.Linear(d, nh, bias=True)       # W0[xx] – free
        self.x0_xy = PosLinear(nh, nh, bias=False)     # W[xy]  – nonneg
        self.x0_xz = nn.Linear(nh, nh, bias=False)     # W[xz]  – free
        self.x0_xt = PosLinear(nh, nh, bias=False)     # W[xt]  – nonneg
        self.x_rest = nn.ModuleList([PosLinear(nh, nh) for _ in range(n_layers-1)])
        self.out   = nn.Linear(nh, 1)

    def _sp(self, x): return F.softplus(x)
    def _sig(self, x): return torch.sigmoid(x)

    def forward(self, X):
        x0,y0,t0,z0 = X[:,0:1], X[:,1:2], X[:,2:3], X[:,3:4]
        yh = y0
        for ly in self.y: yh = self._sp(ly(yh))
        zh = z0
        for lz in self.z: zh = self._sig(lz(zh))
        th = t0
        for lt in self.t: th = self._sig(lt(th))
        # x layer 0 (Eq. 4)
        F0 = self.x0_xx(x0) + self.x0_xy(yh) + self.x0_xz(zh) + self.x0_xt(th)
        xh = self._sp(F0)
        for lx in self.x_rest: xh = self._sp(lx(xh))
        return self.out(xh)


class ISNN2_PT(nn.Module):
    def __init__(self, H=2, nh=15):
        super().__init__()
        d = 1
        nb = H - 1   # number of hidden layers in side branches
        self.y = nn.ModuleList([PosLinear(d if i==0 else nh, nh) for i in range(nb)])
        self.z = nn.ModuleList([nn.Linear(d if i==0 else nh, nh) for i in range(nb)])
        self.t = nn.ModuleList([PosLinear(d if i==0 else nh, nh) for i in range(nb)])
        # x layer 0
        self.x0_xx  = nn.Linear(d, nh, bias=False)
        self.x0_xy  = PosLinear(d, nh, bias=False)
        self.x0_xz  = nn.Linear(d, nh, bias=False)
        self.x0_xt  = PosLinear(d, nh, bias=False)
        self.x0_b   = nn.Parameter(torch.zeros(nh))
        # x layers 1..H-1 with skip from x0
        self.xh_xx   = nn.ModuleList([PosLinear(nh, nh, bias=False) for _ in range(nb)])
        self.xh_xx0  = nn.ModuleList([nn.Linear(d, nh, bias=False) for _ in range(nb)])
        self.xh_xy   = nn.ModuleList([PosLinear(nh, nh, bias=False) for _ in range(nb)])
        self.xh_xz   = nn.ModuleList([nn.Linear(nh, nh, bias=False) for _ in range(nb)])
        self.xh_xt   = nn.ModuleList([PosLinear(nh, nh, bias=False) for _ in range(nb)])
        self.xh_b    = nn.ParameterList([nn.Parameter(torch.zeros(nh)) for _ in range(nb)])
        self.out = nn.Linear(nh, 1)

    def _sp(self, x): return F.softplus(x)
    def _sig(self, x): return torch.sigmoid(x)

    def forward(self, X):
        x0,y0,t0,z0 = X[:,0:1], X[:,1:2], X[:,2:3], X[:,3:4]
        yh = y0
        for ly in self.y: yh = self._sp(ly(yh))
        zh = z0
        for lz in self.z: zh = self._sig(lz(zh))
        th = t0
        for lt in self.t: th = self._sig(lt(th))
        # x layer 0 (Eq. 9)
        xh = self._sp(self.x0_xx(x0) + self.x0_xy(y0) + self.x0_xz(z0) + self.x0_xt(t0) + self.x0_b)
        # x layers 1..H-1 (Eq. 10)
        for Wxx,Wxx0,Wxy,Wxz,Wxt,b in zip(self.xh_xx,self.xh_xx0,self.xh_xy,self.xh_xz,self.xh_xt,self.xh_b):
            xh = self._sp(Wxx(xh) + Wxx0(x0) + Wxy(yh) + Wxz(zh) + Wxt(th) + b)
        return self.out(xh)


class FFNN_PT(nn.Module):
    def __init__(self, n_layers=2, nh=30):
        super().__init__()
        layers = [nn.Linear(4, nh), nn.Tanh()]
        for _ in range(n_layers-1): layers += [nn.Linear(nh,nh), nn.Tanh()]
        layers.append(nn.Linear(nh, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, X): return self.net(X)


def train_pt(model, Xtr, ytr, Xte, yte, epochs=3000, lr=1e-3, log=500):
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.MSELoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.float32).unsqueeze(1)
    tr_ls, te_ls = [], []
    for ep in range(1, epochs+1):
        model.train(); opt.zero_grad()
        loss = mse(model(Xtr_t), ytr_t); loss.backward(); opt.step()
        with torch.no_grad():
            model.eval(); tl = mse(model(Xte_t), yte_t).item()
        tr_ls.append(loss.item()); te_ls.append(tl)
        if ep % log == 0 or ep == 1:
            print(f"    ep {ep:>5d}  train={loss.item():.3e}  test={tl:.3e}")
    return np.array(tr_ls), np.array(te_ls)

def pt_predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy().ravel()

# manual backprop

class AdamState:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr  = lr; self.b1 = b1; self.b2 = b2; self.eps = eps; self.t = 0
        self.m = {k: np.zeros_like(v) for k,v in params.items()}
        self.v = {k: np.zeros_like(v) for k,v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            if grads.get(k) is None: continue
            g = grads[k]
            self.m[k] = self.b1*self.m[k] + (1-self.b1)*g
            self.v[k] = self.b2*self.v[k] + (1-self.b2)*g**2
            mh = self.m[k]/(1-self.b1**self.t)
            vh = self.v[k]/(1-self.b2**self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh)+self.eps)


def _mk_params1(d, nl, nh):
    p = {}
    scale = 0.1
    for i in range(nl):
        ind = d if i==0 else nh
        p[f'Wyy_{i}'] = np.random.randn(nh, ind)*scale    # log(raw) init
        p[f'by_{i}']  = np.zeros(nh)
        p[f'Wzz_{i}'] = np.random.randn(nh, ind)*scale
        p[f'bz_{i}']  = np.zeros(nh)
        p[f'Wtt_{i}'] = np.random.randn(nh, ind)*scale
        p[f'bt_{i}']  = np.zeros(nh)
        p[f'Wxx_{i}'] = np.random.randn(nh, d if i==0 else nh)*scale
        p[f'bx_{i}']  = np.zeros(nh)
    p['Wxy'] = np.random.randn(nh, nh)*scale   # cross: nonneg raw
    p['Wxz'] = np.random.randn(nh, nh)*scale   # cross: free
    p['Wxt'] = np.random.randn(nh, nh)*scale   # cross: nonneg raw
    p['W_out'] = np.random.randn(1, nh)*scale
    p['b_out'] = np.zeros(1)
    return p

def _mk_params2(d, H, nh):
    p = {}; nb = H-1; scale = 0.1
    for i in range(nb):
        ind = d if i==0 else nh
        p[f'Wyy_{i}'] = np.random.randn(nh, ind)*scale
        p[f'by_{i}']  = np.zeros(nh)
        p[f'Wzz_{i}'] = np.random.randn(nh, ind)*scale
        p[f'bz_{i}']  = np.zeros(nh)
        p[f'Wtt_{i}'] = np.random.randn(nh, ind)*scale
        p[f'bt_{i}']  = np.zeros(nh)
    # x layer 0
    p['Wxx0'] = np.random.randn(nh, d)*scale   # free
    p['Wxy0'] = np.random.randn(nh, d)*scale   # nonneg raw
    p['Wxz0'] = np.random.randn(nh, d)*scale   # free
    p['Wxt0'] = np.random.randn(nh, d)*scale   # nonneg raw
    p['bx0']  = np.zeros(nh)
    # x layers 1..H-1
    for i in range(nb):
        p[f'Wxx_{i}']    = np.random.randn(nh, nh)*scale   # nonneg raw
        p[f'Wxx0s_{i}']  = np.random.randn(nh, d)*scale    # skip from x0, free
        p[f'Wxy_{i}']    = np.random.randn(nh, nh)*scale   # nonneg raw
        p[f'Wxz_{i}']    = np.random.randn(nh, nh)*scale   # free
        p[f'Wxt_{i}']    = np.random.randn(nh, nh)*scale   # nonneg raw
        p[f'bx_{i}']     = np.zeros(nh)
    p['W_out'] = np.random.randn(1, nh)*scale
    p['b_out'] = np.zeros(1)
    return p


#issn1
class ManualISNN1:
    def __init__(self, nl=2, nh=10, lr=1e-3, d=1):
        self.nl=nl; self.nh=nh; self.d=d
        self.p = _mk_params1(d, nl, nh)
        self.adam = AdamState(self.p, lr=lr)

    def forward(self, X):
        p = self.p; sp=softplus_np; sg=sigmoid_np
        x0=X[:,0:1]; y0=X[:,1:2]; t0=X[:,2:3]; z0=X[:,3:4]
        c = {'x0':x0,'y0':y0,'t0':t0,'z0':z0}
        # ── y branch ──
        yh=y0; ay=[y0]; Zy=[]
        for i in range(self.nl):
            W=sp(p[f'Wyy_{i}']); z=yh@W.T+p[f'by_{i}']
            Zy.append(z); yh=sp(z); ay.append(yh)
        c['ay']=ay; c['Zy']=Zy; c['yh']=yh
        # ── z branch ──
        zh=z0; az=[z0]; Zz=[]
        for i in range(self.nl):
            z=zh@p[f'Wzz_{i}'].T+p[f'bz_{i}']
            Zz.append(z); zh=sg(z); az.append(zh)
        c['az']=az; c['Zz']=Zz; c['zh']=zh
        # ── t branch ──
        th=t0; at=[t0]; Zt=[]
        for i in range(self.nl):
            W=sp(p[f'Wtt_{i}']); z=th@W.T+p[f'bt_{i}']
            Zt.append(z); th=sg(z); at.append(th)
        c['at']=at; c['Zt']=Zt; c['th']=th
        # ── x layer 0 (Eq.4) ──
        Wxy=sp(p['Wxy']); Wxt=sp(p['Wxt'])
        F0 = x0@p['Wxx_0'].T+p['bx_0'] + yh@Wxy.T + zh@p['Wxz'].T + th@Wxt.T
        xh=sp(F0); ax=[x0,xh]; Zx=[F0]
        c['Wxy_pos']=Wxy; c['Wxt_pos']=Wxt
        # ── x layers 1..nl-1 (Eq.5, NonNeg) ──
        for i in range(1, self.nl):
            W=sp(p[f'Wxx_{i}']); z=xh@W.T+p[f'bx_{i}']
            Zx.append(z); xh=sp(z); ax.append(xh)
        c['ax']=ax; c['Zx']=Zx
        out = xh@p['W_out'].T+p['b_out']
        c['xh']=xh
        return out, c

    def backward(self, X, y, out, c):
        p=self.p; sp_d=softplus_d_np; sg_d=sigmoid_d_np
        N=X.shape[0]
        g={}
        # ── output ──
        dL = 2/N*(out - y.reshape(-1,1))
        g['W_out'] = dL.T@c['xh']
        g['b_out'] = dL.sum(0)
        delta = dL@p['W_out']            # (N, nh)
        # ── x layers nl-1..1 (reverse) ──
        for i in range(self.nl-1, 0, -1):
            W_pos=softplus_np(p[f'Wxx_{i}'])
            d = delta*sp_d(c['Zx'][i])
            g[f'Wxx_{i}'] = (d.T@c['ax'][i])*sp_d(p[f'Wxx_{i}'])  # chain thru softplus
            g[f'bx_{i}']  = d.sum(0)
            delta = d@W_pos
        # ── x layer 0 ──
        d0 = delta*sp_d(c['Zx'][0])
        g['Wxx_0']  = d0.T@c['x0']                                    # free weight
        g['bx_0']   = d0.sum(0)
        g['Wxy']    = (d0.T@c['yh'])*sp_d(p['Wxy'])
        g['Wxz']    = d0.T@c['zh']
        g['Wxt']    = (d0.T@c['th'])*sp_d(p['Wxt'])
        # ── into y branch ──
        dy = d0@c['Wxy_pos']
        for i in range(self.nl-1, -1, -1):
            W_pos=softplus_np(p[f'Wyy_{i}'])
            dy_d = dy*sp_d(c['Zy'][i])
            g[f'Wyy_{i}'] = (dy_d.T@c['ay'][i])*sp_d(p[f'Wyy_{i}'])
            g[f'by_{i}']  = dy_d.sum(0)
            dy = dy_d@W_pos
        # ── into z branch ──
        dz = d0@p['Wxz']
        for i in range(self.nl-1, -1, -1):
            dz_d = dz*sg_d(c['Zz'][i])
            g[f'Wzz_{i}'] = dz_d.T@c['az'][i]
            g[f'bz_{i}']  = dz_d.sum(0)
            dz = dz_d@p[f'Wzz_{i}']
        # ── into t branch ──
        dt = d0@c['Wxt_pos']
        for i in range(self.nl-1, -1, -1):
            W_pos=softplus_np(p[f'Wtt_{i}'])
            dt_d = dt*sg_d(c['Zt'][i])
            g[f'Wtt_{i}'] = (dt_d.T@c['at'][i])*sp_d(p[f'Wtt_{i}'])
            g[f'bt_{i}']  = dt_d.sum(0)
            dt = dt_d@W_pos
        return g

    def train_epoch(self, X, y):
        out, c = self.forward(X)
        loss = float(np.mean((out.ravel()-y.ravel())**2))
        g = self.backward(X, y, out, c)
        self.adam.step(self.p, g)
        return loss

    def predict(self, X):
        out, _ = self.forward(X)
        return out.ravel()


# issn2
class ManualISNN2:
    def __init__(self, H=2, nh=15, lr=1e-3, d=1):
        self.H=H; self.nh=nh; self.d=d; self.nb=H-1
        self.p = _mk_params2(d, H, nh)
        self.adam = AdamState(self.p, lr=lr)

    def forward(self, X):
        p=self.p; sp=softplus_np; sg=sigmoid_np
        x0=X[:,0:1]; y0=X[:,1:2]; t0=X[:,2:3]; z0=X[:,3:4]
        c={'x0':x0,'y0':y0,'t0':t0,'z0':z0}
        yh=y0; ay=[y0]; Zy=[]
        for i in range(self.nb):
            W=sp(p[f'Wyy_{i}']); z=yh@W.T+p[f'by_{i}']
            Zy.append(z); yh=sp(z); ay.append(yh)
        c['ay']=ay; c['Zy']=Zy; c['yh']=yh
        zh=z0; az=[z0]; Zz=[]
        for i in range(self.nb):
            z=zh@p[f'Wzz_{i}'].T+p[f'bz_{i}']
            Zz.append(z); zh=sg(z); az.append(zh)
        c['az']=az; c['Zz']=Zz; c['zh']=zh
        th=t0; at=[t0]; Zt=[]
        for i in range(self.nb):
            W=sp(p[f'Wtt_{i}']); z=th@W.T+p[f'bt_{i}']
            Zt.append(z); th=sg(z); at.append(th)
        c['at']=at; c['Zt']=Zt; c['th']=th
        Wxy0=sp(p['Wxy0']); Wxt0=sp(p['Wxt0'])
        F0=x0@p['Wxx0'].T + y0@Wxy0.T + z0@p['Wxz0'].T + t0@Wxt0.T + p['bx0']
        xh=sp(F0); ax=[x0,xh]; Zx=[F0]
        c['Wxy0_pos']=Wxy0; c['Wxt0_pos']=Wxt0
        Wxx_pos_cache=[]; Wxy_pos_cache=[]; Wxt_pos_cache=[]
        for i in range(self.nb):
            Wxx_pos=sp(p[f'Wxx_{i}'])
            Wxy_pos=sp(p[f'Wxy_{i}'])
            Wxt_pos=sp(p[f'Wxt_{i}'])
            Wxx_pos_cache.append(Wxx_pos)
            Wxy_pos_cache.append(Wxy_pos)
            Wxt_pos_cache.append(Wxt_pos)
            Fh=(xh@Wxx_pos.T + x0@p[f'Wxx0s_{i}'].T
                + yh@Wxy_pos.T + zh@p[f'Wxz_{i}'].T + th@Wxt_pos.T + p[f'bx_{i}'])
            Zx.append(Fh); xh=sp(Fh); ax.append(xh)
        c['ax']=ax; c['Zx']=Zx
        c['Wxx_pos']=Wxx_pos_cache
        c['Wxy_pos']=Wxy_pos_cache
        c['Wxt_pos']=Wxt_pos_cache
        out=xh@p['W_out'].T+p['b_out']
        c['xh']=xh
        return out, c

    def backward(self, X, y, out, c):
        p=self.p; sp_d=softplus_d_np; sg_d=sigmoid_d_np
        N=X.shape[0]; g={}
        x0=X[:,0:1]
        dL=2/N*(out-y.reshape(-1,1))
        g['W_out']=dL.T@c['xh']; g['b_out']=dL.sum(0)
        delta=dL@p['W_out']
      
        for i in range(self.nb-1, -1, -1):
            d=delta*sp_d(c['Zx'][i+1])
            g[f'Wxx_{i}']   = (d.T@c['ax'][i+1])*sp_d(p[f'Wxx_{i}'])
            g[f'Wxx0s_{i}'] =  d.T@x0
            g[f'Wxy_{i}']   = (d.T@c['yh'])*sp_d(p[f'Wxy_{i}'])
            g[f'Wxz_{i}']   =  d.T@c['zh']
            g[f'Wxt_{i}']   = (d.T@c['th'])*sp_d(p[f'Wxt_{i}'])
            g[f'bx_{i}']    =  d.sum(0)
         
            if i == self.nb-1:
                dy_acc = d@c['Wxy_pos'][i]
                dz_acc = d@p[f'Wxz_{i}']
                dt_acc = d@c['Wxt_pos'][i]
            else:
                dy_acc += d@c['Wxy_pos'][i]
                dz_acc += d@p[f'Wxz_{i}']
                dt_acc += d@c['Wxt_pos'][i]
            delta = d@c['Wxx_pos'][i]
        # x layer 0 (Eq.9)
        d0=delta*sp_d(c['Zx'][0])
        g['Wxx0']    = d0.T@c['x0']
        g['Wxy0']    = (d0.T@c['y0'])*sp_d(p['Wxy0'])
        g['Wxz0']    = d0.T@c['z0']
        g['Wxt0']    = (d0.T@c['t0'])*sp_d(p['Wxt0'])
        g['bx0']     = d0.sum(0)
        # y branch backprop
        if self.nb > 0:
            dy = d0@c['Wxy0_pos'] + dy_acc*sp_d(c['Zy'][-1])
        else:
            dy = d0@c['Wxy0_pos']
        for i in range(self.nb-1,-1,-1):
            W_pos=softplus_np(p[f'Wyy_{i}'])
            dy_d=dy*sp_d(c['Zy'][i])
            g[f'Wyy_{i}']=(dy_d.T@c['ay'][i])*sp_d(p[f'Wyy_{i}'])
            g[f'by_{i}']=dy_d.sum(0)
            dy=dy_d@W_pos
        # z branch backprop
        if self.nb > 0:
            dz = d0@p['Wxz0'] + dz_acc*sg_d(c['Zz'][-1])
        else:
            dz = d0@p['Wxz0']
        for i in range(self.nb-1,-1,-1):
            dz_d=dz*sg_d(c['Zz'][i])
            g[f'Wzz_{i}']=dz_d.T@c['az'][i]
            g[f'bz_{i}']=dz_d.sum(0)
            dz=dz_d@p[f'Wzz_{i}']
        # t branch backprop
        if self.nb > 0:
            dt = d0@c['Wxt0_pos'] + dt_acc*sg_d(c['Zt'][-1])
        else:
            dt = d0@c['Wxt0_pos']
        for i in range(self.nb-1,-1,-1):
            W_pos=softplus_np(p[f'Wtt_{i}'])
            dt_d=dt*sg_d(c['Zt'][i])
            g[f'Wtt_{i}']=(dt_d.T@c['at'][i])*sp_d(p[f'Wtt_{i}'])
            g[f'bt_{i}']=dt_d.sum(0)
            dt=dt_d@W_pos
        return g

    def train_epoch(self, X, y):
        out,c=self.forward(X)
        loss=float(np.mean((out.ravel()-y.ravel())**2))
        g=self.backward(X, y, out, c)
        self.adam.step(self.p, g)
        return loss

    def predict(self, X):
        out,_=self.forward(X)
        return out.ravel()

#traing 

def train_manual(model, Xtr, ytr, Xte, yte, epochs=2000, log=400):
    tr_ls, te_ls = [], []
    for ep in range(1, epochs+1):
        tl = model.train_epoch(Xtr, ytr)
        tl_test = float(np.mean((model.predict(Xte)-yte.ravel())**2))
        tr_ls.append(tl); te_ls.append(tl_test)
        if ep % log == 0 or ep == 1:
            print(f"    ep {ep:>5d}  train={tl:.3e}  test={tl_test:.3e}")
    return np.array(tr_ls), np.array(te_ls)



CLRS  = {'FFNN':'#e63946','ISNN-1 (PT)':'#2a9d8f','ISNN-2 (PT)':'#457b9d',
          'ISNN-1 (NP)':'#f4a261','ISNN-2 (NP)':'#264653'}
LSTY  = {'FFNN':'-','ISNN-1 (PT)':'-','ISNN-2 (PT)':'-',
         'ISNN-1 (NP)':'--','ISNN-2 (NP)':'--'}

def plot_loss(results, ds_label, fname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f'Loss Curves — Dataset {ds_label}', fontsize=13, fontweight='bold')
    for nm,(tr,te) in results.items():
        ep=np.arange(1,len(tr)+1)
        axes[0].semilogy(ep, tr, LSTY[nm], c=CLRS[nm], label=nm, lw=1.6, alpha=0.9)
        axes[1].semilogy(ep, te, LSTY[nm], c=CLRS[nm], label=nm, lw=1.6, alpha=0.9)
    for ax, ttl in zip(axes,['(a) Training Loss','(b) Test Loss']):
        ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
        ax.set_title(ttl); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path=f"{OUT}/{fname}"; plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓ {fname}"); return path

def plot_response(preds, true_fn, hi, ds_label, fname):
    """Figure 4/6 style: all models in one figure."""
    v4  = np.linspace(0, 4, 200)
    v_full = np.linspace(0, hi, 400)
    def diag(v): return np.column_stack([v,v,v,v])
    true_full = true_fn(diag(v_full))

    nm_list = list(preds.keys())
    ncols = len(nm_list)
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4))
    if ncols == 1: axes = [axes]
    fig.suptitle(f'Behavioral Response — Dataset {ds_label}', fontsize=13, fontweight='bold')

    for ax, nm in zip(axes, nm_list):
        pred = preds[nm](diag(v_full))
        clr  = CLRS.get(nm, '#888888')
        ax.plot(v_full, true_full, 'k--', lw=1.8, label='True response')
        mask_i = v_full <= 4.0; mask_e = v_full > 4.0
        ax.plot(v_full[mask_i], pred[mask_i], '-', c=clr, lw=2, label='Interpolated')
        ax.fill_between(v_full[mask_e], pred[mask_e]*0.92, pred[mask_e]*1.08,
                        alpha=0.3, color=clr)
        ax.plot(v_full[mask_e], pred[mask_e], '-', c=clr, lw=2, label='Extrapolated')
        ax.axvline(4.0, c='gray', ls=':', lw=1)
        ax.set_title(f'({nm})', fontsize=11)
        ax.set_xlabel('x = y = t = z'); ax.set_ylabel('output')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path=f"{OUT}/{fname}"; plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓ {fname}"); return path



def run(ds_id, make_ds, hi, EPT=3000, ENP=2000):
    print(f"\n{'='*60}"); print(f" DATASET {ds_id}"); print(f"{'='*60}")
    Xtr, ytr, Xte, yte, true_fn = make_ds()
    np.save(f"{OUT}/ds{ds_id}_train.npy", np.c_[Xtr,ytr])
    np.save(f"{OUT}/ds{ds_id}_test.npy",  np.c_[Xte,yte])
    print(f"  {Xtr.shape[0]} train | {Xte.shape[0]} test")

    results={}; preds={}

    print("\n  [PT] FFNN")
    m=FFNN_PT(); tr_l,te_l=train_pt(m,Xtr,ytr,Xte,yte,epochs=EPT,log=EPT//5)
    results['FFNN']=(tr_l,te_l); preds['FFNN']=lambda X,m=m: pt_predict(m,X)

    print("\n  [PT] ISNN-1")
    m=ISNN1_PT(); tr_l,te_l=train_pt(m,Xtr,ytr,Xte,yte,epochs=EPT,log=EPT//5)
    results['ISNN-1 (PT)']=(tr_l,te_l); preds['ISNN-1 (PT)']=lambda X,m=m: pt_predict(m,X)

    print("\n  [PT] ISNN-2")
    m=ISNN2_PT(); tr_l,te_l=train_pt(m,Xtr,ytr,Xte,yte,epochs=EPT,log=EPT//5)
    results['ISNN-2 (PT)']=(tr_l,te_l); preds['ISNN-2 (PT)']=lambda X,m=m: pt_predict(m,X)

    print("\n  [NP] ISNN-1")
    m=ManualISNN1(); tr_l,te_l=train_manual(m,Xtr,ytr,Xte,yte,epochs=ENP,log=ENP//5)
    results['ISNN-1 (NP)']=(tr_l,te_l); preds['ISNN-1 (NP)']=lambda X,m=m: m.predict(X)

    print("\n  [NP] ISNN-2")
    m=ManualISNN2(); tr_l,te_l=train_manual(m,Xtr,ytr,Xte,yte,epochs=ENP,log=ENP//5)
    results['ISNN-2 (NP)']=(tr_l,te_l); preds['ISNN-2 (NP)']=lambda X,m=m: m.predict(X)

    lf=plot_loss(results,ds_id,f"fig{ds_id}_loss.png")
    rf=plot_response(preds,true_fn,hi,ds_id,f"fig{ds_id}_response.png")
    return lf,rf

if __name__=="__main__":
    f1l,f1r = run(1, dataset1, hi=6.0,  EPT=4000, ENP=2500)
    f2l,f2r = run(2, dataset2, hi=10.0, EPT=4000, ENP=2500)
    print("\nAll outputs saved to:", OUT)
