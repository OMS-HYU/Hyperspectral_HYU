"""Known-wavelength native K=1 reproduction CLI. Derived from the actual 2026-11-16 scripts.
Generation and evaluation can read truth; GD and L-BFGS cannot. Never infer phase from a wavelength-only pass.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
DEFAULT_ARCHIVE=HERE.parent/'데이터'/'05_NN_metacam-main.zip'
PLAN=[(64,35),(128,35),(256,40),(512,40)]
TAU=2*np.pi

def sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def loadj(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,obj):
    tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(p)
def save_state(p,**kw):
    tmp=p.with_suffix('.tmp.npz'); np.savez_compressed(tmp,**kw); tmp.replace(p)

def save_phase(p,a):
    tmp=p.with_suffix('.tmp.npy');np.save(tmp,a);tmp.replace(p)
def model(a, wavelength):
    from native_model import Model
    return Model(a.archive,wavelength)
def inputs(a):
    r=a.run; m=loadj(r/'manifest.json'); y=np.load(r/'sensor.npy',allow_pickle=False)
    if m['geometry'] != [5713,2857,512,3000] or m['observation']!='amplitude' or not m['known']: raise ValueError('Not a native known-amplitude run')
    if y.shape!=(3000,3000) or sha(y)!=m['sensor_sha256']: raise ValueError('Sensor hash or shape mismatch')
    return r,m,y

def generate(a):
    from scipy import fft
    t=np.load(a.truth,allow_pickle=False)
    if t.shape!=(512,512) or not np.isfinite(t).all() or t.min()<0 or t.max()>TAU+1e-6: raise ValueError('Truth must be a finite 512x512 phase in [0,2pi]')
    r=a.run; r.mkdir(parents=True,exist_ok=True)
    if any((r/name).exists() for name in ('manifest.json','sensor.npy','private_truth.npy')):
        raise FileExistsError(f'Run already contains generated inputs: {r}')
    with fft.set_workers(4):
        m=model(a,a.wavelength); clean=m.measure(t)
    sigma=.01*float(clean.mean()); y=np.maximum(0,clean+sigma*np.random.default_rng(a.seed).standard_normal(clean.shape).astype(np.float32))
    save_phase(r/'private_truth.npy',t); save_phase(r/'sensor.npy',y)
    write(r/'manifest.json',dict(scene=a.scene,geometry=[5713,2857,512,3000],wavelength_nm=a.wavelength,known=True,noise_seed=a.seed,noise_sigma=sigma,sensor_sha256=sha(y),truth_sha256=sha(t),archive_sha256=m.archive_sha256,expected_noise_l2=float(sigma*np.sqrt(y.size)/np.linalg.norm(y.ravel())),initialization='zero; no transfer',observation='amplitude'))
    print('GENERATED',r,'sensor_sha256',sha(y),flush=True)

def gd(a):
    import torch
    import torch.nn.functional as F
    from scipy import fft
    r,man,sensor=inputs(a); y=torch.from_numpy(sensor); state=r/'gd_state.npz'; hist=[]
    if a.steps<1: raise ValueError('steps must be positive')
    def lift(x,n):
        t=torch.from_numpy(x.reshape(n,n)); return t if n==512 else F.interpolate(t[None,None],size=(512,512),mode='bilinear',align_corners=False)[0,0]
    def phase(x,n): return lift(x,n).numpy().copy()
    def pull(g,n):
        if n==512:return g.copy()
        t=torch.zeros((n,n),dtype=torch.float64,requires_grad=True); z=F.interpolate(t[None,None],size=(512,512),mode='bilinear',align_corners=False)[0,0]; z.backward(torch.from_numpy(g));return t.grad.numpy().copy()
    cfg=dict(scene=man['scene'],wavelength_nm=man['wavelength_nm'],geometry=man['geometry'],sensor_sha256=man['sensor_sha256'],pyramid=[list(p) for p in PLAN],gd_prior=0,model_sha256=hashlib.sha256((HERE/'native_model.py').read_bytes()).hexdigest(),solver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest())
    if (r/'native_gd_config.json').exists() and loadj(r/'native_gd_config.json')!=cfg: raise ValueError('Resume configuration mismatch')
    write(r/'native_gd_config.json',cfg)
    with fft.set_workers(4):
        m=model(a,man['wavelength_nm']); start=time.monotonic(); stage=local=total=0; n=64; x=np.zeros((n,n)); xp=gp=np.empty(0)
        def fg(x,n):
            f,g,d,reg,q=m.fg(phase(x,n),y,0.);return f,pull(g,n),d,reg,q
        def checkpoint():
            save_state(state,x=x,g=g,xprev=xp,gprev=gp,n=n,stage=stage,local=local,total=total,last=np.array([f,d,reg,q]),sensor_sha=man['sensor_sha256'],history_json=json.dumps(hist))
            p=np.mod(phase(x,n),TAU); save_phase(r/'latest_phase.npy',p); write(r/'history.json',hist);write(r/'progress.json',dict(stage=stage,grid=n,stage_accepted=local,gd_accepted=total,gd_target=150,latest=hist[-1],complete=total==150,sensor_sha256=man['sensor_sha256'],phase_sha256=sha(p)))
            if total in [10,20,35,70,110,150]:save_phase(r/f'gd_{total:03}.npy',p)
        if state.exists():
            with np.load(state,allow_pickle=False) as z:
                if str(z['sensor_sha'])!=man['sensor_sha256']:raise ValueError('GD state sensor mismatch')
                x=z['x'];g=z['g'];xp=z['xprev'];gp=z['gprev'];n=int(z['n']);stage=int(z['stage']);local=int(z['local']);total=int(z['total']);f,d,reg,q=z['last']
                hist=json.loads(str(z['history_json'])) if 'history_json' in z else loadj(r/'history.json')
            assert hist[-1]['gd']==total
            checkpoint()  # Rebuild derived files from the one atomic authoritative state.
        else:
            f,g,d,reg,q=fg(x,n);hist.append(dict(gd=0,stage=stage,grid=n,stage_step=0,sensor_l2=float(np.sqrt(d)),objective=float(f),prior=float(reg),scale=float(q),step=0,backtracks=0));checkpoint()
        target=min(total+a.steps,150)
        while total<target:
            if local==PLAN[stage][1]:
                stage+=1; nn=PLAN[stage][0]; x=F.interpolate(torch.from_numpy(x)[None,None],size=(nn,nn),mode='bilinear',align_corners=False)[0,0].numpy().copy(); n=nn;local=0;xp=gp=np.empty(0);f,g,d,reg,q=fg(x,n)
            gn=float(np.max(np.abs(g)));assert np.isfinite(gn) and gn>0;step=.5/gn
            if xp.size:
                s=x-xp;v=g-gp;sv=float(np.sum(s*v));ss=float(np.sum(s*s))
                if sv>0:step=min(step,max(ss/sv,1e-12))
            slope=float(np.sum(g*g));accepted=False
            for bt in range(16):
                z=x-step*g;fn,gg,dn,rn,qn=fg(z,n)
                if np.isfinite(fn) and fn<=f-1e-4*step*slope:accepted=True;break
                step*=.5
            if not accepted:print('STOP line search',total,flush=True);break
            xp,gp=x,g;x,g,f,d,reg,q=z,gg,fn,dn,rn,qn;total+=1;local+=1
            row=dict(gd=total,stage=stage,grid=n,stage_step=local,sensor_l2=float(np.sqrt(d)),objective=float(f),prior=float(reg),scale=float(q),step=float(step),backtracks=bt,elapsed_seconds=time.monotonic()-start,forward_calls=m.count,adjoint_calls=m.adj);hist.append(row);checkpoint()
            print('ACCEPT',json.dumps(row),flush=True)
        print('END GD',total,'target',target,flush=True)
        if total<target:raise RuntimeError('GD stopped before requested accepted count')

def lbfgs(a):
    import torch,scipy
    from scipy import fft
    from scipy.optimize import _lbfgsb
    if scipy.__version__!='1.17.0':raise RuntimeError('Full-memory resume requires scipy==1.17.0')
    if not 1<=a.until<=150 or a.alpha<0: raise ValueError('invalid iteration or prior')
    r,man,sensor=inputs(a); gp=loadj(r/'progress.json'); assert gp['gd_accepted']==150
    p=np.load(r/'gd_150.npy',allow_pickle=False);assert sha(p)==gp['phase_sha256'];y=torch.from_numpy(sensor)
    out=r/f'long_alpha_{a.alpha:g}';out.mkdir(exist_ok=True);state=out/'optimizer_state.npz'
    config=dict(alpha=a.alpha,init_sha256=sha(p),sensor_sha256=sha(sensor),gd_accepted=150,planned_lbfgs=150,scipy=scipy.__version__,maxcor=10,maxls=20,ftol=1e-15,gtol=1e-10,objective_scale=1e5,truth_used=False,model_sha256=hashlib.sha256((HERE/'native_model.py').read_bytes()).hexdigest(),solver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest())
    n=p.size;m=10;x=p.ravel().copy();f=np.array(0.,dtype=np.float64);g=np.zeros(n);wa=np.zeros(2*m*n+5*n+11*m*m+8*m);iwa=np.zeros(3*n,dtype=np.int32);task=np.zeros(2,dtype=np.int32);ln_task=task.copy();lsave=np.zeros(4,dtype=np.int32);isave=np.zeros(44,dtype=np.int32);dsave=np.zeros(29);nbd=np.zeros(n,dtype=np.int32);l=np.zeros(n);u=np.zeros(n);history=[];nfev=0;last=np.zeros(4);accepted=0;elapsed=0.
    if state.exists():
        assert loadj(out/'config.json')==config
        with np.load(state,allow_pickle=False) as z:
            x=z['x'];f=z['f'];g=z['g'];wa=z['wa'];iwa=z['iwa'];task=z['task'];ln_task=z['ln_task'];lsave=z['lsave'];isave=z['isave'];dsave=z['dsave'];last=z['last'];nfev=int(z['nfev']);accepted=int(z['accepted']);elapsed=float(z['elapsed'])
            history=json.loads(str(z['history_json'])) if 'history_json' in z else loadj(out/'history.json')
        assert accepted==len(history) and task[0]==1
    else:write(out/'config.json',config)
    with fft.set_workers(4):
        optical=model(a,man['wavelength_nm']);start=time.monotonic();termination='paused at accepted iteration; full memory preserved'
        def checkpoint():
            ph=np.mod(x.reshape(512,512),TAU);save_state(state,x=x,f=f,g=g,wa=wa,iwa=iwa,task=task,ln_task=ln_task,lsave=lsave,isave=isave,dsave=dsave,last=last,nfev=nfev,accepted=accepted,elapsed=elapsed+time.monotonic()-start,history_json=json.dumps(history));save_phase(out/f'phase_{accepted:03}.npy',ph);write(out/'history.json',history);write(out/'progress.json',dict(accepted=accepted,target=150,complete=accepted==150,termination=termination,phase_sha256=sha(ph),nfev=nfev,sensor_l2=float(np.sqrt(last[1])),objective=float(last[0]),elapsed_seconds=elapsed+time.monotonic()-start))
        if accepted:checkpoint()
        if accepted>=a.until: print('Already at',accepted);return
        while accepted<a.until:
            _lbfgsb.setulb(m,x,l,u,nbd,f,g,1e-15/np.finfo(float).eps,1e-10,wa,iwa,task,lsave,isave,dsave,20,ln_task)
            if task[0]==3:
                j,grad,d,reg,q=optical.fg(x.reshape(512,512),y,a.alpha);f=np.array(j*1e5,dtype=np.float64);g=np.asarray(grad.ravel()*1e5,dtype=np.float64);last=np.array([j,d,reg,q]);nfev+=1
            elif task[0]==1:
                accepted+=1;row=dict(lbfgs=accepted,objective=float(last[0]),sensor_l2=float(np.sqrt(last[1])),prior=float(last[2]),scale=float(last[3]),nfev=nfev,elapsed_seconds=elapsed+time.monotonic()-start);history.append(row);checkpoint();print('ACCEPT',a.alpha,row,flush=True)
            else:termination=f'optimizer terminated: task={task.tolist()}';break
        if accepted:write(out/'progress.json',dict(accepted=accepted,target=150,complete=accepted==150,termination=termination,phase_sha256=sha(np.mod(x.reshape(512,512),TAU)),nfev=nfev,sensor_l2=float(np.sqrt(last[1])),objective=float(last[0]),elapsed_seconds=elapsed+time.monotonic()-start))
        print('END L-BFGS',accepted,termination,flush=True)
        if accepted<a.until:raise RuntimeError(termination)

def fourier(a):
    """Post-L60 matched circular / edge ablation: restart curvature, preserve the common phase."""
    import torch
    from scipy import fft
    from scipy.optimize import minimize
    from native_model import prior
    r,man,sensor=inputs(a);y=torch.from_numpy(sensor); source=r/'long_alpha_0.0001'; gp=loadj(source/'progress.json');assert gp['accepted']==60
    p0=np.load(source/'phase_060.npy',allow_pickle=False);assert sha(p0)==gp['phase_sha256'];out=r/('fourier_'+a.arm);out.mkdir(exist_ok=False)
    DELTA=.3; ALPHA=1e-4
    def edge(p):
        g=np.zeros_like(p);val=0.
        for axis in (0,1):
            d=np.diff(p,axis=axis);u=1-np.cos(d);s=np.sqrt(1+2*u/DELTA**2);val+=float(np.mean(DELTA**2*(s-1)));v=np.sin(d)/s/d.size
            if axis==0:g[1:]+=v;g[:-1]-=v
            else:g[:,1:]+=v;g[:,:-1]-=v
        return val,g
    with fft.set_workers(4):
        f=fft.fftfreq(512)*512; rr=np.hypot(f[:,None],f[None,:]);mult=1+3*(rr/128)**4/(1+(rr/128)**4)
        def P(x):return fft.ifft2(fft.fft2(x)*mult).real
        _,cg=prior(p0);_,eg=edge(p0);alpha=ALPHA*float(np.linalg.norm(P(eg)))/float(np.linalg.norm(P(cg))) if a.arm=='matched' else ALPHA
        m=model(a,man['wavelength_nm']);history=[];nfev=0;last=None;start=time.monotonic()
        def unpack(x):return p0+P(x.reshape(512,512))
        def fun(x):
            nonlocal nfev,last
            p=unpack(x);_,g,d,rc,q=m.fg(p,y,0); reg,rg=edge(p) if a.arm=='edge' else prior(p);g+=alpha*rg;j=d+alpha*reg;nfev+=1;last=dict(objective=j,data_loss=d,sensor_l2=float(np.sqrt(d)),prior=reg,circular_prior=rc,scale=q,nfev=nfev);return j*1e5,P(g).ravel()*1e5
        def cb(x):
            p=np.mod(unpack(x),TAU);k=len(history)+1;row=dict(step=k,**last,phase_sha256=sha(p),elapsed_seconds=time.monotonic()-start);history.append(row);np.save(out/f'phase_{k:03}.npy',p);write(out/'history.json',history);print('ACCEPT',row,flush=True)
        opt=minimize(fun,np.zeros(p0.size),jac=True,method='L-BFGS-B',callback=cb,options=dict(maxiter=a.steps,maxcor=10,maxls=20,ftol=1e-15,gtol=1e-10));p=np.mod(unpack(opt.x),TAU);np.save(out/'final_phase.npy',p);write(out/'result.json',dict(arm=a.arm,alpha=alpha,accepted=int(opt.nit),evaluations=nfev,termination=str(opt.message),phase_sha256=sha(p),initial_sha256=sha(p0),sensor_sha256=sha(sensor),**last))

def evaluate(a):
    r,man,sensor=inputs(a);p=np.load(a.phase,allow_pickle=False);t=np.load(a.truth,allow_pickle=False)
    if p.shape!=(512,512) or t.shape!=p.shape or sha(t)!=man['truth_sha256']:raise ValueError('Truth/phase mismatch')
    if a.output.exists():raise FileExistsError(a.output)
    d=p-t;e=np.angle(np.exp(1j*d));off=np.angle(np.mean(np.exp(1j*d)))
    from scipy import fft
    with fft.set_workers(4):
        optical=model(a,man['wavelength_nm'])
        predicted=optical.measure(p)
    gain=float(np.sum(predicted*sensor)/np.sum(predicted*predicted))
    sensor_l2=float(np.linalg.norm((gain*predicted-sensor).ravel())/np.linalg.norm(sensor.ravel()))
    result=dict(scene=man['scene'],wavelength_nm=man['wavelength_nm'],phase_sha256=sha(p),truth_sha256=sha(t),sensor_sha256=sha(sensor),circular_rmse=float(np.sqrt(np.mean(e**2))),linear_rmse=float(np.sqrt(np.mean(d**2))),piston_offset=float(off),sensor_l2=sensor_l2,sensor_gain=gain,gd_accepted=int(loadj(r/'progress.json')['gd_accepted']),lbfgs_accepted=int(loadj(r/'long_alpha_0.0001'/'progress.json')['accepted']),gate_pass=dict(circular=float(np.sqrt(np.mean(e**2)))<=.15,sensor=sensor_l2<=.025),truth_used_for_evaluation_only=True)
    preview=a.output.with_suffix('.png')
    try:
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        axes[0].imshow(t,cmap='twilight',vmin=0,vmax=TAU);axes[0].set_title('Ground truth')
        axes[1].imshow(np.mod(p,TAU),cmap='twilight',vmin=0,vmax=TAU);axes[1].set_title('Recovered')
        axes[2].imshow(e,cmap='coolwarm',vmin=-.4,vmax=.4);axes[2].set_title('Circular error')
        for axis in axes: axis.axis('off')
        fig.tight_layout();fig.savefig(preview,dpi=120);plt.close(fig)
        result['preview']=preview.name
    except ImportError:
        result['preview']=None
    write(a.output,result);print(json.dumps(result,indent=2))

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    def cmd(name,func,desc):
        s=sub.add_parser(name,help=desc);s.add_argument('--run',type=Path,required=True);s.add_argument('--archive',type=Path,default=DEFAULT_ARCHIVE);s.set_defaults(func=func);return s
    s=cmd('generate',generate,'Generate known synthetic amplitude from a 512x512 truth NPY');s.add_argument('--truth',type=Path,required=True);s.add_argument('--scene',required=True);s.add_argument('--seed',type=int,required=True);s.add_argument('--wavelength',type=float,default=660.)
    s=cmd('gd',gd,'Resume or run zero-start Pyramid GD150');s.add_argument('--steps',type=int,default=150)
    s=cmd('lbfgs',lbfgs,'Resume full-memory weak circular L-BFGS');s.add_argument('--alpha',type=float,default=1e-4);s.add_argument('--until',type=int,default=60)
    s=cmd('fourier',fourier,'Post-L60 Fourier ablation with restarted curvature');s.add_argument('--arm',choices=['matched','edge','circular'],default='matched');s.add_argument('--steps',type=int,default=20)
    s=cmd('evaluate',evaluate,'Evaluate a saved phase separately from inverse');s.add_argument('--truth',type=Path,required=True);s.add_argument('--phase',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.func(a)
if __name__=='__main__':main()
