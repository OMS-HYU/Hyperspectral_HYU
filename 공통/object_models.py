"""Native-grid synthetic object models. Truth belongs to generation/evaluation only."""
from pathlib import Path
import numpy as np
from scipy import ndimage
TAU=2*np.pi
N=512
PITCH_UM=.35
C=(.207161,2.002558,-1.122762)
RBC_MODELS=("normal","variable","tilt","double","triple")
EXTRA_IMAGE_MODELS=("lena","barbara","house","starfish","monarch","airplane","parrot","man","couple","camera_set12","pepper_set12","boat_set12","mandrill","aerial","bark","brick_wall","electron_micrograph","motioncar","plane_c4l","random_texture","sand","stream","boat_c4l","tank_c4l")
IMAGE_MODELS=("camera","camera_rot90","moon","brick","chelsea","coins","checkerboard","astronaut","boat","tank","pepper","custom")+EXTRA_IMAGE_MODELS
SOURCES={
 "shape":"Evans & Fung 1972, doi:10.1016/0026-2862(72)90069-6; Dao et al. 2003, Eq.7a,b, doi:10.1016/j.jmps.2003.09.019",
 "index":"Ahmad et al. 2020, doi:10.1117/1.JBO.25.11.116501 (RBC 1.40, PBS 1.33)",
 "hemoglobin":"Park et al. 2009, doi:10.1364/OL.34.003668 (660 nm alpha=.221 mL/g); Park et al. 2008, doi:10.1073/pnas.0806100105 (30.9 +/- 3.1 g/dL)"
}
def wrap(x):
    return np.mod(np.asarray(x,dtype=np.float64),TAU)
IMAGE_DIR=Path(__file__).resolve().parents[1]/"데이터"/"입력이미지"
IMAGE_FILES={"camera":"camera.png","camera_rot90":"camera.png","chelsea":"chelsea.png",
             "checkerboard":"checkerboard.png","moon":"moon.png","brick":"brick.png",
             "coins":"coins.png","astronaut":"astronaut.png","boat":"boat.jpg",
             "tank":"tank.png","pepper":"pepper.png"}
IMAGE_FILES.update({name:name+".png" for name in EXTRA_IMAGE_MODELS})
def image_phase(name,path=None):
    from skimage import color,transform,io,util
    if name not in IMAGE_MODELS: raise ValueError(name)
    if name=="custom":
        if path is None: raise ValueError("This image requires --image-file PATH")
        a=io.imread(path)
    elif name in ("boat","tank","pepper") and path is not None:
        a=io.imread(path)
    else:
        a=io.imread(IMAGE_DIR/IMAGE_FILES[name])
        if name=="camera_rot90": a=np.rot90(a)
    if a.ndim==3:
        if a.shape[-1]==4: a=color.rgba2rgb(a)
        a=color.rgb2gray(a)
    else: a=util.img_as_float(a)
    if a.ndim!=2: raise ValueError("Expected a grayscale/RGB image")
    if a.shape!=(N,N):
        a=transform.resize(a,(N,N),order=3,anti_aliasing=True,preserve_range=True)
    a=np.clip(np.rint(a*255),0,255)
    return wrap(a*(TAU/255))
def rbc_phase(kind,wavelength_nm,seed=20261127):
    """Full 512x512 transverse sampling; projected 3D thickness, not scattering.
    Hb alpha is a 660 nm literature value. For other wavelengths, constant
    index contrast is an explicitly nondispersive approximation, not measured dispersion.
    """
    if kind not in RBC_MODELS: raise ValueError(kind)
    rng=np.random.default_rng(seed)
    if kind=="triple":
        sites=[]
        for _ in range(100000):
            q=rng.uniform(-78,78,2)
            if all(np.linalg.norm(q-p)>=14 for p in sites): sites.append(q)
            if len(sites)==50: break
        if len(sites)!=50: raise RuntimeError("Could not place 50 groups")
        counts=[1]*20+[2]*10+[3]*20
    else:
        counts=[1]*100 if kind!="double" else [1]*50+[2]*25
        axis=np.linspace(-77,77,10 if len(counts)==100 else 9)
        sites=np.array([(x,y) for y in axis for x in axis])[:len(counts)]
        sites=sites+rng.uniform(-.7,.7,sites.shape)
    centers=[];group=[]
    for j,(site,n) in enumerate(zip(sites,counts)):
        a=rng.uniform(0,TAU); direction=np.array([np.cos(a),np.sin(a)])
        sep=0. if n==1 or j%2==0 else rng.uniform(1,2.5)
        for k in range(n):
            centers.append(site+(k-(n-1)/2)*sep*direction);group.append(j+1)
    group=np.asarray(group); centers=np.asarray(centers)
    rng=np.random.default_rng(seed+1)
    diam=np.full(100,7.82) if kind=="normal" else rng.uniform(7,8.5,100)
    hb=rng.uniform(27.8,34,100)
    dn=np.full(100,.07) if kind in ("normal","variable") else .221*hb/100
    tilt=np.zeros(100) if kind in ("normal","variable") else rng.uniform(0,45,100)
    az=rng.uniform(0,TAU,100)
    if kind=="triple":
        stack=group==31; diam[stack]=8.5;dn[stack]=.221*.34;tilt[stack]=45;az[stack]=.7
    coords=(np.arange(N)-255.5)*PITCH_UM
    opd=np.zeros((N,N));labels=np.zeros((N,N),np.int16);groups=np.zeros_like(labels)
    z=np.arange(-4.5+.005,4.5,.01)[None,None,:]
    for i,((cx,cy),d,th,a) in enumerate(zip(centers,diam,tilt,az)):
        ix=np.flatnonzero(abs(coords-cx)<=4.5);iy=np.flatnonzero(abs(coords-cy)<=4.5)
        yy,xx=np.meshgrid(coords[iy]-cy,coords[ix]-cx,indexing="ij")
        u=(xx*np.cos(a)+yy*np.sin(a))[...,None];v=(-xx*np.sin(a)+yy*np.cos(a))[...,None]
        r=d/2;theta=np.deg2rad(th)
        xp=u*np.cos(theta)-z*np.sin(theta);zp=u*np.sin(theta)+z*np.cos(theta)
        s=(xp*xp+v*v)/r**2
        half=.5*r*np.sqrt(np.maximum(1-s,0))*(C[0]+C[1]*s+C[2]*s*s)
        local=((s<=1)&(np.abs(zp)<=half)).sum(axis=2)*.01
        sl=np.ix_(iy,ix);mask=local>0
        if np.any((groups[sl][mask]!=0)&(groups[sl][mask]!=group[i])):
            raise RuntimeError("Unexpected overlap between independent groups")
        groups[sl]=np.where(mask,group[i],groups[sl]);labels[sl]+=mask
        opd[sl]+=dn[i]*local
    return {"phase":wrap(TAU*opd/(wavelength_nm/1000)),
            "labels":labels,"groupmap":groups,"opd_um":opd,
            "diameter_um":diam,"delta_n":dn,"tilt_deg":tilt,"centers_um":centers}
