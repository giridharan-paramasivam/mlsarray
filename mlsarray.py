import numpy as np
import cupy as cp
from cupyx.scipy.fft import rfft2,irfft2,fft,ifft

class slicelist:
    def __init__(self,Nx,Ny):
        shp=(Nx,Ny)
        pshp=(int(np.ceil((Nx*3/2)/2)*2),int(np.ceil((Ny*3/2)/2)*2))
        insl=[np.s_[0:1,1:int(Ny/2)],np.s_[1:int(Nx/2),:int(Ny/2)],np.s_[-int(Nx/2)+1:,1:int(Ny/2)]]
        shps=[[len(range(*(l[j].indices(shp[j])))) for j in range(len(l))] for l in insl]
        Ns=[np.prod(l) for l in shps]
        outsl=[np.s_[sum(Ns[:l]):sum(Ns[:l])+Ns[l]] for l in range(len(Ns))]
        self.insl,self.shape,self.shps,self.Ns,self.outsl,self.pshp=insl,shp,shps,Ns,outsl,pshp

class mlsarray(cp.ndarray):
    def __new__(cls,Nx,Ny):
        v=cp.zeros((Nx,int(Ny/2)+1),dtype=complex).view(cls)
        return v
    def __getitem__(self,key):
        if(isinstance(key,slicelist)):
            return [cp.ndarray.__getitem__(self,l).ravel() for l in key.insl]
        else:
            return cp.ndarray.__getitem__(self,key)
    def __setitem__(self,key,value):
        if(isinstance(key,slicelist)):
            for l,j,shp in zip(key.insl,key.outsl,key.shps):
                self[l]=value.ravel()[j].reshape(shp)
        else:
            cp.ndarray.__setitem__(self,key,value)

def init_kgrid(sl,Lx,Ly):
    Nx,Ny=sl.shape
    kxl=np.r_[0:int(Nx/2),-int(Nx/2):0]
    kyl=np.r_[0:int(Ny/2+1)]
    dkx,dky=2*np.pi/Lx,2*np.pi/Ly
    kx,ky=np.meshgrid(kxl*dkx,kyl*dky,indexing='ij')
    kx = cp.hstack([cp.asarray(kx[l].ravel()) for l in sl.insl])
    ky = cp.hstack([cp.asarray(ky[l].ravel()) for l in sl.insl])
    return kx,ky

def irft2(uk,sl):
    u=mlsarray(*sl.pshp)
    u[sl]=uk
    Nx=sl.shape[0]
    u[-1:-int(Nx/2):-1,0]=u[1:int(Nx/2),0].conj()
    u.view(dtype=float)[:,:-2]=irfft2(u,norm='forward',overwrite_x=True)
    return u.view(dtype=float)[:,:-2]

def rft2(u,sl):
    uk=rfft2(u,norm='forward',overwrite_x=True).view(type=mlsarray)
    return np.hstack(uk[sl])

def irft(vk,sl):
    Nx=sl.shape[0]
    Npx=sl.pshp[0]
    v = cp.zeros(int(Npx/2)+1, dtype='complex128')
    v[1:int(Nx/2)] = vk[:]
    return cp.fft.irfft(v, norm='forward')

def rft(v,sl):
    Nx=sl.shape[0]
    return cp.fft.rfft(v, norm='forward')[1:int(Nx/2)]

def irft2np(uk,sl):
    uk_cp = cp.asarray(uk)
    result = irft2(uk_cp,sl)
    return cp.asnumpy(result)

def rft2np(u,sl):
    u_cp = cp.asarray(u)
    result = rft2(u_cp,sl)
    return cp.asnumpy(result)

def irftnp(vk,sl):
    vk_cp = cp.asarray(vk)
    result = irft(vk_cp,sl)
    return cp.asnumpy(result)

def rftnp(v,sl):
    v_cp = cp.asarray(v)
    result = rft(v_cp,sl)
    return cp.asnumpy(result)
