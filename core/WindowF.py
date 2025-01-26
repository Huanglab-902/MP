import numpy as np
import torch
import math
import Utils as _U


def mask2(opt,kuodaxy=1,sigma = 0.25,padsize=0):
    """windows """
    ny = opt.nx*kuodaxy
    nx = opt.ny*kuodaxy
    nz = opt.nz

    x = torch.arange(1,nx+1).reshape(1,-1)
    y = torch.arange(1,ny+1).reshape(-1,1)

    sigma=sigma;
    mask2d = ((torch.sigmoid(sigma*(x))-torch.sigmoid(sigma*(x-ny-1))).repeat(nx,1)
                * (torch.sigmoid(sigma*(y))-torch.sigmoid(sigma*(y-nx-1))).repeat(1,ny))

    return mask2d.numpy();


def mask2_self(nx,ny,kuodaxy=1,sigma = 0.25,padsize=0):
    """windows set nx ny"""

    x = torch.arange(1,nx+1).reshape(-1,1)
    y = torch.arange(1,ny+1).reshape(1,-1)

    sigma=sigma;
    mask2d = ((torch.sigmoid(sigma*(x))-torch.sigmoid(sigma*(x-nx-1)))
                * (torch.sigmoid(sigma*(y))-torch.sigmoid(sigma*(y-ny-1))))

    return mask2d.numpy();


def bhs_3D_Z(opt, plong,fc,kuodaxy=2,kuodaz=1,Debug=0,Zjust = 300):
    """
    Apo
    """
    nx = opt.nx*kuodaxy
    ny = opt.ny*kuodaxy
    nz = (opt.nz+2*opt.Zpadding)*kuodaz
    NA = opt.na
    pixelsizeZ =  opt.dz/(2*Zjust) #
    wavelength = opt.Wjs/1000  #

    zv,xv,yv = _U.meshgrid(opt,kuodaxy=kuodaxy,kuodaz=kuodaz)

    dpz = 1/(nz*pixelsizeZ)
    k_xymax = plong+fc
    k_zmax = 2* ((NA**2/(2*wavelength))/dpz)
    k_xy = np.sqrt(xv**2 + yv**2)
    k_z = np.sqrt(zv**2);
    rhox = abs(xv)/(2*k_xymax)
    rhoy = abs(yv)/(2*k_xymax)
    rhoz = abs(zv)/(2*k_zmax)
    rho = np.sqrt(rhox*rhox + rhoy*rhoy + rhoz*rhoz)
    apodiz = np.cos(np.pi*rho)
    indi_xy = k_xy > k_xymax
    indi_z = k_z > k_zmax
    msk = indi_xy|indi_z
    apodiz[msk==1] = 0
    apodiz[apodiz<0] = 0
    apodiz = apodiz**2
    if Debug==1:
        _U.save_tif('apodiz1.tif',apodiz)
    return apodiz


#################notch_filter############################
def valAttenuation(opt,px=0,py=0,pz=0,Astr=1,fwhm=1):
    nz,nx,ny = (opt.nz_re),opt.nx*2,opt.ny*2
    # nz = 16
    x = np.arange(np.ceil(-nx/2),np.ceil(nx/2))
    y = np.arange(np.ceil(-ny/2),np.ceil(ny/2))
    z = np.arange(np.ceil(-nz/2),np.ceil(nz/2))
    zv, xv, yv = np.meshgrid(z, x, y, indexing='ij', sparse=True)
    
    rad1 = np.sqrt( (xv-px) **2 + (yv-py)**2)*opt.cyclesPerMicron   # 0.0248

    va=1-Astr*(np.exp(-np.power(rad1,2)/(np.power(0.5*fwhm,2))));
    if opt.flag_N==3:
        return 1-(1-va)

    rad2 = np.sqrt( (zv-pz)**2)*opt.cyclesPerMicron
    rad2= rad2/np.max(rad2)
    return 1-(1-va)#*(1-rad2)


def getotfAtt(opt,px=0,py=0,pz=0,Astr=1,fwhm=1):    
    nz,nx,ny = (opt.nz_re),opt.nx*2,opt.ny*2
    x = np.arange(np.ceil(-nx/2),np.ceil(nx/2))
    y = np.arange(np.ceil(-ny/2),np.ceil(ny/2))
    z = np.arange(np.ceil(-nz/2),np.ceil(nz/2))
    zv, xv, yv = np.meshgrid(z, x, y, indexing='ij', sparse=True)
    
    rad1 = np.sqrt( (xv-px) **2 + (yv-py)**2) *opt.cyclesPerMicron   # 0.0248

    rad2 = np.sqrt( (zv-pz)**2)*opt.cyclesPerMicron
    rad2= rad2/np.max(rad2)

    va=1-Astr*(np.exp(-np.power(rad1,4)/(2*np.power(fwhm,4)))); 
    zATT = build_gaussian_z(opt,z=nz,x=nx,y=ny)
      
    return 1-(1-va)*(1-rad2)

def build_gaussian_z(opt,z,x,y,pz=0,mean=0,std=1,lenz=2):
    nz,nx,ny = z,x,y 
    len = lenz
    x = np.linspace(-len,len, nx)
    y = np.linspace(-len, len, ny);
    z = np.linspace(-len+pz, len+pz, nz);

    x,z,y= np.meshgrid(x,z,y) 
    z = np.exp(-((z-mean)**2) /(2*(std**2)))  #
    z = z/(math.pow((2*np.pi)*std,2))   #
    z = z/np.max(z)
    return z;