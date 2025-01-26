import numpy as np
import Utils as _U

def SepMatrix5(faiXY,order):
    """
    example cp = SepMatrix5(2/5,[01234])
    Args:
        faiXY (_type_): Phase interval
        order (_type_): Phase order

    Returns:
        _type_: cp (3,5,5)
    """
 
    faiPi = faiXY*np.pi
    # faiPiZ = faiZ*np.pi
    ps = faiPi*np.array(order)
    # psz = np.array((faiPiZ,-faiPiZ))
    
    cp = np.array([[1,1,1,1,1],
                   [1,np.exp(2j*ps[1]),np.exp(-2j*ps[1]),np.exp(1j*ps[1]),np.exp(-1j*ps[1])],
                   [1,np.exp(2j*ps[2]),np.exp(-2j*ps[2]),np.exp(1j*ps[2]),np.exp(-1j*ps[2])],
                   [1,np.exp(2j*ps[3]),np.exp(-2j*ps[3]),np.exp(1j*ps[3]),np.exp(-1j*ps[3])],
                   [1,np.exp(2j*ps[4]),np.exp(-2j*ps[4]),np.exp(1j*ps[4]),np.exp(-1j*ps[4])]]) 

    
    return np.repeat(cp[np.newaxis,:,:],3,axis=0)

def Matual(opt,Dr,cp):
    #Dr 4d ； cp 3d
    nc,nz,nx,ny = Dr.shape   #12,1,nx,ny
    # print('Matual-nc,nz,nx,ny',nc,nz,nx,ny)

    Cr = np.zeros((opt.Nangles,opt.Nshifts,nz,nx,ny),dtype=np.complex64) 

    Dr_1D = Dr.reshape(opt.Nangles,opt.Nshifts,-1) 
    print(Cr.shape,Dr_1D.shape)

    for i in range(opt.Nangles):     
        temp = (cp[i] @ Dr_1D[i])  
        Cr[i] = (temp).reshape(opt.Nshifts,nz,nx,ny)  

    return Cr.reshape(nc,nz,nx,ny)

def Matual_self(opt,Dr,cp,Nangles=3,Nshifts=5):
    #Dr 4d ； cp 3d
    nc,nz,nx,ny = Dr.shape   #12,nz,nx,ny
    # print('Matual-nc,nz,nx,ny',nc,nz,nx,ny)

    Cr = np.zeros((Nangles,Nshifts,nz,nx,ny),dtype=np.complex64) 

    Dr_1D = Dr.reshape(Nangles,Nshifts,-1) 
    print(Cr.shape,Dr_1D.shape)

    for i in range(Nangles):   
        temp = (cp[i] @ Dr_1D[i]) 
        Cr[i] = (temp).reshape(Nshifts,nz,nx,ny)  #

    return Cr.reshape(nc,nz,nx,ny)


#Step 1 Separation: Typically 5x5, start with 2nd order separation.
def separate_step_one(opt,Dr,run_parameter,Debug=0,Nshifts=5):
    'get 0 -2,+2,-1,+1'
    assert Dr.shape[0]==15  
    "SepMatri"
    cp = SepMatrix5(opt.faiXY,opt.order5)  #2/5  [01234]
    
    "pinv"
    cp_1 = np.zeros_like(cp)
    for i in range(opt.Nangles):
        cp_1[i] = np.linalg.pinv(cp[i])
        
    "FFT"
    nc = opt.Nangles * 5          #
    assert nc == Dr.shape[0]
    
    Dk = np.zeros_like(Dr).astype(opt.dtype_com)
    for i in range(nc):
        Dk[i] = _U.fftn(Dr[i])

    'save rawdata'
    if Debug==1:
        _U.save_tif_v21(f'RAW-DATA-K—ori123-.TIF',abs(Dk[[0]]),run_parameter)
    
    "bands sep"
    ck = Matual_self(opt,(Dk),cp_1,Nangles=3,Nshifts=Nshifts)   

    return ck

#2x2 Separation
def GETck(opt,ck1,ck2,run_parameter):
    'combine  (-1,-1) (-1,+1) (+1,+1),(+1,-1)'
    _,nz,nx,ny = ck1.shape
    ck = np.zeros((21,nz,nx,ny),dtype=np.complex64)
    for i in range(3):
        for j in range(5):
            if j < 3:   #2-order
                ck[i*7+j]=ck1[i*5+j]    
            elif j==3 : # 1-order 3 4
                ck[i*7+3],ck[i*7+4] = comb1jie(opt,ck1[i*5+j],ck2[i*5+j],opt.faiZ*np.pi)
            elif j==4 : # 1-order 5 6
                ck[i*7+5],ck[i*7+6] = comb1jie(opt,ck1[i*5+j],ck2[i*5+j],opt.faiZ*np.pi)    
    return ck
                

def comb1jie(opt,ckk1,ckk2,fai7XY=0.52*np.pi):
    psz = np.array([0,  fai7XY])
    cp = np.array([[np.exp(1j*psz[0]),np.exp(-1j*psz[0])],
                   [np.exp(1j*psz[1]),np.exp(-1j*psz[1])]]) 
    cp_1 = np.linalg.pinv(cp)
    
    nz,nx,ny = ckk1.shape
    stack_ = np.zeros((2,nz,nx,ny),dtype=opt.dtype_com)
    stack_[0] = ckk1
    stack_[1] = ckk2
    
    #get ck3 ck4/ ck5 ck6
    nc,nz,nx,ny = stack_.shape   #2,8,nx,ny
    
    Dr_1D = stack_.reshape(2,-1) #

    temp = (cp_1 @ Dr_1D)  #  2x2 @ 2x-1
    Ck_1jie = (temp).reshape(2,nz,nx,ny)  

    return Ck_1jie[0],Ck_1jie[1]

    