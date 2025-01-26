import numpy as np
import torch
from tifffile import imread
from pyotf.otf import HanserPSF
import Utils as _U
import WindowF as _WF
'''
raw data read
parameter setting
raw data preprocess
    deconvolution
raw data seperate
'''

def GetParams(opt,Dr_read,Debug=False): # uniform randomisation
    '''
    Optical-related parameters
    '''
    
    #dtype
    opt.dtype_real = np.float32
    opt.dtype_com = np.complex64 
    
    opt.tdtype_real = torch.float32
    opt.tdtype_com = torch.complex64
    
    #device
    opt.gpu_ids = 1 #
    if opt.gpu_ids and torch.cuda.is_available():
        opt.device = torch.device("cuda:0")
    else:
        opt.device = torch.device("cpu")

    opt.kuodaxy = 2       
    if opt.flag_N=="2D":     #
        opt.kuodapz = 1
        opt.Zpadding =  0 
        opt.nz_re = 1  
    elif opt.flag_N=="3D":
        opt.kuodapz = 1
        opt.nz_re = opt.kuodapz*(opt.nz+2*opt.Zpadding)  
        
    elif opt.flag_N=="MP":            
        opt.kuodapz = 2                                 
        opt.nz_re = opt.kuodapz*(opt.nz+2*opt.Zpadding)  
        

    opt.eps = np.finfo(float).eps
    opt.cyclesPerMicron = 1/((opt.nx*opt.dx)*1e-3)     
    
    
    'OTF Cutoff frequency fc'
    opt.fcXYnm = 2*opt.NAjs / opt.Wjs
    opt.fcZnm = opt.NAjs**2 / (2*opt.Wjs)
    #Pixel discretization
    opt.fc = opt.dx*opt.nx * opt.fcXYnm                     # OTF fc-xy
    opt.fc_z = opt.dz*(opt.nz+2*opt.Zpadding) * opt.fcZnm   # OTF fc-z

    'Ir Wave vector P' 
    opt.kxynm = 2*opt.NAzm/opt.Wzm
    opt.kznm  = opt.n*(1-np.cos(opt.theta)) / opt.Wzm
    #Pixel discretization
    opt.pxy = opt.dx*(opt.nx)* opt.kxynm                    # Ir PG-xy
    opt.pz  = opt.dz*(opt.nz+2*opt.Zpadding)* opt.kznm      # Ir PG-z

    opt.fc =  opt.fc 
    opt.pxy = opt.pxy 
    opt.pz = opt.pz 
    
    opt.EEdge = opt.fc+opt.pxy 
    if Debug ==True:
        print('GetParams:opt.fc={},\n opt.fc_z={},\n opt.pxy={},\n opt.pz={}'.format(np.round(opt.fc,2),np.round(opt.fc_z,2),np.round(opt.pxy,2),np.round(opt.pz,2)))
    return opt



#load raw data
def load_RAW_DATA(pathfiname, Debug = False ):
    '''
    load raw data, adjust to Square
    '''

    '1、load raw data'
    Dr_read = imread(pathfiname)
    
    '2、adjust to Square,'
    Dr_read = _U.image_reshape(Dr_read,dim=3)
    nz,nx,ny = Dr_read.shape
    if nx != ny:             
        maxx= max(nx,ny)
        print(f'load_RAW_DATA: 0-padding shape->{nz,nx,ny}->{nz,maxx,maxx}')
        Dr_read_F = _U.pad_or_cut(Dr_read,x=maxx,y=maxx,z=nz)
    else:
        Dr_read_F = Dr_read
    
    assert len(Dr_read_F.shape)==3                    #(nc ,nz,nx,ny)
    
    return Dr_read_F,nx,ny



def Transfer2NPD(opt,Dr_read):
    #transfer (phase,direction,nz)-> (nz,phase,direction)
    nc,nz,nx,ny = opt.Nangles*opt.Nshifts, opt.nz, opt.nx, opt.ny 
    assert len(Dr_read.shape)==3  #3D
    Dr_PAZ = Dr_read.reshape(nz,nc,nx,ny)  #PAZ  == （nz,np,na,...） = (nz,nc,...)
    Dr_ZPA =np.swapaxes( Dr_PAZ,0,1)       
    return Dr_ZPA

def Image_zpadding(opt,Dr_zpd,xishu=0):

    nc,nz,nx,ny = opt.Nangles*opt.Nshifts, opt.nz+2*opt.Zpadding, opt.nx, opt.ny 
    Zp = opt.Zpadding

    Dr_zpadding = np.zeros( (nc,nz,nx,ny),dtype=opt.dtype_real )
    print(f'Dr_zpadding={Dr_zpadding.shape},Dr_zpd={Dr_zpd.shape}')
    for i in range(nc): 
        Dr_zpadding[i,0:Zp]= xishu*Dr_zpd[i,0]  #0 
        Dr_zpadding[i,Zp:nz-Zp] = Dr_zpd[i]
        Dr_zpadding[i,nz-Zp:] = xishu*Dr_zpd[i,-1:None] 
    return Dr_zpadding

def windows(opt,data,nx_raw,ny_raw):
    mask = _WF.mask2_self(nx_raw,ny_raw,sigma=0.25)
    nc,nz,nx,ny = data.shape
    if nx!=nx_raw or ny!=ny_raw:
        mask = _U.pad_or_cut(mask,x=nx,y=ny)
    print(data.shape,mask.shape)
    data = data*(mask**9)
    return data


#############################loadANDprocess_Dr################################################
def loadANDprocess_Dr(opt,p_i,run_parameter):
    "load Dr4D_ZPA"
    # 1、load raw data, 、adjust ,  #return 3D(nz,nx,ny)
    Dr3D_square,nx_raw,ny_raw = load_RAW_DATA(p_i)   
    
    # 2 get opt
    opt = GetParams(opt,Dr3D_square)
    
    #'3、PAZ - > ZPA'           # 3 transfer (phase,direction,nz)-> (nz,phase,direction) # return 4D(nc,nz,nx,ny)
    Dr4D_ZPA = Transfer2NPD(opt,Dr3D_square)  
    
    #'4、windows'
    Dr4D_ZPA = windows(opt,Dr4D_ZPA,nx_raw,ny_raw)
    
    X_stare = run_parameter['movex']
    X_long = run_parameter['movex_long']
    Y_stare = run_parameter['movey']
    Y_long = run_parameter['movex_long']
    
    Dr4D_ZPA_crop =  (Dr4D_ZPA[:,:,X_stare:X_stare+X_long,Y_stare:Y_stare+Y_long])
    
    Dr4D_ZPA_crop_p = _U.pad_or_cut(Dr4D_ZPA_crop,opt.nx,opt.ny,opt.nz)
    
    
    Dr4D_ZPA = Dr4D_ZPA.astype(opt.dtype_real)
    Dr4D_ZPA_crop_p = Dr4D_ZPA_crop_p.astype(opt.dtype_real)
    assert len(Dr4D_ZPA.shape)==4
    return Dr4D_ZPA_crop_p,opt


###############OTF/OTFmask###########################

def load_OTF(opt,pathfiname,Debug=False):
    '''
    1、load OTF
    2、normalize
    3、get 2D
    '''
    
    '1、load OTF '
    OTFload = imread(pathfiname)
    nz,nx,ny = OTFload.shape
    assert nx==opt.nx and ny == opt.ny    #asure shape 
    
    PSF = (np.fft.fftshift(_U.ifftn(OTFload)))
    PSF[PSF<0]=0
    OTF = OTFload 
    
    '2、normalize'
    OTF_n = _U.normalize(OTF)
    
    '3、get 2D'
    OTF_n_3D = _U.image_reshape(OTF_n,dim=3)  #
    nz,nx,ny = OTF_n_3D.shape
    OTF_n_2D = OTF_n_3D[nz//2]
    
    return OTF_n_3D,OTF_n_2D,PSF  

def otf_mask_formula(opt,nx,ny,nz,PGxy=[0,0],PGz=[0],Debug=0,kuodaxy=1):
    """
    """
    try:
        NA = opt.na_mask
        refmed = opt.n_mask
        NAmin = min(NA,refmed)
    except:
        NA = opt.na
        refmed = opt.n
        NAmin = opt.namin
    
    lamdex = opt.Wzm
    lamdem = opt.Wjs
    theta = opt.theta
    dx = opt.dx          
    dy = opt.dx
    
    padd_scale = 1
    piexl_scale=1       

    if opt.nz==1:   
        nz = 2
    DqxSupport = 1 / ((nx*padd_scale)*(dx*piexl_scale))        
    DqySupport = 1 / ((ny*padd_scale)*(dx*piexl_scale))
    DqzSupport = 1 / (nz*opt.dz)                                                       
    QXSupport = torch.arange(np.ceil(-nx),np.ceil(nx))*DqxSupport;    
    QYSupport = torch.arange(np.ceil(-ny),np.ceil(ny))*DqySupport;    
    QZSupport = torch.arange(np.ceil(-nz/2),np.ceil(nz/2))*DqzSupport; 
    [qz,qx,qy] = torch.meshgrid(QZSupport,QXSupport,QYSupport,indexing='ij')
    

    q0 = refmed/lamdem    #
    NA1 = (NAmin/lamdem)   #
    NB1 =  np.sqrt(q0**2-NA1**2)

    qpar = torch.sqrt( (qx-PGxy[0]*DqxSupport)**2+(qy-PGxy[1]*DqxSupport)**2)  #
    axialcutoff = torch.sqrt(q0**2-(qpar-NA1)**2)-NB1;
    axialcutoff = (qpar<=2*NA1)*axialcutoff;

    otf_mask = (axialcutoff+DqzSupport/2 >= abs(qz-PGz[0]*DqzSupport))
    otf_mask = (qpar<=2*NA1)*otf_mask
    otf_mask = otf_mask.to(torch.uint8)
    otf_mask = otf_mask.numpy()

    
    if opt.nz==1:   
        nz,nx,ny = otf_mask.shape
        maxindex=0
        maxpp = np.mean(otf_mask[maxindex])
        for i in range(nz):
            tempp = np.mean(otf_mask[i])
            if maxpp<tempp:
                maxindex = i
                maxpp = np.mean(otf_mask[maxindex])
        otf_mask = otf_mask[maxindex]

    if Debug == 1:   
        _U.save_tif('④otf_mask--mask0.tif',otf_mask.astype(bool))
        pass
    return _U.image_reshape(otf_mask,dim=3)  

def otf_mask_formula2(opt,nx,ny,nz,PGxy=[0,0],PGz=[0],Debug=0,kuodaxy=1):
    try:
        NA = opt.na_mask
        refmed = opt.n_mask
        NAmin = min(NA,refmed)
    except:
        NA = opt.na
        refmed = opt.n
        NAmin = opt.namin
    
    lamdex = opt.Wzm
    lamdem = opt.Wjs
    theta = opt.theta
    dx = opt.dx          
    dy = opt.dx

    padd_scale = 1
    piexl_scale=1       

    if opt.nz==1:   
        nz = 2
    DqxSupport = 1 / ((nx*padd_scale)*(dx*piexl_scale))        
    DqySupport = 1 / ((ny*padd_scale)*(dx*piexl_scale))
    DqzSupport = 1 / (nz*opt.dz)                                                      
    QXSupport = torch.arange(np.ceil(-nx),np.ceil(nx))*DqxSupport;    
    QYSupport = torch.arange(np.ceil(-ny),np.ceil(ny))*DqySupport;    

    QZSupport = torch.arange(np.ceil(-nz),np.ceil(nz))*DqzSupport;     
    [qz,qx,qy] = torch.meshgrid(QZSupport,QXSupport,QYSupport,indexing='ij')
    
    
    q0 = refmed/lamdem    
    NA1 = (NAmin/lamdem)   
    NB1 =  np.sqrt(q0**2-NA1**2)

    qpar = torch.sqrt( (qx-PGxy[0]*DqxSupport)**2+(qy-PGxy[1]*DqxSupport)**2)  
    axialcutoff = torch.sqrt(q0**2-(qpar-NA1)**2)-NB1;
    axialcutoff = (qpar<=2*NA1)*axialcutoff;

    otf_mask = (axialcutoff+DqzSupport/2 >= abs(qz-PGz[0]*DqzSupport))
    otf_mask = (qpar<=2*NA1)*otf_mask
    otf_mask = otf_mask.to(torch.uint8)
    otf_mask = otf_mask.numpy()

    if opt.nz==1:   
        nz,nx,ny = otf_mask.shape
        maxindex=0
        maxpp = np.mean(otf_mask[maxindex])
        for i in range(nz):
            tempp = np.mean(otf_mask[i])
            if maxpp<tempp:
                maxindex = i
                maxpp = np.mean(otf_mask[maxindex])
        otf_mask = otf_mask[maxindex]

    if Debug == 1:   
        _U.save_tif('④otf_mask--mask0.tif',otf_mask.astype(bool))
        pass
    return _U.image_reshape(otf_mask,dim=3)  



def otf_mask_formula3(opt,nx,ny,nz,PGxy=[0,0],PGz=[0],piexl_scale=1,piexl_scalez=1,Debug=0):
    """
    """
    try:
        NA = opt.na_mask
        refmed = opt.n_mask
        NAmin = min(NA,refmed)
    except:
        NA = opt.na
        refmed = opt.n
        NAmin = opt.namin
    
    lamdex = opt.Wzm  
    lamdem = opt.Wjs  
    theta = opt.theta
    dx = opt.dx          
    dy = opt.dx

    DqxSupport = 1 / ((nx)*(dx*piexl_scale))        
    DqySupport = 1 / ((ny)*(dx*piexl_scale))
    DqzSupport = 1 / ( nz*opt.dz*piexl_scalez )                                             
    QXSupport = torch.arange(np.ceil(-nx/2),np.ceil(nx/2))*DqxSupport
    QYSupport = torch.arange(np.ceil(-ny/2),np.ceil(ny/2))*DqySupport
    QZSupport = torch.arange(np.ceil(-nz/2),np.ceil(nz/2))*DqzSupport

    [qz,qx,qy] = torch.meshgrid(QZSupport,QXSupport,QYSupport,indexing='ij')
    
    
    q0 = refmed/lamdem    #
    NA1 = (NAmin/lamdem)   #
    NB1 =  np.sqrt(q0**2-NA1**2)

    qpar = torch.sqrt( (qx-PGxy[0]*DqxSupport)**2+(qy-PGxy[1]*DqxSupport)**2)  #
    axialcutoff = torch.sqrt(q0**2-(qpar-NA1)**2)-NB1;
    axialcutoff = (qpar<=2*NA1)*axialcutoff;

    otf_mask = (axialcutoff+DqzSupport/2 >= abs(qz-PGz[0]*DqzSupport))
    otf_mask = (qpar<=2*NA1)*otf_mask
    otf_mask = otf_mask.to(torch.uint8)
    otf_mask = otf_mask.numpy()
    
    if opt.nz==1:   #
        nz,nx,ny = otf_mask.shape
        maxindex=0
        maxpp = np.mean(otf_mask[maxindex])
        for i in range(nz):
            tempp = np.mean(otf_mask[i])
            if maxpp<tempp:
                maxindex = i
                maxpp = np.mean(otf_mask[maxindex])
        otf_mask = otf_mask[maxindex]

    return _U.image_reshape(otf_mask,dim=3)  



def otf_mask_3d_2d(opt,Debug,kuodapz=2,PGxy=[0,0],PGz=[0]):
    if kuodapz==1:
        OTFmask3D_raw = otf_mask_formula(opt,nx=opt.nx, ny=opt.ny, nz=opt.nz, PGxy=PGxy,PGz=PGz,Debug=Debug,kuodaxy=1) 
    elif kuodapz ==2:
        OTFmask3D_raw = otf_mask_formula2(opt,nx=opt.nx, ny=opt.ny, nz=opt.nz, PGxy=PGxy,PGz=PGz,Debug=Debug,kuodaxy=1) 
    
    OTFmask3D = np.zeros_like(OTFmask3D_raw)

    nz,nx,ny = OTFmask3D_raw.shape
    nz_med = nz//2
    OTFmask3D[nz_med:] = OTFmask3D_raw[nz_med:]
    for ii in range(nz_med):
        OTFmask3D[ii+1] = OTFmask3D_raw[-(ii+1)]  #1 -1;   2 -2  ;3  -3;  
     
    OTFmask3D = _U.pad_or_cut(OTFmask3D,x=2*opt.nx,y=2*opt.ny,z = opt.kuodapz*opt.nz) #

    if opt.Zpadding!=0:
        nx,ny,nz =  opt.kuodaxy*opt.nx, opt.kuodaxy*opt.ny, opt.nz_re
        OTFmask3D = _U.Interpolate(OTFmask3D.astype(np.float32),nx,ny,nz)
        OTFmask3D[OTFmask3D < 0.1] = 0
        OTFmask3D[OTFmask3D != 0 ] = 1
        OTFmask3D = OTFmask3D.astype(np.uint8)

    #OTFmask2D
    OTFmask2D = OTFmask3D[opt.nz_re//2]
    
    return OTFmask3D,OTFmask2D,OTFmask3D_raw


def otf_mask_PE(opt,Debug,PGxy=[0,0],PGz=[0],flag=0):
    if flag ==0: 
        OTFmask3D_raw = otf_mask_formula3(opt,nx=opt.nx, ny=opt.ny, nz=opt.nz+2*opt.Zpadding, PGxy=PGxy,PGz=PGz,piexl_scale=1,piexl_scalez= 1 ,Debug=Debug) 
    else:
        OTFmask3D_raw = otf_mask_formula3(opt,nx=2*opt.nx, ny=2*opt.ny, nz=opt.nz_re, PGxy=PGxy,PGz=PGz,piexl_scale=0.5,piexl_scalez= 0.5 ,Debug=Debug) 

    if PGz[0]==0:
        OTFmask3D_raw = duicehng(OTFmask3D_raw)
    
    #OTFmask2D
    nz,_,_ = OTFmask3D_raw.shape
    OTFmask2D = OTFmask3D_raw[nz//2]

    return OTFmask3D_raw,OTFmask2D

def duicehng(data):
    assert(len(data.shape)==3)
    nz,_,_ = data.shape
    nz_med = nz//2
    data_out = np.zeros_like(data)
    for i in range(nz-nz_med): #6,12
        data_out[nz_med-i] = np.copy(data[nz_med+i]) 
        data_out[nz_med+i] = np.copy(data[nz_med+i]) 
    return data_out



def drawDr(Dr,Nshift=8,number=[0,1,2,3,4]):
    Drout = np.zeros_like(Dr[:3*len(number)])
    for i in range(3):
        ii = (i)*Nshift + np.array(number) #[0,1,2,3,4]
        Drout[i*len(number):(i+1)*len(number)] = np.copy(Dr[[ii]]) #
    return Drout


def generatemask(opt,isdebug):
    # # OTFmask3D/2D
    opt.na_mask = 1.21
    OTFmask3D_121,OTFmask2D = otf_mask_PE(opt,Debug=isdebug['loa']) #
    opt.na_mask = 1.3
    OTFmask3D_13,_ = otf_mask_PE(opt,Debug=isdebug['loa']) #
    center_x, center_y = OTFmask3D_13.shape[1] // 2, OTFmask3D_13.shape[2] // 2
    y, x = np.ogrid[:OTFmask3D_13.shape[1], :OTFmask3D_13.shape[2]]
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    maskotf44 = OTFmask3D_13-OTFmask3D_121
    maskotf44[int(maskotf44.shape[0]/2-opt.pz/1):int(maskotf44.shape[0]/2+opt.pz/1)+2,distance_from_center<30] = 0
    OTFmask3D = OTFmask3D_121+maskotf44
    OTFmask3D[OTFmask3D.shape[0]//2+1:] = np.flip(OTFmask3D[1:OTFmask3D.shape[0]//2], axis=0)
    OTFmask3D[OTFmask3D>1] =1

    return OTFmask3D, OTFmask2D