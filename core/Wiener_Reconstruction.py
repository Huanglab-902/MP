import numpy as np
import torch 
import Utils as _U
import WindowF as _WF


fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftn = np.fft.fftn
ifftn = np.fft.ifftn
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift

tfft2 = torch.fft.fft2
tifft2 = torch.fft.ifft2
tfftn = torch.fft.fftn
tifftn = torch.fft.ifftn
tfftshift = torch.fft.fftshift
tifftshift = torch.fft.ifftshift


def IrTEST(opt,PGxy,PGz,kuodaxy=2,kuodaz=2,Zchushu=0,pz_true=0):
    '''
    generate move phase matrix"
    '''
    nz=(opt.nz+2*opt.Zpadding)*kuodaz
    nx=opt.nx*kuodaxy
    ny=opt.ny*kuodaxy
    ZV,XV,YV = _U.meshgrid(opt,kuodaxy=kuodaxy,kuodaz=kuodaz)
    Irtest = []
    # Irtest_1 = []
    for i in range(opt.Nangles):
        Pxyabs = PGxy[i]/nx
        Pzabs = PGz[i]/nz 
        for j in range(opt.Norders):#7   0  -2  2  (-1,-1)  (-1,1） （1，-1） （1，1）
            Pxy = Pxyabs[j]        
            Pz =  Pzabs[j]
            shiftmat = np.exp(2j*np.pi*(Pxy[0]*XV+Pxy[1]*YV + Pz*ZV),dtype=opt.dtype_com) 
            Irtest.append(shiftmat)
    return Irtest

def choose_jiesu(opt,flagg):
    if flagg==0:              #0
        jiesu=[0]

    elif flagg==1 or flagg==3:  # ±1 or 0±1              
        if opt.Norders==5:       ##slice ±1
            jiesu=[3,4]
        elif opt.flag_N == 'MP' and opt.Norders==7:   ##mp ±1
            jiesu=[3,4,5,6]
        else:
            raise 'Order error: no first order'
        if flagg ==3:
            jiesu.insert(0,0)

    elif flagg == 2:            #0 ±2
        jiesu = [1,2]
    elif flagg ==4 :
        jiesu = [0,1,2]

    elif flagg == 5 or flagg==6:  #±1
        if opt.Norders==5:       ##slice ±1
            jiesu=[1,2,3,4]
        elif opt.flag_N == 'MP' and opt.Norders==7 :   ##mp ±1
            jiesu=[1,2,3,4,5,6]

        elif opt.Norders==3 and opt.flag_N == '2D':
            jiesu = [1,2]

        if flagg ==6:
            jiesu.insert(0,0)
            
    return jiesu

def removefai(opt,ck,cfai):
    #Eliminate the initial phase of ck
    assert len(ck.shape)==4
    sk = np.zeros_like(ck)
    nc,nz,nx,ny = ck.shape
    Norders = nc//opt.Nangles
    for i in range(opt.Nangles):
        for j in range(Norders):
            if j ==0:
                assert cfai[i,j,0]==1 and cfai[i,j,1]==0 
                
            sk[i*Norders+j] = ck[i*Norders+j]/( np.exp(1j* cfai[i,j,1] ))  
    return sk
            
            

##########################
def GetFZotf(opt,Irtest,order_flag,run_parameter,PGxy,PGz,direction=[0,1,2]):
    """
    Read opt.OTF, move OTF based on Irtest
    """
    
    nx = opt.nx*opt.kuodaxy
    ny = opt.ny*opt.kuodaxy
    nz = opt.nz_re          
    Norders=opt.Norders                #5
    nc = opt.nc

    "1 OTF to psf"
    if run_parameter['meas'] ==1:
        pass
    else:
        opt.OTF = opt.OTF * opt.OTFmask3D 
        
    OTF3D = _U.ToTensor(opt.OTF).to(device=opt.device)  
    PSF3D = tfftshift(_U.tifftn(OTF3D)).real         
    OTF3D = OTF3D.cpu()  
    

    if torch.is_complex(OTF3D):
        dtype = opt.tdtype_com
    else:
        dtype = opt.tdtype_real
    
    FZotf = torch.zeros((nc,nz,nx,ny),dtype=dtype)     #
    FZotfmask = np.zeros((nc,nz,nx,ny),dtype=np.bool_)
    
    jiesu = choose_jiesu(opt,order_flag)

    for i in direction:    
        for j in (jiesu):  
            if j == 0:
                FZotf[i*Norders+j] = OTF3D
            else:
                "Moving Matrix"
                filename = str(i)+str(j)+'.tif'
                tIrtest =_U.ToTensor(Irtest[i*Norders+j]).to(device=opt.device)        
 
                'Moving OTF'
                OTF_move = abs(_U.tfftn(tifftshift (PSF3D*tIrtest) )).cpu()           
                OTF_move = _U.normalize(OTF_move)

                'FZotf'
                FZotf[i*Norders+j] = (OTF_move)

    return FZotf,FZotfmask


def GetFZ(opt,ck,Irtest,order_flag,run_parameter,direction=[0,1,2]):
    if opt.device != 'cpu':
        torch.cuda.empty_cache()

    nx = opt.nx*opt.kuodaxy                
    ny = opt.ny*opt.kuodaxy                 
    nz = opt.nz_re                
    nc = opt.nc
    Norders=opt.Norders

    ck = _U.ToTensor(ck)
    FZall =torch.zeros((nc,nz,nx,ny),dtype=torch.complex64)                            


    jiesu = choose_jiesu(opt,order_flag)
    for i in direction:
        for j in jiesu:
            if j==0:
              ck_pad = _U.pad_or_cut(ck[i*Norders+j],x=nx,y=ny,z=nz)   
              FZall[i*Norders+j] = ck_pad
            else:
                ck_pad = _U.pad_or_cut(ck[i*Norders+j],x=nx,y=ny,z=nz)
                
                'Moving Matrix'
                filename = str(i)+str(j)
                tIrtest = _U.ToTensor(Irtest[i*Norders+j]).to(device=opt.device)   

                'Moving ck_pad'
                ck_pad_GPU = ck_pad.to(device=opt.device)                     
                ck_pad_move = _U.tfftn(_U.tifftn(ck_pad_GPU)*tIrtest).cpu()                    
                FZall[i*Norders+j] = ck_pad_move
    
    return FZall


def GetFZ_Joint(opt,ck,ck_02,Irtest,order_flag,run_parameter,direction=[0,1,2]):
    if opt.device != 'cpu':
        torch.cuda.empty_cache()

    nx = opt.nx*opt.kuodaxy                
    ny = opt.ny*opt.kuodaxy                 
    nz = opt.nz_re                 
    nc = opt.nc                  
    Norders=opt.Norders            

    ck = _U.ToTensor(ck)
    ck_02 = _U.ToTensor(ck_02)
    
    FZall =torch.zeros((nc,nz,nx,ny),dtype=torch.complex64)                    
    FZall_02 = torch.zeros((3*3,nz,nx,ny),dtype=torch.complex64)

    jiesu = choose_jiesu(opt,order_flag)
    for i in direction:
        for j in jiesu:
            if j==0:
                ck_pad = _U.pad_or_cut(ck[i*Norders+j],x=nx,y=ny,z=nz)   
                FZall[i*Norders+j] = ck_pad 
                del ck_pad
                
                ck_02_pad = _U.pad_or_cut(ck_02[i*3+0],x=nx,y=ny,z=nz)   
                FZall_02[i*3+0] = ck_02_pad 
                del ck_02_pad
                
            elif j <3:   
                ck_pad = _U.pad_or_cut(ck[i*Norders+j],x=nx,y=ny,z=nz)
                ck_02_pad = _U.pad_or_cut(ck_02[i*3+j],x=nx,y=ny,z=nz)  
                
                'Moving Matrix'
                filename = str(i)+str(j)
                tIrtest = _U.ToTensor(Irtest[i*Norders+j]).to(device=opt.device)   

                'Moving ck_pad'
                ck_pad_GPU = ck_pad.to(device=opt.device)                     
                ck_pad_move = _U.tfftn(_U.tifftn(ck_pad_GPU)*tIrtest).cpu()                   
                FZall[i*Norders+j] = ck_pad_move
                
                del ck_pad_GPU,ck_pad_move
                
                
                ck_02_pad_GPU = ck_02_pad.to(device=opt.device)                     
                ck_02_pad_move = _U.tfftn(_U.tifftn(ck_02_pad_GPU)*tIrtest).cpu()              
                FZall_02[i*3+j] = ck_02_pad_move
                
                del ck_02_pad_GPU,ck_02_pad_move
                
            else:
                ck_pad = _U.pad_or_cut(ck[i*opt.Norders+j],x=nx,y=ny,z=nz)
                
                'Moving Matrix'
                filename = str(i)+str(j)
                tIrtest = _U.ToTensor(Irtest[i*opt.Norders+j]).to(device=opt.device)   

                'Moving ck_pad'
                ck_pad_GPU = ck_pad.to(device=opt.device)                     
                ck_pad_move = _U.tfftn(_U.tifftn(ck_pad_GPU)*tIrtest).cpu()                   

                FZall[i*opt.Norders+j] = ck_pad_move
                del ck_pad
                
    return FZall,FZall_02


def GetWiener1(opt,FZall,FZotf,PGxy,PGz,cfai,w1,order_flag,run_parameter,notch,direction=[0,1,2]):
    "Combine the numerator, add according to direction,,mask、notch"
    nx = opt.nx*opt.kuodaxy                 
    ny = opt.ny*opt.kuodaxy                 
    nz = opt.nz_re                 
    nc = opt.nc
    Norders=opt.Norders
    
    out_K_all =  np.zeros((nz,nx,ny),dtype=np.complex64)
    
    jiesu = choose_jiesu(opt,order_flag)
    for i in direction:
        FZ_dir = np.zeros((nz,nx,ny),dtype=np.complex64)
        for j in jiesu:
            'make notch'
            if j ==0:
                Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['notch0th'],fwhm=run_parameter['attFWHM0'])) 
            elif j!=0:
                if notch == 0:
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=0,fwhm=1))  
                elif notch ==1 and j>0 and j <=2: # 2 order notch
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['attStrFZ'],fwhm=run_parameter['attFWHMFZ'])) 
                elif notch ==1 and j>2:           # 1 order notch
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['attStr'],fwhm=run_parameter['attFWHM'])) 
                else:
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=0,fwhm=1))  
                
            'combine cfai'
            amp = cfai[i,j,0]*np.exp(1j*cfai[i,j,1])
            
            'Combine into a larger spectrum'
            scale = Atteg1*(FZotf[i*Norders+j].conj())*np.conj(amp)
            temp_FZ = FZall[i*Norders+j]*scale
            FZ_dir  += temp_FZ
            
        out_K_all +=  1*FZ_dir
        
    return out_K_all


def GetWiener1_Joint(opt,FZall,FZall_02,FZotf,PGxy,PGz,cfai,cfai_02,w1,order_flag,run_parameter,notch,direction=[0,1,2]):
    "Combine the numerator, add according to direction, mask、notch"
    nx = opt.nx*opt.kuodaxy                
    ny = opt.ny*opt.kuodaxy               
    nz = opt.nz_re                 
    nc = opt.nc
    Norders=opt.Norders
    
    out_K_all =  np.zeros((nz,nx,ny),dtype=np.complex64)
    jiesu = choose_jiesu(opt,order_flag)
    for i in direction:
        FZ_dir = np.zeros((nz,nx,ny),dtype=np.complex64)
        for j in jiesu:
            'make notch'
            if j ==0:
                Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['notch0th'],fwhm=run_parameter['attFWHM0'])) 
                Atteg1 = Atteg1*1
            elif j!=0:
                if notch == 0:
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=0,fwhm=1))  
                elif notch ==1 and j>0 and j <=2: # 2-notch
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['attStrFZ'],fwhm=run_parameter['attFWHMFZ'])) 
                elif notch ==1 and j>2:           # 1-notch
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['attStr'],fwhm=run_parameter['attFWHM'])) 
                else:
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=0,fwhm=1))  
                
                
            'combine cfai'
            amp = cfai[i,j,0]*np.exp(1j*cfai[i,j,1])
            if j<3:
                amp_02 = cfai_02[i,j,0]*np.exp(1j*cfai_02[i,j,1])
            
            'Combine into a larger spectrum'
            scale = Atteg1* (FZotf[i*Norders+j].conj()) *np.conj(amp)
            temp_FZ = FZall[i*Norders+j]*scale
            FZ_dir  += temp_FZ
            
            if j<3 and j>=run_parameter['joints']:
                scale_02 = Atteg1* (FZotf[i*Norders+j].conj()) * np.conj(amp_02)
                temp_FZ_02 = FZall_02[i*3+j]*scale_02
                FZ_dir  += temp_FZ_02                                             

        out_K_all += 1*FZ_dir
        
    return out_K_all


def GetWiener2(opt,Wiener1,FZotf,PGxy,PGz,cfai,w1,order_flag,run_parameter,notch,direction=[0,1,2]):
    'combine Msum'
    Norders=opt.Norders
    'Construct Wiener Denominator'
    Msum_all = w1
    jiesu = choose_jiesu(opt,order_flag)
    for i in direction:
        for j in jiesu:
            'amp'
            amp = cfai[i,j,0]*np.exp(1j*cfai[i,j,1])
            Msum_all += FZotf[i*Norders+j]**2 * abs(amp)**2 
            
    return Msum_all


def GetWiener2_Joint(opt,Wiener1,FZotf,PGxy,PGz,cfai,cfai_02,w1,order_flag,run_parameter,notch,direction=[0,1,2]):
    'combine Msum'
    Norders=opt.Norders
    'Construct Wiener Denominator'
    Msum_all = w1
    jiesu=[0,1,2,3,4,5,6]
    for i in direction:
        for j in jiesu:
            'amp'
            amp = cfai[i,j,0]*np.exp(1j*cfai[i,j,1])
            Msum_all += (FZotf[i*Norders+j]* (FZotf[i*Norders+j].conj()) ) * abs(amp)**2 
            
            if j>=run_parameter['joints'] and j<3:
                amp_02 = cfai_02[i,j,0]*np.exp(1j*cfai_02[i,j,1])
                Msum_all += (FZotf[i*Norders+j]* (FZotf[i*Norders+j].conj()) ) * abs(amp_02)**2
    return Msum_all


def Final_ifft(opt,fimage_k,chushi,w1,notch,flagg,direction,run_parameter,filename,p1_i_):
    """
    IFFT,save
    """
    temp_r = _U.ifftn((fimage_k))
    
    #crop the padding
    if opt.Zpadding!=0:
        temp_r_cai = temp_r[2*opt.Zpadding:-2*opt.Zpadding].real
    else:
        temp_r_cai = temp_r.real

    #Truncate negative values.
    temp_r_cai_j = np.copy(temp_r_cai)
    temp_r_cai_j[temp_r_cai_j<0] = 0  
    
    
    _U.save_tif_self3((run_parameter['data_path']), temp_r_cai_j, sub_filename='Result' , filename='Wiener_results.tif' ,min=np.min(temp_r_cai_j),max=np.max(temp_r_cai_j))

    return 