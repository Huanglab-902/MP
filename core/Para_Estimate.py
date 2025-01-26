import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import cv2
import Utils as _U
import WindowF as _WF
import DataProcess as _DP
from pathlib import Path

from PyEMD import EMD

import cv2
import math
import datetime 

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


def flip180(arr): 
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr.copy()


def dft_corr2d(opt,h,g):
    """Perform 2D cross-correlation"""
    g = torch.conj(g)   
    try:
        g = (flip180(g.numpy()))
    except:
        g = (flip180(g.resolve_conj().numpy())) 
    k= np.array(h.shape)
    L = np.array(g.shape)

    g = _U.ToTensor(g)
    h = _U.ToTensor(h)
    f = torch.zeros_like(_U.ToTensor(k+L),dtype = opt.tdtype_com)
    f = tifftn( tfftn( h,s=tuple(k+L-1) )*tfftn( (g),s=tuple(k+L-1) ) )
    nx,ny = f.shape

    if nx!= (k+L)[0]: 
        f = _U.pad_or_cut(f,x=(k+L)[0],y=(k+L)[1])
    return _U.ToNumpy(f)

def dft_corr3d(opt,h,g):
    """Perform 3D cross-correlation"""
    g = torch.conj(g)
    try:
        g = (flip180(g.numpy()))
    except:
        g = (flip180(g.resolve_conj().numpy()))

    k= np.array(h.shape)
    L = np.array(g.shape)

    g = _U.ToTensor(g)
    h = _U.ToTensor(h)
    f = torch.zeros_like(_U.ToTensor(k+L),dtype = opt.tdtype_com)
    f = tifftn(tfftn(h,s=tuple(k+L-1) )*tfftn((g),s=tuple(k+L-1)))
    nz,nx,ny = f.shape
    
    if nx!= (k+L)[0]: 
        f = _U.pad_or_cut(f,x=(k+L)[1],y=(k+L)[2],z=(k+L)[0])
    return _U.ToNumpy(f)


def Cir_Erode(data,size=3):
    """
    input: np
    output: np
    """
    if size == 0:
        return data
    kernel = np.ones((size, size), np.uint8)
    kernel_re = []
    rows, cols = kernel.shape
    for ii in range(rows):
        result = [0 if math.sqrt((ii-size//2)**2+(jj-size//2)**2) > size//2 else 1 for jj in range(cols)]
        kernel_re.append(result)
    kernel_re = np.array(kernel_re, np.uint8)

    data_erode =cv2.erode(data,kernel_re)

    if np.mean(data)!=0 and np.mean(data_erode) == 0:
        raise 'The 3D erosion is too severe; the size needs to be reduced'
    return data_erode


def Cir_Erode_3D(data,size=3):
    data2=_U.image_reshape(data,dim=3)
    nz,nx,ny = data2.shape
    for i in range(nz):
        data2[i]=Cir_Erode(data[i],size=size)
    return data2


def saveparameter(path,PGxy,PGz,cfai,run_parameter):
    '''
    save parameters
    '''
    path = str(path)
    filename = run_parameter['filename']
    Dr_sub = run_parameter['Dr_sub']
    pzz = str(run_parameter['pzzz'])
    
    if PGxy.any():
        filename1 = path+'\\'+'PGxy'+'-'+Dr_sub+'-'+filename+'-'+pzz+'.npy'
        np.save(filename1,PGxy)

    if PGz.any():
        filename2 = path+'\\'+'PGz'+'-'+Dr_sub+'-'+filename+'-'+pzz+'.npy'
        np.save(filename2,PGz)

    if cfai.any():
        filename3 = path+'\\'+'cfai'+'-'+Dr_sub+'-'+filename+'-'+pzz+'.npy'
        np.save(filename3,cfai)


def saveparameter02(path,cfai,run_parameter):
    '''
    save parameters 2
    '''
    path = str(path)
    filename = run_parameter['filename']
    Dr_sub = run_parameter['Dr_sub']
    pzz = str(run_parameter['pzzz'])
    
    if cfai.any():
        filename3 = path+'\\'+'cfa02'+'-'+Dr_sub+'-'+filename+'-'+pzz+'.npy'
        np.save(filename3,cfai)


def loadparameter(opt,path,run_parameter):
    '''
    load parameter 
    '''
    PGzF=0
    PGxyF=0
    cfaiF=0
    Dr_sub = run_parameter['Dr_sub']
    loop_file = Path(str(path)).iterdir()
    for ldr_i in loop_file:
        if ldr_i.is_file() == 1 :  
            if   ('PGxy' in ldr_i.name  ) :
                name = ldr_i.name
                print(f'load PGxy: {name}')
                PGxy = np.load(ldr_i)
                PGxyF=1
            elif ('cfai' in ldr_i.name ) :
                name = ldr_i.name
                print(f'load cfai: {name}')
                cfai = np.load(ldr_i)
                cfaiF=1
            elif ('PGz' in ldr_i.name ):
                name = ldr_i.name
                print(f'load PGz: {name}')
                PGz = np.load(ldr_i)
                PGzF=1


    if PGxyF == 1:
        plong=0
        for i in range(opt.Nangles):
            for j in range(1,3):   #±2
                plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)
            for j in range(3,opt.Norders):   #±1
                plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2
        plong = plong/(opt.nc-3)


    if PGzF==0:
        PGz = np.zeros((3,7))
    return cfai,PGxy,PGz,plong



def loadparameter2(opt,path,run_parameter):
    loop_file = Path(str(path)).iterdir()
    PGzF=0
    PGxyF=0
    Dr_sub = run_parameter['Dr_sub']
    for ldr_i in loop_file:
        if ldr_i.is_file() == 1 :  
            if   ('PGxy' in ldr_i.name):
                name = ldr_i.name
                print(f'load PGxy: {name}')
                PGxy = np.load(ldr_i)
                PGxyF=1
            elif ('PGz' in ldr_i.name) :
                PGz = np.load(ldr_i)
                PGzF=1

    plong=0
    for i in range(opt.Nangles):
        for j in range(1,3):   #±2
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)
        for j in range(3,opt.Norders):   #±1
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2
    plong = plong/(opt.nc-3)
    
    if PGzF==0:
        PGz = np.zeros((3,7))
        print('Fail to load parameter')
    return PGxy,PGz,plong

def loadparameter3(opt,path):
    '''
    only load cfai
    '''
    try:
        filename3 = path+'cfai.npy'
        cfai = np.load(filename3)
    except:
        print('Fail to load parameter ')
        pass

    return cfai

def loadparameter4(opt,path,run_parameter):
    '''
    load cfai02
    '''
    from pathlib import Path
    loop_file = Path(str(path)).iterdir()
    Dr_sub = run_parameter['Dr_sub']
    for ldr_i in loop_file:
        if ldr_i.is_file() == 1 :  
            if ('cfa02' in ldr_i.name ) :
                name = ldr_i.name
                print(f'load cfai02: {name}')
                print(f'load cfai02: {ldr_i}')
                cfai02 = np.load(ldr_i)
                return cfai02
    
    raise('No find cfai02')


#############the parameter estimation section##########################
def get_intpxy(opt,ck0,ck1,i,j,methods='FAN'):
    '''
    '''
    assert len(ck0.shape) == 3
    assert len(ck1.shape) == 3
    nz,nx,ny = ck0.shape
    print(nz,nx,ny)
    if methods=='FAN':
        Hzuan = _U.ToTensor(_U.pad_or_cut(opt.OTFmask2D,x=nx,y=ny))
        Hzuan = _U.image_reshape(Hzuan,dim=2)
        assert len(Hzuan.shape)==2
        
        cishu = abs(dft_corr2d(opt,Hzuan,Hzuan))     
        _U.save_tif('cishu.tif',cishu.real)

        
        imagf0AC = ck0[nz//2]                         # projetction
        imagf0_ = imagf0AC/(np.abs(imagf0AC)+opt.eps) #
        imagf1AC = ck1[nz//2]                         
        imagf1_ = imagf1AC/(np.abs(imagf1AC)+opt.eps) 
        
        
        imagf0_ = _U.ToTensor(imagf0_)
        imagf1_ = _U.ToTensor(imagf1_)

        jieguo = dft_corr2d(opt,(imagf0_),(imagf1_)); 
        lihe = abs(jieguo/(cishu+opt.eps));           
        
    elif methods=='SHAO':
        imagf0AC = ck0
        imagf1AC = ck1
        
        imagf0AC_pad = _U.pad_or_cut(imagf0AC,x=2*nx,y=2*ny,z=1*nz)
        imagf1AC_pad = _U.pad_or_cut(imagf1AC,x=2*nx,y=2*ny,z=1*nz)
        imagf0AC_kong = _U.ifftn(imagf0AC_pad)      
        imagf1AC_kong = _U.ifftn(imagf1AC_pad)      

        lihe = (np.max( abs(_U.fftn(imagf0AC_kong*imagf1AC_kong.conj() ))/1000,axis=0))   

    XV,YV=_U.meshgridxy(opt,kuodaxy=2)
    kr = np.sqrt(XV**2+YV**2)
    lihe[kr>(opt.pg+opt.fanwei)]=0
    lihe[kr<(opt.pg-opt.fanwei)]=0

    lihe = abs(lihe)
    _U.save_tif(f"4 lihe-pxy-{i}-{j}.tif",lihe)
    opt.lihe2d.append(lihe)
    maxx = np.where(lihe==np.max(lihe))[0][0] # 168。8
    maxy = np.where(lihe==np.max(lihe))[1][0] # 127.5
    return maxx-opt.nx,maxy-opt.ny



def get_intpxy_2D(opt,ck0,ck1,i,j,methods='FAN'):
    '''
    input 2D
    '''
    assert len(ck0.shape) == 2
    assert len(ck1.shape) == 2
    nz,nx,ny = ck0.shape
    print(nz,nx,ny)
    if methods=='FAN':
        Hzuan = _U.ToTensor(_U.pad_or_cut(opt.OTFmask2D,x=nx,y=ny))
        Hzuan = _U.image_reshape(Hzuan,dim=2)
        assert len(Hzuan.shape)==2
        
        cishu = abs(dft_corr2d(opt,Hzuan,Hzuan))     
        _U.save_tif('cishu.tif',cishu.real)

        
        imagf0AC = ck0                        
        imagf0_ = imagf0AC/(np.abs(imagf0AC)+opt.eps) 
        imagf1AC = ck1                  
        imagf1_ = imagf1AC/(np.abs(imagf1AC)+opt.eps) 
        _U.save_tif('imagf0AC.tif',imagf0AC.real)
        imagf0_ = _U.ToTensor(imagf0_)
        imagf1_ = _U.ToTensor(imagf1_)

        jieguo = dft_corr2d(opt,(imagf0_),(imagf1_));
        _U.save_tif('jieguo.tif',jieguo.real)
        lihe = abs(jieguo/(cishu+opt.eps));

    elif methods=='SHAO':
        imagf0AC = ck0
        imagf1AC = ck1
        
        imagf0AC_pad = _U.pad_or_cut(imagf0AC,x=2*nx,y=2*ny,z=1*nz)
        imagf1AC_pad = _U.pad_or_cut(imagf1AC,x=2*nx,y=2*ny,z=1*nz)
        imagf0AC_kong = _U.ifftn(imagf0AC_pad)      
        imagf1AC_kong = _U.ifftn(imagf1AC_pad)      

        lihe = (np.max( abs(_U.fftn(imagf0AC_kong*imagf1AC_kong.conj() ))/1000,axis=0))  

    XV,YV=_U.meshgridxy(opt,kuodaxy=2)
    kr = np.sqrt(XV**2+YV**2)
    lihe[kr>(opt.pg+opt.fanwei)]=0
    lihe[kr<(opt.pg-opt.fanwei)]=0

    lihe = abs(lihe)
    # _U.save_tif(f"4 lihe-pxy-{i}-{j}.tif",lihe)
    opt.lihe2d.append(lihe)
    maxx = np.where(lihe==np.max(lihe))[0][0] # 168。8
    maxy = np.where(lihe==np.max(lihe))[1][0] # 127.5

    return maxx-opt.nx,maxy-opt.ny


def get_intpz(opt,ck0,ck1,i,j,methods='FAN'):
    assert len(ck0.shape) == 3
    nz,nx,ny = ck0.shape
    
    #
    ck0 = ck0/np.max(abs(ck0))
    ck1 = ck1/np.max(abs(ck1))
    notch_filter1 = _WF.getotfAtt(opt,fwhm = 0.2*1000/opt.fc);

    fushi = 0
    if methods=='FAN':
        Hzuan = (_U.pad_or_cut(opt.OTFmask3D,x=nx,y=ny,z=nz)) 
        Hzuan = _U.ToTensor(Cir_Erode(Hzuan,size = fushi))

        cishu = abs(dft_corr3d(opt,Hzuan,Hzuan))     
        imagf0AC = ck0                     
        imagf0_ = imagf0AC/(np.abs(imagf0AC)+opt.eps) 
        imagf1AC = ck1                         
        imagf1_ = imagf1AC/(np.abs(imagf1AC)+opt.eps) 

        imagf0_ = _U.ToTensor(imagf0_)*Hzuan
        imagf1_ = _U.ToTensor(imagf1_)*Hzuan

        jieguo = dft_corr3d(opt,(imagf0_),(imagf1_));     
        lihe = abs(jieguo/(cishu+opt.eps));           
        lihe = _U.pad_or_cut(lihe,x=nx,y=ny,z=2*nz)

    elif methods=='SHAO':

        ck0_pad = _U.pad_or_cut(ck0,x=1*nx,y=1*ny,z=2*nz)
        ck1_pad = _U.pad_or_cut(ck1,x=1*nx,y=1*ny,z=2*nz)


        Hzuan = _U.pad_or_cut(opt.OTFmask3D,x=nx,y=ny,z=2*nz)
        Hzuan = Cir_Erode(Hzuan,size = fushi)
        imagf0AC_pad = ck0_pad * Hzuan * notch_filter1                         
        imagf1AC_pad = ck1_pad * Hzuan * notch_filter1
        
        imagf0AC_kong = _U.ifftn(imagf0AC_pad)      #
        imagf1AC_kong = _U.ifftn(imagf1AC_pad)      #
        lihe = abs(_U.fftn(imagf0AC_kong*imagf1AC_kong.conj() ))/1000
        

    XV,YV=_U.meshgridxy(opt,kuodaxy=1) 
    kr = np.sqrt(XV**2+YV**2)
    lihe = abs(lihe/np.max(lihe));

    lihe[:,kr>(opt.pg+opt.fanwei)]=0
    lihe[:,kr<(opt.pg-opt.fanwei)]=0
    lihe[0:nz-opt.pgz] = 0    
    lihe[nz+opt.pgz:] = 0     

    lihe[nz] = 0          

    lihe2 = (np.max( abs(lihe)/10,axis=2))  
    lihe2[nz,:] = 0

    opt.lihe3d.append(lihe)
    opt.lihe3dxy.append(lihe2)

    # _U.save_tif(f"lihez-{i}-{j}-{methods}.tif",lihe)
    
    maxz = np.where(lihe==np.max(lihe))[0][0]
    maxx = np.where(lihe==np.max(lihe))[1][0] # 168。8
    maxy = np.where(lihe==np.max(lihe))[2][0] # 127.5

    return int(maxz-(opt.nz_re/2)),int(maxx-opt.nx/2),int(maxy-opt.ny/2)  


#---------------------------------------------------------
def get_subpxy(opt,i,j,ck_0,ck_p,pgx,pgy,func):
    #The ‘Lihe’ function, used as the basis for updates.
    xstep = 1
    ystep = 1
    buchangy = ystep
    buchangx = xstep
    jindu = 0.05
    PXY =np.array([pgx,pgy])/(2*opt.nx)
    he_0 = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1])
    while( (buchangx>jindu) or (buchangy>jindu)):

        test = np.zeros(4)

        """max x"""
        maxx_tmp1 = pgx-1e-5; #UP
        maxx_tmp2 = pgx+1e-5; #DOWN
        
        for i in range(2):
            if i==0:
                kxtest=maxx_tmp1;
            else:
                kxtest=maxx_tmp2;
            PXY =np.array([kxtest,pgy])/(2*opt.nx) 
        
            he1 = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1])
            test[i] = he1
        
        if((test[0]>test[1])):  
            flag_maxx=-1;       
        elif(test[0]<test[1]):
            flag_maxx=1;       
        else:
            try:
                flag_maxx=-flag_maxx
            except:
                flag_maxx=1

        while(buchangx>(jindu)):   
            kxtest = pgx+flag_maxx*buchangx;     

            PXY = np.array([kxtest,pgy])/(2*opt.nx)   
            he_tmp = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1]) 
            if(he_tmp<=he_0):
                buchangx=buchangx*0.5
            else:
                print("updata pgx =",kxtest,'hetemp = ',he_tmp)
                pgx=kxtest          
                he_0=he_tmp
                #reset
                buchangy = ystep
                break            
            

        """max y"""
        maxy_tmp1 = pgy-1e-5; #left
        maxy_tmp2 = pgy+1e-5; #right
        for i in range(2,4): 
            if i==2:
                kytest=maxy_tmp1;
            else:
                kytest=maxy_tmp2;
            PXY = np.array([pgx,kytest])/(2*opt.ny)   
        
            he1 = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1])
            test[i] = he1
        
        if((test[2]>test[3])): 
            flag_maxy=-1;      
        elif(test[2]<test[3]):
            flag_maxy=1;       
        else:
            try:
                flag_maxy=-flag_maxy
            except:
                flag_maxy=1
            
        
        while(buchangy>(jindu)):
            kytest = pgy+flag_maxy*buchangy;      
            PXY = np.array([pgx,kytest])/(2*opt.ny) 
            he_tmp = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1])
            if(he_tmp<=he_0):
                buchangy=buchangy*0.5
            else:
                print("updata pgy =",kytest,'hetemp = ',he_tmp)
                pgy=kytest
                he_0=he_tmp
                #reset
                buchangx = xstep
                break
    return pgx,pgy

def get_subpz(opt,i,j,ck_0,ck_p,pgx,pgy,pgz,m,func,jindu=0.1,fanwei=6):
    #Equal Division Method
    nz = (opt.nz+2*opt.Zpadding) #
    nx = opt.nx  #

    PXY = np.array([pgx,pgy])/(2*nx)
    print('ck_p.shape',ck_p.shape)
    print('nz=',nz)
    PGZ =(pgz)/(nz*2)

    he_0 = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PGZ,m)
    fanwei=fanwei 
    he_array = np.zeros(2*fanwei-1)
    he_array[fanwei-1] = he_0
    print('stand：',he_0.round(6))
    for ii in range(1,fanwei):
        #+i right
        print('+',ii,' ',pgz+ii*jindu)
        PGZ=(pgz+ii*jindu)/(nz*2)
        temp = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PGZ,m)
        print(temp.round(6))
        he_array[(fanwei-1)+ii] = temp
        
    for ii in range(1,fanwei):
        #-i left
        print('-',ii,' ',pgz-ii*jindu)
        PGZ=(pgz-ii*jindu)/(nz*2)
        temp = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PGZ,m)
        he_array[(fanwei-1)-ii] = temp
        print(temp.round(6))
    print(he_array)

    X = np.linspace(-(fanwei-1)*jindu+pgz,(fanwei-1)*jindu+pgz,fanwei*2-1)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(X , he_array, color = 'b', width = jindu/2)
    plt.xlim(-(fanwei-1)*jindu+pgz,(fanwei-1)*jindu+pgz)
    # plt.show()

    j = np.where(he_array==np.max(he_array))[0][0]
    
    if j>(fanwei-1): # 6,7,8,9,10 
        print('j +',( j-(fanwei-1) ))
        print(pgz+( j-(fanwei-1) )*jindu)
        return pgz+( j-(fanwei-1) )*jindu
    else:
        print('j -',((fanwei-1)-j ))
        print(pgz- ((fanwei-1)-j )*jindu)
        return pgz-((fanwei-1)-j )*jindu


def LiheXYZ(opt,i,j,ck_0,ck_p,kx,ky,kz=0,m=2,cfai=0,mask_ger=0,run_parameter=''):
    "Both 2D plane calculations and 3D calculations are supported"
    if m>2 and opt.flag_N=='MP' and opt.Norders==7 : 
        flag=1 

    if j <3 and cfai!=4: #
        piancha = 0
        flag= 0  
        filename=''

    elif j>=3 and cfai!=4: 
        piancha = 0 
        flag= 1 
        filename=''

    else:
        piancha = 0
        flag= 1
        filename = ''


    assert len(ck_0.shape)==3

    nz,nx,ny = ck_0.shape

    #move matrix.
    ZV,XV,YV = _U.meshgrid(opt,kuodaxy=2,kuodaz=opt.kuodapz)
    ysh  = (np.exp(2j*np.pi*(kx*XV+ky*YV+kz*ZV)))  

    if flag==0: #±2  2D
        
        ysh = ysh[0]        
        #[1]0-order 【image】
        imgf0AC = ck_0[nz//2+piancha,:,:]  #0
        imgf0 = imgf0AC/(abs(imgf0AC)+opt.eps)   #
        imgf0 = _U.pad_or_cut(imgf0,x=nx*2 ,y=ny*2,z=1)    
        
        #[2]0-order 【otf、otfmask】2d
        otf = opt.OTF2D
        otfmask = opt.OTFmask2D

        #[3]high-order image
        imgfmAC = ck_p[nz//2+piancha,:,:]
        imgfm = imgfmAC/(abs(imgfmAC)+opt.eps)   #
        imgfm = _U.pad_or_cut(imgfm,x=nx*2 ,y=ny*2,z=1)    # Dr 2x


    else: #±1 3D
        #[1] 0-order 【image】
        imgf0AC = ck_0  
        imgf0 = imgf0AC/(abs(imgf0AC)+opt.eps)   
        imgf0 = _U.pad_or_cut(imgf0,x=nx*2 ,y=ny*2,z=opt.nz_re)    
        
        #[2] 0-order 【otf】3d
        otf = _U.image_reshape(opt.OTF,dim=3)
        otfmask = opt.OTFmask3D

        #[3] high-order 【image】
        imgfmAC = ck_p
        imgfm = imgfmAC/(abs(imgfmAC)+opt.eps)  
        imgfm = _U.pad_or_cut(imgfm,x=nx*2 ,y=ny*2,z=opt.nz_re)    # Dr 2x
        
    #################move ##########################################
    ysh   = (torch.from_numpy(ysh)).to(device=opt.device)
    otf = (torch.from_numpy(otf)).to(device=opt.device)
    otfmask = (torch.from_numpy(otfmask)).to(device=opt.device)
    imgfm = (torch.from_numpy(imgfm)).to(device=opt.device)

    #[3.]imgf high-order move
    imgfm_move = _U.tfftn(_U.tifftn(imgfm)*(ysh))

    #[4.]otf_move generate
    psf_ = tfftshift(_U.tifftn(otf)).real    
    otf_move = abs(_U.tfftn( tifftshift(psf_*ysh)) )
    
    #[]otfmask_move 
    psf_mask_ = tfftshift(_U.tifftn(otfmask)).real    
    otfmask_move = abs(_U.tfftn( tifftshift(psf_mask_*ysh)) )
    otfmask_move[otfmask_move>0.9]=1
    otfmask_move[otfmask_move!=1]=0
     
    #CPU
    ysh = ysh.cpu().numpy()
    otf = otf.cpu().numpy()
    otfmask = otfmask.cpu().numpy()
    imgfm = imgfm.cpu().numpy()

    imgfm_move = imgfm_move.cpu().numpy()
    otf_move = otf_move.cpu().numpy()
    otfmask_move = otfmask_move.cpu().numpy()
    del psf_,psf_mask_

    ##############################################################
    #overlap area
    w0 = (imgf0 * otfmask) * (otf_move   * otfmask_move)
    w1 = (otf   * otfmask) * (imgfm_move * otfmask_move) 
    w0 = w0.astype(np.complex64)
    w1 = w1.astype(np.complex64)
    overlap_mask = (otfmask_move * otfmask)

    #w0 w1
    overlap_sum = abs(np.sum((w1*w0.conj()))); 
    overlap_mask_sum = abs(np.sum((overlap_mask)))

    a = overlap_sum /overlap_mask_sum
    return abs(a)        

def cal_c_fai(opt,i,j,ck_0,ck_p,kx,ky,kz=0,filename='',run_parameter=""):
    assert len(ck_0.shape)==3
    nz,nx,ny = ck_0.shape
    # move matrix
    ZV,XV,YV = _U.meshgrid(opt,kuodaxy=2,kuodaz=opt.kuodapz)
    ysh  = (np.exp(2j*np.pi*(kx*XV+ky*YV+kz*ZV)))   
 
    #[1] 0-order ck
    imgf0 = np.copy(ck_0)
    imgf0 = _U.pad_or_cut(imgf0,x=nx*2 ,y=ny*2,z=opt.nz_re)

    #[2] 0-order OTFmask
    otf = abs(_U.image_reshape(opt.OTF,dim=3))
    otfmask = opt.OTFmask3D

    #[3] high-order ck
    imgfm = np.copy(ck_p)
    imgfm = _U.pad_or_cut(imgfm,x=nx*2 ,y=ny*2,z=opt.nz_re)    # Dr 2x
    
    #[4] high-order【OTFmask_move】
    otfmask_move , _,_ = _DP.otf_mask_3d_2d(opt,PGxy =[kx*(2*opt.nx),ky*(2*opt.ny)],PGz = [kz*opt.nz_re], Debug=0) #
    otfmask_move = Cir_Erode_3D(otfmask_move,size=3)   

    
    ################move##########################################
    ysh   = (torch.from_numpy(ysh)).to(device=opt.device)
    otf = (torch.from_numpy(otf)).to(device=opt.device)
    otfmask = (torch.from_numpy(otfmask)).to(device=opt.device)
    imgfm = (torch.from_numpy(imgfm)).to(device=opt.device)

    #[3.]imgf high-order 
    imgfm_move = _U.tfftn(_U.tifftn(imgfm)*(ysh))

    #[4.]otf_move 
    psf_ = tfftshift(_U.tifftn(otf)).real    
    otf_move = abs(_U.tfftn( tifftshift(psf_*ysh)) ) 
    
    #[]otfmask_move 
    psf_mask_ = tfftshift(_U.tifftn(otfmask)).real    
    otfmask_move = abs(_U.tfftn( tifftshift(psf_mask_*ysh)) )
    otfmask_move[otfmask_move>0.9]=1
    otfmask_move[otfmask_move!=1]=0
    
    #CPU
    ysh = ysh.cpu().numpy()
    otf = otf.cpu().numpy()
    otfmask = otfmask.cpu().numpy()
    imgfm = imgfm.cpu().numpy()
    

    imgfm_move = imgfm_move.cpu().numpy()
    otf_move = otf_move.cpu().numpy()
    otfmask_move = otfmask_move.cpu().numpy()
    del psf_,psf_mask_
    

    #overlap area
    w0 = (imgf0 * otfmask) * (otf_move   * otfmask_move)
    w1 = (otf   * otfmask) * (imgfm_move * otfmask_move)
    overlap_mask = (otfmask_move * otfmask)
    
    #mask smaller
    overlap_mask = _U.Cir_Erode_3D(overlap_mask,size=int(opt.fc//5))
    
    #rate
    overlap_cfai_3D = (w1/(w0+opt.eps))   * overlap_mask.astype(np.bool_)
    overlap_cfai = np.copy(overlap_cfai_3D) #

    # get Modulation depth， Initial phase
    overlap_c    = np.abs(overlap_cfai)
    overlap_fai  = np.angle(overlap_cfai/(abs(overlap_cfai)+opt.eps)); 

    #recoder overlap 
    opt.abscm3d.append(overlap_c)
    opt.angcm3d.append(overlap_fai)
    opt.overlap_mask3d.append(overlap_mask)

    
    'EMD' 
    C,fai,C2 = Cal_parameter(overlap_c,overlap_fai,1,i,j,filename,run_parameter)
    return C,fai,C2



def Cal_parameter(overlap_c,overlap_fai,flag3D,i,j,filename,run_parameter,updown='whole'):
    overlap_c_1d = overlap_c.flatten()[np.argwhere(overlap_c.flatten())].squeeze()
    overlap_fai_1d = overlap_fai.flatten()[np.argwhere(overlap_fai.flatten())].squeeze()
    overlap_fai_1d_T = overlap_fai_1d.T
    overlap_c_1d_T =  overlap_c_1d.T
    
    'get c by mean'
    if flag3D==0:  
        overlap_c_1d_T2 = overlap_c_1d_T*overlap_c_1d_T[::-1]
    else:                   
        overlap_c_1d_T2 = overlap_c_1d_T**2
    overlap_c_ = overlap_c_1d_T2**0.5
    c_mean = np.mean(overlap_c_[(overlap_c_<1) & (overlap_c_>0.03)])
    
    'get initial phase by EMD'
    FAI_interval = 0.02  
    FAI_range = np.arange(-np.pi,np.pi,FAI_interval) # fai
    
    FAI_histogram,_ = np.histogram(overlap_fai_1d_T,FAI_range)  
    FAI_histogram[int(np.pi/FAI_interval)]=(FAI_histogram[int(np.pi/FAI_interval)+1]+FAI_histogram[int(np.pi/FAI_interval)-1])/2
    
    fai,FAI_histogram_4pi,FAI_EMD,rate,FAI_idx_max = extend(FAI_histogram,j,FAI_interval)
    'get C by EMD'
    C_interval = 0.01    
    C_range = np.arange(C_interval+0,1+C_interval,C_interval) #c
    C_histogram,_ = np.histogram(overlap_c_,C_range)
    
    emd = EMD()
    C_imf = emd(C_histogram)  #c
    C_EMD = np.sum(C_imf[0:,:],0)
    C_EMD[0:10]=0   
    C_idx_max = np.where(C_EMD == np.max(C_EMD)) 
    cvalue = C_interval*np.mean(C_idx_max)+C_interval+0

    # print('c_mean:',c_mean) 
    # print("fai=",fai)
    # print("c=",cvalue)
    
    draw_c_fai(C_histogram,C_EMD,FAI_histogram_4pi,FAI_EMD,i,j,np.mean(FAI_idx_max),c_mean,cvalue,fai,filename,run_parameter,np.mean(C_idx_max),C_interval,FAI_interval)

    if rate>0.2: 
        return cvalue, fai,c_mean
    else:       
        return c_mean,fai,cvalue
    
def draw_c_fai(c,c2,fa,fa2,i,j,h,c_mean,c_EMD,fai,filename,run_parameter,hh,jdc,jdjd):
    import matplotlib.pyplot as plt
    'fig'
    fig = plt.figure(figsize=(15,3))
    sub1 = plt.subplot(1, 2, 1)  
    
    c_max = np.max(c2)
    # plt.plot()
    start = 10
    plt.plot(c2)
    plt.title(f'{i+1,j}-C-Modulation depth',fontname='Times New Roman',weight ='bold')
    # sub1.text(start+1,  0,"MEAN:",ha='center', va='bottom')
    sub1.text(0.98, 0.98,'MEAN:'+str(c_mean.round(2)),
             horizontalalignment='right',
             verticalalignment='top',
             transform = sub1.transAxes)
    # sub1.text(start+1,  0,str(c_mean.round(2)),ha='center', va='bottom')    
    sub1.text(hh, 0,str((c_EMD).round(2)),ha='center', va='bottom')
    
    keduc = [20,40,60,80,100]
    # keduc = np.array(keduc)-20
    keduc2 = []
    for ii in range(len(keduc)):
            keduc2.append( np.round((keduc[ii])*jdc,1) )
    sub1.set_xlim((start, 100))
    sub1.set_xticks(keduc)
    sub1.set_xticklabels(keduc2)
    sub1.set_ylim(bottom=0.)  #Y

    #####################fai############
    sub2 = plt.subplot(1, 2, 2)  
    plt.plot(fa)
    plt.plot(fa2)
    plt.title(f'{i+1,j}-φ-phase',fontname='Times New Roman',weight ='bold')
    sub2.text(h, 0,fai.round(2),ha='center', va='bottom')
    kedu = [0,157,314,471,628]
    kedu2 = []
    for ii in range(len(kedu)):
        if kedu[ii]<=314:
            pianyi = np.pi
        else:
            pianyi = 3*np.pi

        if kedu[ii]==314:
            kedu2.append( '±'+str(abs(np.round((kedu[ii]*jdjd-pianyi),2))) )
        else:
            kedu2.append( np.round((kedu[ii]*jdjd-pianyi),2) )
    sub2.set_xticks(kedu)
    sub2.set_xticklabels(kedu2)
    sub2.set_ylim(bottom=0.)  

    cfai_png = plt2np(plt)

    # name = _U.save_path(f'{i+1,j}-cfai-{filename}.png',run_parameter,sub_filename='cfai'+'-'+str(run_parameter['pzzz'])+'-'+str(run_parameter['Dr_sub']))
    # plt.savefig(name) # save fig

    return cfai_png

def plt2np(plt):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import PIL.Image as Image
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    rgb_image = image[:, :, :3]
    
    return rgb_image


def extend(Initial_phase_curve,j,jdjd):
    'The initial phase is extended to two cycles for better fitting with EMD.'
    IPC = Initial_phase_curve
    IPC_copy = np.copy(IPC)
    IPC_4pi = np.concatenate((IPC, IPC_copy), axis=0)

    emd = EMD()
    imf = emd(IPC_4pi)

    'EMD'
    rate = np.max(IPC_4pi) / np.sum(IPC_copy)
    # print( 'rate：',rate )
    if rate > 0.02 :              # Signal-to-noise very high
        EMD_curve = np.sum(imf[:,:],0)

    elif rate > 0.013:            # Signal-to-noise high
        EMD_curve = np.sum(imf[1:,:],0)

    elif rate > 0.008:            # Signal-to-noise not good
        EMD_curve = np.sum(imf[2:,:],0)

    else:# rate > 0.4:            # Signal-to-noise is bad
        EMD_curve = np.sum(imf[3:,:],0)

    nx = IPC_4pi.reshape(-1).shape

    'IPC_4pi'
    EMD_curve_pre = np.copy(EMD_curve)   #1st cycles
    EMD_curve_pre[int(nx[0]/2):]=0  
    idx_max = np.where(EMD_curve_pre==np.max(EMD_curve_pre)) # 
    fai = np.mean(idx_max)*jdjd - np.pi
    if fai < -2:  #2nd cycles
        EMD_curve_later = np.copy(EMD_curve) #2nd cycles
        EMD_curve_later[:int(nx[0]/2)]=0  
        idx_max = np.where(EMD_curve_later==np.max(EMD_curve_later)) 
        fai = np.mean(idx_max)*jdjd - 3*np.pi

        return fai,IPC_4pi,EMD_curve_later,rate,idx_max
    return fai,IPC_4pi,EMD_curve_pre,rate,idx_max

###################################################



def parameter_estimation3(opt,ck,PGxy,filename='',RL=False,Debug=False,run_parameter='',jieshu = [1,2,3,4,5,6]):
    #cal 1-order cfai 
    assert len(ck.shape)==4
    if run_parameter['pg']!=0:
        pg = run_parameter['pg']
        opt.pg = run_parameter['pg']
        opt.fanwei = run_parameter['fanwei']    
    else:
        pg = int(opt.pxy)
        opt.pg = int(opt.pxy)
        opt.fanwei = run_parameter['fanwei']         

    opt.cutoff = 1e-4
    opt.pgz = round(opt.pz+opt.Zpadding+2) 
    opt.fanweiz = 0 

    if opt.flag_N =='MP' and opt.Norders==7 and opt.nz!=1 and (opt.fanweiz <0 or opt.pgz>=opt.nz_re):
        print('pgz=',opt.pgz)
        print("fanweiz=",opt.fanweiz)
        raise('fanweiz error')
    elif opt.flag_N =='MP' and opt.Norders==7 and opt.nz==1: 
        opt.pgz=0
        opt.fanweiz = 0

    PGz = np.zeros((opt.Nangles,opt.Norders))               # 0，-2，2 ；-1-1 ；-1 1 ；1 -1 ；1 1 ；
                                                            # 0， 1, 2    3      4     5      6
    cfai = np.zeros((opt.Nangles,opt.Norders,3)) 
    pgz = 0
    if opt.flag_N =='MP' and opt.Norders==7 :
        kuodapz = 2            
    else:
        kuodapz = 1           

    nz = opt.nz_re 

    try:
        ck=ck.numpy()
    except:
        pass
    
    opt.lihe2d = []
    opt.overlap_mask2d = [] 
    opt.abscm2d = []
    opt.angcm2d = []

    opt.lihe3d = []
    opt.lihe3dxy = []
    opt.overlap_mask3d = [] 
    opt.abscm3d = []
    opt.angcm3d = []
    ''' <<<<<<<<<<<<<<<parameter estimate >>>>>>>>>>>>>>>>>>'''
    starttime = datetime.datetime.now()
    for i in range(3):         
        
        if RL: #never use
            RL_output=[]
            for j in range(opt.Norders):
                RL_output.append(_U.RL_deconvolve(ck[i*opt.Norders+j], opt.OTF,fre = True) )
        else: 
            RL_output=[]
            for j in range(opt.Norders):
                RL_output.append(ck[i*opt.Norders+j])

            ########################################
        RL_output = np.array(RL_output)
        assert len(RL_output.shape)==4
        cfai[i,0,0] = 1 
        

        Ns = opt.Norders
        
        for j in jieshu:
            pgx = PGxy[i,j,0]
            pgy = PGxy[i,j,1]          
            if j <=1 :  #±2-order
                c,fai,c_EMD= cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),run_parameter=run_parameter) #
                cfai[i,j,:] = np.array([c,fai,c_EMD])
            elif j == 2: ####
                cfai[i,j,0] = cfai[i,1,0]
                cfai[i,j,2] = cfai[i,1,2]
                cfai[i,j,1] = -cfai[i,1,1]

            elif j>=3 and j<=4 and opt.flag_N =='MP' and opt.Norders==7:     
                'cal pz '
                if run_parameter['pzzz']==0:
                    opt.pg = pg//2 
                    pzint,pxint,pyint = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='FAN')
                    
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pzint,m=j,fanwei=7,func=LiheXYZ) 
                    #again
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pgz,m=j,jindu=0.01,fanwei=6,func=LiheXYZ) 
                    
                
                elif run_parameter['pzzz']!=0 and run_parameter['pzzz_range'] !=0 : #'give pz and adjust'
                    opt.pg = pg//2 
                    pzint,pxint,pyint = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='FAN')
                    
                    if j == 3 or j == 5:
                        pgz = -run_parameter['pzzz'] 
                    elif j == 4 or j == 6:
                        pgz =  run_parameter['pzzz'] 
                    
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pzint,m=j,fanwei=7,func=LiheXYZ) 
                    # again
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pgz,m=j,jindu=0.01,fanwei=6,func=LiheXYZ) 
                    
                
                elif run_parameter['pzzz']!=0 and run_parameter['pzzz_range'] ==0 :  #'give PZ'
                    if j == 3 or j == 5:
                        pgz = -run_parameter['pzzz'] 
                    elif j == 4 or j == 6:
                        pgz =  run_parameter['pzzz'] 
                else:
                    raise('parameter error')

                c,fai,c_EMD = cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),run_parameter=run_parameter)
                cfai[i,j,:] = np.array([c,fai,c_EMD])
                PGz[i,j] = pgz        
            
            elif j >= 5 and opt.Norders == 7:
                if j == 5:
                    cfai[i,j,0] = cfai[i,4,0]
                    cfai[i,j,2] = cfai[i,4,2]
                    cfai[i,j,1] = -cfai[i,4,1]
                    PGz[i,j] =  PGz[i,3]      #warn  pz = 5 match 3
                elif j==6:
                    cfai[i,j,0] = cfai[i,3,0]
                    cfai[i,j,2] = cfai[i,3,2]
                    cfai[i,j,1] = -cfai[i,3,1]
                    PGz[i,j] =  PGz[i,4]      #warn pz =6 match 4
                else:
                    raise('j error')
            
            elif j==3 and opt.flag_N =='MP' and opt.Norders==5:            
                c,fai,c_EMD = cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),run_parameter=run_parameter)
                cfai[i,j,:] = np.array([c,fai,c_EMD])
                PGz[i,j] = pgz      
            elif j==4 and opt.flag_N =='MP' and opt.Norders==5:
                cfai[i,j,0] = cfai[i,3,0]
                cfai[i,j,2] = cfai[i,3,2]
                cfai[i,j,1] = -cfai[i,3,1]

    endtime = datetime.datetime.now()
    # print ('Calculate the time required for parameter estimation.',endtime - starttime )
    plong=0
    for i in range(opt.Nangles):
        for j in range(1,3):   #±2
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)
        for j in range(3,opt.Norders):   #±1
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2
    plong = plong/(opt.nc-3)

    del opt.overlap_mask3d,opt.abscm3d,opt.angcm3d


    return cfai,PGxy,PGz,plong,pgz


def parameter_estimationP2(opt,ck,filename='',RL=False,Debug=False,run_parameter=''):
    # only cal 2-order P,ck(3,5)
    assert len(ck.shape)==4
    Norders = 5
    
    PGxy = np.zeros((opt.Nangles,Norders,2))
    if run_parameter['pg']!=0:
        pg = run_parameter['pg']
        opt.pg = run_parameter['pg']
        opt.fanwei = run_parameter['fanwei']    
    else:
        pg = int(opt.pxy)
        opt.pg = int(opt.pxy)
        opt.fanwei = run_parameter['fanwei']       
    
    try:
        ck=ck.numpy()
    except:
        pass
    
    opt.lihe2d = []
    opt.overlap_mask2d = [] 
    opt.abscm2d = []
    opt.angcm2d = []

    opt.lihe3d = []
    opt.lihe3dxy = []
    opt.overlap_mask3d = [] 
    opt.abscm3d = []
    opt.angcm3d = []
    ''' <<<<<<<<<<<<<<<parameter estimate >>>>>>>>>>>>>>>>>>'''
    starttime = datetime.datetime.now()
    for i in range(3):        
        
        if RL:  #never use
            RL_output=[]   
            for j in range(Norders):
                RL_output.append(_U.RL_deconvolve(ck[i*Norders+j], opt.OTF,fre = True) )
        else:    # no RL
            RL_output=[]
            for j in range(Norders): 
                RL_output.append(ck[i*Norders+j])


        RL_output = np.array(RL_output)
        assert len(RL_output.shape)==4        

        for j in [1,2]:        
            if j ==1 :  
                opt.pg = pg
                pxint,pyint = get_intpxy(opt,RL_output[0],RL_output[j],i,j,methods='FAN') # 
                print(f'±2 Integer estimation: {pxint , pyint},length: {np.sqrt((pxint)**2+(pyint)**2)}')
                
                #subpxy
                pgx,pgy = get_subpxy(opt,i,j,(RL_output[0]),(RL_output[j]),pxint,pyint,func=LiheXYZ) 
                print(f'±2 Integer estimation{pgx , pgy},length: {np.sqrt((pgx)**2+(pgy)**2)}')
                
                PGxy[i,j,:] = np.array([pgx,pgy])  
                
            elif j == 2: ##########################
                PGxy[i,j,:] = -PGxy[i,1,:]
            

    lihe2d = np.array(opt.lihe2d)
    kzz,kxx,kyy = lihe2d.shape
    lihe2d[:,kxx//2,kyy//2] = 0.1
    lihe2d = np.insert(lihe2d,0,values = np.max(lihe2d,axis=0),axis=0) 
    lihe2d = np.insert(lihe2d,0,values = np.min(lihe2d,axis=0),axis=0) 

    Dr_sub = run_parameter['Dr_sub']
    return PGxy



def parameter_estimation_02(opt,ck,PGxy,Norders,filename='',RL=False,Debug=False,run_parameter=''):
    'return cfai (3,3,2) by PGxy'
    assert len(ck.shape)==4
    
    cfai = np.zeros((opt.Nangles,3,3)) #
    try:
        ck=ck.numpy()
    except:
        pass
    
    opt.lihe2d = []
    opt.overlap_mask2d = [] 
    opt.abscm2d = []
    opt.angcm2d = []

    opt.lihe3d = []
    opt.lihe3dxy = []
    opt.overlap_mask3d = [] 
    opt.abscm3d = []
    opt.angcm3d = []
    ''' <<<<<<<<<<<<<<<parameter estimate >>>>>>>>>>>>>>>>>>'''
    starttime = datetime.datetime.now()
    for i in range(3):        
        if RL:
            RL_output=[]
            for j in range(Norders):
                RL_output.append(_U.RL_deconvolve(ck[i*Norders+j], opt.OTF,fre = True) )
        else: #NO RL
            RL_output=[]
            for j in range(Norders):
                RL_output.append(ck[i*Norders+j])
                
        RL_output = np.array(RL_output)
        assert len(RL_output.shape)==4
        cfai[i,0,0] = 1 
        
        for j in range(1,Norders):
            pgx = PGxy[i,j,0]
            pgy = PGxy[i,j,1]          
            if j <=1 :  # ±2 order
                
                c,fai,c_EMD= cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),filename=filename,run_parameter=run_parameter)             

                cfai[i,j,:] = np.array([c,fai,c_EMD])

            elif j == 2: 
                cfai[i,j,0] = cfai[i,1,0]
                cfai[i,j,2] = cfai[i,1,2]
                cfai[i,j,1] = -cfai[i,1,1]
                
    return cfai




