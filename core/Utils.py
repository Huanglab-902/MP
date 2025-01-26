import numpy as np
from numpy import pi, cos, sin,exp
import torch 
from pathlib import Path
# from loguru import logger
import math
import cv2
import Utils as _U

from tifffile import imread, imwrite,TiffWriter


def fftn(data, axes=None):
    """utility method that includes shifting"""
    return np.fft.fftshift(np.fft.fftn(data))

def ifftn(data, axes=None):
    """utility method that includes shifting"""
    return np.fft.ifftn(np.fft.ifftshift(data))

def tfftn(data, axes=None):
    """utility method that includes shifting"""
    return torch.fft.fftshift(torch.fft.fftn(data))
def tifftn(data, axes=None):
    """utility method that includes shifting"""
    return torch.fft.ifftn(torch.fft.ifftshift(data))



def fftshift(data):
    return np.fft.fftshift(data)
def visual(ik): 
    return np.log(abs(ik)+1)

#change dim
def image_reshape(image,dim=4):
    """
    Transform Dr into 4D, setting nc, nz, nx, and ny to the desired dimensions.
    """
    cur_dim = len(image.shape)
    
    if cur_dim==dim:
        return image             
    
    while (cur_dim<dim):       # + dim
        image=np.expand_dims(image, axis=0)
        cur_dim = len(image.shape)

    while (cur_dim>dim):       # - dim
        shape_array = image.shape
        shape_array2 = []
        shape_array2.append(shape_array[0]*shape_array[1])
        shape_array2[1:] = shape_array[2:]
        image = image.reshape(shape_array2)
        cur_dim = len(image.shape)
    return image


#pad or cut： xy，z
def pad_or_cut(image,x=512,y=512,z=1,mode = 'constant',biasx=0,biasy=0):
    """
    Croppable and zero-padding enabled, requires input of the final image size
    mode = 'constant' (default)
        Pads with a constant value.
    'edge'
        Pads with the edge values of array.
     ....
    """

    #type
    flag_dtype=0
    if isinstance(image,torch.Tensor) :
        flag_dtype = 1
    dim_before = len(image.shape)

    nz,nx,ny = z,x,y
    image = image_reshape(image,dim = 4) 
    raw_nz,raw_nx,raw_ny = image.shape[1:]  
    
    #pad
    nz_left,nz_right =  pad_or_cut_cal(nz,raw_nz)
    nx_left,nx_right =  pad_or_cut_cal(nx,raw_nx)
    ny_left,ny_right =  pad_or_cut_cal(ny,raw_ny)
    
    image = image_reshape(image,dim = 4) 
    pad = ( (0,0),
            (nz_left,nz_right),
            (nx_left,nx_right), 
            (ny_left,ny_right) )
    image_adjust = np.pad(image,pad,mode=mode) 
    
    #cut
    if nx<raw_nx or ny<raw_ny or nz<raw_nz:
        image_adjust = crop(image_adjust,nx=nx,ny=ny,nz=nz,biasx=biasx,biasy=biasy)

    #back to 
    if flag_dtype == 1 :  
        image_adjust = torch.from_numpy(image_adjust)

    image_adjust = image_reshape(image_adjust,dim = dim_before) 
    return image_adjust
    
def pad_or_cut_cal(target_length,raw_length):
    left=0
    right=0
    if target_length > raw_length:
        Diff_nx = (target_length-raw_length)/2
        left,right = Diff_nx,Diff_nx # 
        if((target_length-raw_length)%2==1 and right !=0 ):    #If it's an odd number, then add 1 more on the left; the corresponding trimming would also be on the left
            left +=1 
    return int(left),int(right)

def crop(image,scale=0.5,nx=0,ny=0,nz=0,biasx=0,biasy=0):
    image = image_reshape(image,dim = 4) 
    _,raw_nz,raw_nx,raw_ny = image.shape
    if scale==0.5 and nx ==0 and ny==0: 
        nx = raw_nx//2
        ny = raw_ny//2
        
    nx_stare = int(np.ceil((raw_nx-nx)/2))
    nx_end = nx_stare+nx
    ny_stare = int(np.ceil((raw_ny-ny)/2))
    ny_end = ny_stare+ny
    # print(nx_stare,ny_stare)
    nz_stare = int(np.ceil((raw_nz-nz)/2))
    nz_end = nz_stare+nz 
    image_small = image[:,nz_stare:nz_end,nx_stare+biasx:nx_end+biasx ,ny_stare+biasy:ny_end+biasy ]
    
    return image_small


#change type to torch
def ToTensor(h):
    try:
        h = torch.from_numpy(h)
    except:
        pass
    return h
def ToNumpy(h):
    try:
        h = (h).numpy()
    except:
        pass
    return h


def meshgrid(opt,kuodaxy=1,kuodaz=1,index='ij'):
    'XYZ'
    nx = opt.nx*kuodaxy
    ny = opt.ny*kuodaxy
    nz = opt.nz_re
    x = np.arange(np.ceil(-nx/2),np.ceil(nx/2))
    y = np.arange(np.ceil(-ny/2),np.ceil(ny/2))
    z = np.arange(np.ceil(-nz/2),np.ceil(nz/2))

    zv, xv, yv = np.meshgrid(z, x, y, indexing=index, sparse=True)
    return zv,xv,yv

def meshgridxy(opt,kuodaxy=1,kuodaz=1,index='ij'):
    'XY'
    nx = opt.nx*kuodaxy
    ny = opt.ny*kuodaxy
    x = np.arange(np.ceil(-nx/2),np.ceil(nx/2))
    y = np.arange(np.ceil(-ny/2),np.ceil(ny/2))
    xv, yv = np.meshgrid(x, y, indexing=index, sparse=True)
    return xv,yv
def meshgridxy_self(nx,ny,nz=1,index='ij'):
    'XYZ set nx ny nz'
    x = np.arange(np.ceil(-nx/2),np.ceil(nx/2))
    y = np.arange(np.ceil(-ny/2),np.ceil(ny/2))
    z = np.arange(np.ceil(-nz/2),np.ceil(nz/2))
    zv, xv, yv = np.meshgrid(z, x, y, indexing=index, sparse=True)
    return zv,xv,yv

#move
def shiftmat(opt,kx=0,ky=0,kz=0):
    movemat = np.exp(2j*np.pi*(kx*opt.xv+ky*opt.yv)) * np.exp(2j*np.pi*kz*opt.zv)
    return movemat

# torch function of Interpolate
def Interpolate(image,nx,ny,nz,strt = 'trilinear'):
    """
     mode (str): algorithm used for upsampling:
        ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
        ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
    """
    
    before_dim = len(image.shape)

    image = image_reshape(image,dim=5)   #1,cc,nz,nx,ny)

    #to torch
    image = _U.ToTensor(image) 

    if torch.is_complex(image)!=True : 
        image_inter = torch.nn.functional.interpolate((image),size=[nz,nx,ny],mode=strt,align_corners=False)
    else: 
        image_real = image.real
        image_imag = image.imag
        image_real_inter = torch.nn.functional.interpolate((image_real),size=[nz,nx,ny],mode=strt,align_corners=False)
        image_imag_inter = torch.nn.functional.interpolate((image_imag),size=[nz,nx,ny],mode=strt,align_corners=False)
        image_inter = image_real_inter+1j*image_imag_inter
        
    return image_reshape(_U.ToNumpy(image_inter),dim=before_dim)

#save tif
def save_tif(filename,data):
    """
    process: if new_dir!= None  make folder
    """
    # data = image_reshape(data,dim=3)
    # path = Path.cwd()/'dataOut'
    # if not path.exists():
    #     path.mkdir()
    #     print(f' maker new folder :{path}')
    
    # savefolder = path/Path.cwd().name  #
    # if not savefolder.exists():
    #     savefolder.mkdir()
    #     print(f' maker new folder :{savefolder}')

    # savefile = savefolder/filename
    # try:
    #     data = data.numpy()
    # except:
    #     pass

    # imwrite(
    #     savefile,data.astype(np.float32),#.transpose(),
    #     imagej=True,
    #     photometric='minisblack',
    #     metadata={'axes': 'ZYX'}
    #     )
    # print(f"data successly save in ({savefile})")
    return

def save_tif_v21(filename,data,run_parameter,sub_filename='',datatype=np.float32):
    # data = image_reshape(data,dim=3)
    # path = Path.cwd()/'dataOut'
    # if not path.exists():
    #     path.mkdir()
    
    # path_data = path/run_parameter['data_name']
    # if not path_data.exists():
    #     path_data.mkdir()

    # path_data_dir = path_data/(run_parameter['dir_name']+"-"+run_parameter['ver'])
    # if not path_data_dir.exists():
    #     path_data_dir.mkdir()

    # if sub_filename!='':
    #     path_data_dir = path_data_dir/sub_filename
    #     if not path_data_dir.exists():
    #         path_data_dir.mkdir()

    # savefile = path_data_dir/filename
    # try:
    #     data = data.numpy()
    # except:
    #     pass

    # imwrite(
    #     savefile,data.astype(datatype),
    #     imagej=True,
    #     photometric='minisblack',
    #     metadata={'axes': 'ZYX'}
    #     )
    # print(f"data successly save in ({savefile})")
    return

def save_txt(filename,data):
    """
    input：data，data{dict}
    """
    # path = Path.cwd()/'dataOut'
    # if not path.exists():
    #     path.mkdir()
    #     print(f' maker new folder :{path}')
    
    # savefolder = path/Path.cwd().name  #
    # if not savefolder.exists():
    #     savefolder.mkdir()
    #     print(f' maker new folder :{savefolder}')

    # savefile = filename
    
    # #
    # with open(savefile, 'w') as f:
    #     for key, value in data.items():
    #         f.write(key)
    #         f.write(': ')
    #         f.write(str(value))
    #         f.write('\n')

    # print(f"data successly save in ({savefile})")
    return


def save_hyperstack(path_filename,data,colormap=''):
    data = data/np.max(data)*65535
    data = data.astype(np.uint16)
    
    metadata = {
        'axes':'TZYX',
        'unit': 'um',
    }
    imwrite(path_filename,data,imagej=True,metadata=metadata)
    return 0



def save_tif_self(pathname,data):
    """
    input：path+file name，data
    """
    path = Path(str(pathname))  
    
    path1 = path.parent
    if not path1.exists():
        path1.mkdir()
        print(f' maker new folder :{path1}')

    imwrite(path,data,photometric='minisblack')
    print(f"data successly save in ({path})" )
    return

def save_tif_self3(path_data_dir,data,sub_filename='',subsub_filename='',filename = '',axes = 'ZYX',min=0,max=255):
    """
    save results
    """
    if axes=='ZYX':
        data = image_reshape(data,dim=3)

    try :
        path_data_dir = Path(path_data_dir)
    except:
        pass
    'Subfolder'
    if sub_filename!='':
        path_data_dir = path_data_dir/sub_filename
        if not path_data_dir.exists():
            path_data_dir.mkdir()
    'Subsubfolder'
    if subsub_filename!='':
        path_data_dir = path_data_dir/subsub_filename
        if not path_data_dir.exists():
            path_data_dir.mkdir()

    'save '
    savefile = path_data_dir/filename
    try:
        data = data.numpy()
    except:
        pass

    with TiffWriter(savefile,imagej=True) as tif:
        tif.write(data.astype(np.float32), metadata={'axes': axes,'min': min,'max': max})
    
    return

def save_path(filename,run_parameter,sub_filename=''):
    path = Path.cwd()/'dataOut'
    if not path.exists():
        path.mkdir()
    path_data = path/run_parameter['data_name']
    if not path_data.exists():
        path_data.mkdir()

    path_data_dir = path_data/(run_parameter['dir_name']+"-"+run_parameter['ver'])
    if not path_data_dir.exists():
        path_data_dir.mkdir()

    if sub_filename!='':
        path_data_dir = path_data_dir/sub_filename
        if not path_data_dir.exists():
            path_data_dir.mkdir()

    savefile = path_data_dir/filename

    return savefile



def save_tif_v22_path(run_parameter):
    path = Path.cwd()/'dataOut'
    if not path.exists():
        path.mkdir()

    path_data = path/(run_parameter['data_name']+"_only")
    if not path_data.exists():
        path_data.mkdir()

    path_data_dir = path_data/(run_parameter['dir_name']+"-"+run_parameter['ver'])
    if not path_data_dir.exists():
        path_data_dir.mkdir()

    return path_data_dir

def normalize(data):
    from math import sqrt
    
    if type(data) == complex:
        real = data.real
        imag = data.imag
        temp = sqrt(real*real+imag*imag)
        real = real/temp
        imag = imag/temp * 1j
        norm_data = real + imag
    elif isinstance(data,torch.Tensor):
        mag = torch.abs(data)
        max_mag = torch.max(mag)
        norm_data = data/max_mag
    elif isinstance(data,np.ndarray):
        mag = np.abs(data)
        max_mag = np.max(mag)
        norm_data = data/max_mag
    else:
        raise(r'error')
    
    return norm_data


def release():
    import gc
    gc.collect() 


def fill0(data,num=1):
    if data>1:
        return str(data).zfill(num)      
    else:
        return str(data).ljust(num,'0')  


###########################Morphological operation#############
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
        print('3D erosion is too severe; the size needs to be reduced.')
    return data_erode

def Cir_dilate(data,size=3):
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

    data_erode =cv2.dilate(data,kernel_re)
    return data_erode


def Cir_Erode_3D(data,size=3):
    data2=image_reshape(data,dim=3)
    nz,nx,ny = data2.shape
    num_ling = 0
    for i in range(nz):
        data2[i]=Cir_Erode(data[i],size=size)
        if np.mean(data[i])!=0 and np.mean(data2[i]) == 0:
            num_ling+=1
    
    if num_ling >= nz - 2 :
        _U.save_tif(r'查看腐蚀之后data2.tif',data2)
        raise '3D erosion is too severe; the size needs to be reduced.'
    
    return data2

def Cir_dilate_3D(data,size=3):
    data2=image_reshape(data,dim=3)
    nz,nx,ny = data2.shape
    for i in range(nz):
        data2[i]=Cir_dilate(data[i],size=size)
    return data2
##################################################


def flip180(arr,biaozhi):  
    # data can be 2D or 3D
    arr3D = image_reshape(arr,dim=3)
    
    assert len(arr3D.shape)==3
    arr3D_180 = np.zeros_like(arr3D)
    nz = arr3D_180.shape[0]
    
    if biaozhi=='lr':
        for i in range(nz):
            arr3D_180[-i - 1] = np.fliplr(arr3D[i])  
    elif biaozhi =='ud':
        for i in range(nz):
            arr3D_180[-i - 1] = np.flipud(arr3D[i])  
    return arr3D_180 



def find_dates(text):
    import re
    pattern = r"(\d{4})(\d{2})(\d{2})"
    matches = re.findall(pattern, text)
    formatted_dates = [f"{year}{month}{day}" for year, month, day in matches]
    return formatted_dates


