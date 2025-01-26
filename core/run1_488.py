import Run_recon_MP as _RUN1
import argparse
import numpy as np
import Utils as _U

def opt_opical_parameter1():
    #488-525 oil 
    opt_op = argparse.Namespace()
    
    opt_op.n = 1.33                                   
    opt_op.theta = 72.7 * (np.pi / 180)   

    opt_op.NAzm = opt_op.n * np.sin(opt_op.theta)     # NA of Illumination optical path 
    opt_op.NAjs = 1.33                                # NA of Receiving optical path
    opt_op.na = opt_op.NAjs                           
    opt_op.namin = min(opt_op.NAjs, opt_op.n)         # min NA

    opt_op.Wzm = 488  # Ir  
    opt_op.Wjs = 525  # OTF

    opt_op.n_mask = 1.33

    # n*sinθ   
    opt_op.waterN = 1.33 #
    opt_op.lamdex = 488  #                      #Excitation wavelength
    opt_op.lamdem = 525  #                      #Emission wavelength

    return opt_op


def opt_RUN1_parameter1(opt_1):
    #base MP 5+5
    opt_1.flag_N  = "MP"    # imaging setting parameter MP=7
    opt_1.Nshifts = 5       # phase shifts for each stripe (raw data) 
    opt_1.Norders = 7       # after separate total orders  0，±2，(±1,±1) 3D-MP=7
    opt_1.Nangles = 3       # number of orientations of stripes
    opt_1.nc = opt_1.Nangles*opt_1.Norders  #
    opt_1.nz = 8
    opt_1.nx = 462                        
    opt_1.ny = 462
    opt_1.sequence = 55                   # 15+15 mode
    opt_1.faiXY = 2/5                     #XY  Lateral phase spacing
    opt_1.faiZ  = 0.52                    #Z   Axial phase spacing 
    opt_1.order7 = [0,1,2,3,4, 0,2]       #52(3x7) [2/5 π] mode order
    opt_1.order5 = [0,1,2,3,4]            #c15+15 [2/5 π] mode order 
    
    #pixelsize
    opt_1.dx = 60                         # Lateral piexlsize 
    opt_1.dz = 193.8                      # Axial piexlsize
    
    #Apoz 
    opt_1.Apoz = 830                      # Axial range of the apodization function 
    opt_1.fcxs = 1.2                      # Axial range of the apodization function 1
    opt_1.plongxs = 1.2                   # Axial range of the apodization function 2
    
    opt_1.W = [1e-4]
    
    return opt_1


def define_run_parameter():
    run_parameter = dict(
        createOTF=0,      #<--！！！-           # 0:Load OTF  1: generate OTF
        loadflag=0,      #<--！！！--           # 0:Recalculate parameters and save file; 
                                                # 1:load all parameters; 
                                                # 2:only load wave vector P, calculate modulation depth and phase, and save file; 
                                                # 3:only load wave vector P, calculate modulation depth and phase without saving the file
                                                # 4: [0 or 3]
                                                # 5: [0 or 1]
        just_cfai=0,                            # 1: only estimate parameters    0:  Reconstructed image 
                                                                                       
        fanwei=20,                              # Range for calculating the wave vector (20 define)
        
        pzzzs=[0],                              # Theoretical pz length
        pzzz_range=0,                           
        
        # notch   #################################################################
        notch=[0],             # 0: Not using first and second-order notch; 1: Not using first and second-order notch
                               # Note, this parameter does not control the 0th-order notch
        # 0-order notch                                                                           ##
        notch0ths=[0.6],      # 0th-order, Notch suppression level（0-1）
        attFWHM0s=[1.5],       # 0th-order, Notch range
        # 2-order notch                                                                       ##
        attStrFZs=[0.8],       # 2nd-order, Notch suppression level（0-1）
        attFWHMFZs=[1.5],      # 2nd-order, Notch range
        # 1-order notch                                                                         ##
        attStrs=[0.8],         # 1st-order, Notch suppression level（0-1）
        attFWHMs=[1.5],        # 1st-order,Notch range
        notch1st_pangs=[1],    # First-order side lobes (1 define)
        #########################################################################################
        
        directions=[[0,1,2]],  #Direction of reconstruction
        orderForRe=[6],        #Direction of reconstruction     #0 0th  ；1 ±1st ； 2 ±2th  #3 0±1   ；4 0±2  ；5 ±1±2   6 all (define)
        Method_c=[5],          #Method for eliminating modulation depth (5: define) 
        Msum_maskMove=1,       #Msum mask is 1 for generating (default) or 2 for shifting

        joint_deconvolution = 1,    #Joint deconvolution ,0 no,   1 yes (define)
        joints = 0,                 #0 （0,2 order）  or 1 (2 order) Control which orders of the joint deconvolution to apply.
        remove2 = 1,                #remove 2th phase
        Dr_sub = "01",              # cReconstructed sub-filec

        pad_xishu = 0,              #padding Coefficient
        #file
        save= 0  ,               # save some file
        fast = 0  ,              # Reconstructed sub-filec
        chazhi = 0    ,          # Whether to perform automatic axial interpolation

        XZ_save = 0,
        YZ_save = 0,

        piancha=0,                              # Amount of movement during surface parameter estimation
        flagV=1,                                # Whether to perform automatic axial interpolation: 1 (definecc)
        pg=0,                                   # The parameter size during estimation. If set to 0, it will be assigned a value using opt.pxy later; if 0, it will be calculated automatically
    )
    return run_parameter
    


def Get_path(run_parameter,para=''):
    # get path
    OTFone = r'./OTF'    
    run_parameter['data_path'] = para['data_path']  
    run_parameter['OTFname'] = OTFone         
    savepath0 = para['data_path']  
    return savepath0,OTFone,run_parameter

def Run_recon(para,w,Channel=488):
    # Initialize optical path parameters, run parameters
    opt = opt_opical_parameter1()   #488 oil 
    opt = opt_RUN1_parameter1(opt)  
    run_parameter = define_run_parameter() # running parameter
    savepath0,OTFone,run_parameter = Get_path(run_parameter,para) #set path
    
    if Channel==561:
        opt.Wzm = 561
        opt.Wjs = 600
        opt.lamdex = 561
        opt.lamdem = 600
        OTFone = r'./OTF_561'
        pz = 2.6
        Dr_sub_1 = "01-561-AVG"
        Dr_sub_2 = "01-561"
        W_order = "561-AVG-P"
    elif Channel == 488:
        pz = 3
        Dr_sub_1 = "01-488"
        Dr_sub_2 = "01-488"
        W_order = "488-P"
    else:
        raise('Channel only 488 or 561')
    
    
    run_parameter['movex']=0
    run_parameter['movey']=0
    run_parameter['movex_long'] = 462
    run_parameter['movey_long'] = 462
    run_parameter['use_mask'] = 1       
    run_parameter['meas'] = 2          
    
    opt.nz = 8
    opt.nx = 462
    opt.ny = 462
    opt.W = [w]
    pz8 = pz              #The number of axial layers is too small, leading to inaccurate cross-correlation. Assign a theoretical value to pz      
    opt.Zpadding = 4      #Z-padding,  Avoid duplication of the first and last layers caused by circular convolution.
    run_parameter['DEBUG'] = dict(save=1, loa=1,sep=1,est=1,mov=1,wie=1)



    
    '1. Data requiring padding, calculate parameters with Pre-average'
    # run_parameter['Dr_sub'] = Dr_sub_1           # Pre-average datas
    # run_parameter['W_order'] = W_order       # parameter file


    # run_parameter['loadflag'] = 0                    # Calculate parameters
    # run_parameter['pzzz'] = pz8/8*(opt.nz+2*opt.Zpadding) # cal PZ after z-padding
    # run_parameter['just_cfai'] = 1                   # Calculate parameters only

    # _RUN1.main(opt,run_parameter,savepath0,OTFone)    

    
    "2. Reconstruction"
    run_parameter['Dr_sub'] = Dr_sub_2               #Single time-point datas
    run_parameter['W_order'] = W_order       # parameter file

    run_parameter['notch'] = [0]           
    run_parameter['notch0ths']=[0.6]
    run_parameter['loadflag'] = 5                    # load parameter
    run_parameter['pzzz'] = pz8/8*(opt.nz+2*opt.Zpadding) 
    run_parameter['just_cfai'] = 0                   # Reconstruction 
    
 
    _RUN1.main(opt,run_parameter,savepath0,OTFone)  
    