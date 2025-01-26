import numpy as np
from pathlib import Path
import DataProcess as _DP
import Para_Estimate as _PE
import Seperate_CK as _SC
import Utils as _U
import Wiener_Reconstruction as _WR
import WindowF as _WF

def parameter_Finalsetting(opt,PGxy,cfai_02_ck1,cfai_1jie,cfai_02_ck2,run_parameter):
    cfai = np.zeros((3,7,3))
    cfai[:,[0,1,2],:] = cfai_02_ck1
    cfai[:,[3,4,5,6],:] = cfai_1jie

    cfai_02  = np.copy(cfai_02_ck2)
    cfai[:,0,2] = 1
    
    PGz = np.zeros((3,7))
    if run_parameter['pzzz'] != 0 and opt.flag_N=='MP':
        PGz[:,[3,5]] = -run_parameter['pzzz']
        PGz[:,[4,6]] = run_parameter['pzzz']
    
    P_long = (np.sqrt(PGxy[:,:,0]**2 + PGxy[:,:,1]**2)).round(2)
    Plong = np.mean(P_long[:,1])
    

    cfai_save = np.copy(cfai)
    cfai_02_save = np.copy(cfai_02_ck2)
    if run_parameter['remove2']==1:
        cfai[:,[0,1,2],1] = 0           
        cfai_02[:,[0,1,2],1] = 0
    return PGxy,cfai,cfai_save,cfai_02,cfai_02_save,PGz,P_long,Plong 


def main(opt,file_name1,file_name2,file_name3,run_parameter,isdebug=False):
    #file operation
    p1 = Path(str(file_name1)).iterdir() #loop
    p2 = Path(str(file_name2)).iterdir() #
    
    files = list(Path(str(file_name1)).iterdir())
    
    if Path(str(file_name2)).is_dir() and not any(p2): #OTF check
        raise('OTF not found based on the OTF path')
    else:
        p2 = Path(str(file_name2)).iterdir() #
    

    same_region_fai = []  
    same_region_pxy = []
    p1_i_ = -1
    for p1_i in p1:
        if not p1_i.is_file():           
            continue
        
        p1_i_+=1  
        if opt.sequence==55 and p1_i_%2==1 : 
            print("Skip:",p1_i)
            continue
        
        print("p1_i:",p1_i)
        filename = p1_i.name[-15:-11] 
        run_parameter['filename'] = filename
        # print('**Suffix**:',filename)
        
        #######################################1[ load Dr ]#########################################
        print()
        print("1 load raw data------------")
        #'raw data'
        Dr4D_ZPA,opt = _DP.loadANDprocess_Dr(opt,p1_i,run_parameter) #load; Ajust; ZPA; windows

        #Zpadding
        Dr = _DP.Image_zpadding(opt,Dr4D_ZPA,xishu=run_parameter['pad_xishu'])
        del Dr4D_ZPA
        
        #save WF
        if isdebug['loa']==1:
            movex = run_parameter['movex']
            movey = run_parameter['movey']
            _U.save_tif_v21(rf'WF8-rawdataSUM-{movex,movey}.tif',np.mean(Dr,axis=0),run_parameter)#
            _U.save_tif_v21('Dr.tif',Dr[[0,1,2]],run_parameter)
        
        #
        if opt.sequence==55 or opt.sequence==5343 or opt.sequence==5352:
            #Read in another set of time points
            idx = p1_i_+1
            p2_i = p1_i.parent/files[idx].name
            Dr4D_ZPA2,opt = _DP.loadANDprocess_Dr(opt,p2_i,run_parameter)
            
            Dr_2 = _DP.Image_zpadding(opt,Dr4D_ZPA2,xishu=run_parameter['pad_xishu'])

            del Dr4D_ZPA2
            
            if isdebug['loa']==1:
                _U.save_tif_v21('Dr_2.tif',Dr_2[[0,1,2]],run_parameter)
            
        #####################################2[ OTF] #####################################################
        print()
        print("2 load OTF ---------")
        #load_OTF
        if run_parameter['createOTF']==0:       #load otf
            for p2_k in p2:
                if not p2_k.is_file():           #
                    continue                
                #MP/2D       (nz，nx，ny)    (nx,ny)
                OTF3D,OTF2D,psf = _DP.load_OTF(opt,p2_k,Debug=isdebug['loa'])
                OTF3D[abs(OTF3D)<1e-10]=0
                OTF2D[abs(OTF2D)<1e-10]=0
                
                if isdebug['loa']==1:
                    _U.save_tif_v21('OTF3D_load.tif',abs(OTF3D),run_parameter)
                    _U.save_tif_v21('OTF2D_load.tif',abs(OTF2D),run_parameter)
        else:
            raise('can\'t generate 3DSIM OTF,please input OTF ')
        
        #OTF
        if opt.nz !=1 and opt.Zpadding!=0:
            nz,nx,ny = OTF3D.shape
            OTF3D = _U.Interpolate(OTF3D,nx=nx,ny=ny,nz=opt.nz+2*opt.Zpadding)
        
        #####################################3 [OTFmask]#####################################################
        print()
        print("3 OTFmask generate---------")
        # OTFmask3D/2D
        OTFmask3D,OTFmask2D = _DP.generatemask(opt,isdebug)
        
        assert len(OTFmask3D.shape)==3                 # (nz_re，2nx，2ny)
        assert len(OTFmask2D.shape)==2                 # (2nx,2ny)
        
        if isdebug['loa']==1:
            _U.save_tif_v21( 'OTFmask3D.tif', OTFmask3D, run_parameter )
            

        ##################################[Lateralx2  Axialx2] OTF3D OTF2D、OTFmask3D OTFmask2D  #######################
        
        opt.OTF = _U.pad_or_cut((OTF3D),x=2*opt.nx,y=2*opt.ny,z=opt.nz_re)
        opt.OTF2D = _U.pad_or_cut((OTF2D),x=2*opt.nx,y=2*opt.ny,z=1)
        # opt.psf = psf
        opt.OTFmask3D = _U.pad_or_cut(OTFmask3D,x=2*opt.nx,y=2*opt.ny,z=opt.nz_re)
        opt.OTFmask2D = _U.pad_or_cut(OTFmask2D,x=2*opt.nx,y=2*opt.ny,z=1)

        if isdebug['loa']==1:
            _U.save_tif_v21( 'opt.OTF.tif', abs(opt.OTF), run_parameter )
            _U.save_tif_v21( 'opt.OTF2D.tif', abs(opt.OTF2D), run_parameter )
            _U.save_tif_v21( 'opt.OTFmask3D.tif', opt.OTFmask3D, run_parameter )
            _U.save_tif_v21( 'opt.OTFmask2D.tif', opt.OTFmask2D, run_parameter )
        
    
        #####################################4 [separate ]##############################
        print()
        print("4 separate bands ------------")    
        for alpha_i in [0]:
            if opt.sequence==55:
                #Dr,Dr_2
                assert Dr.shape[0] == 15
                assert Dr_2.shape[0] == 15
                print("first data sep.....")
                ck1 = _SC.separate_step_one(opt,Dr,run_parameter,Nshifts=5,Debug=isdebug['sep'])  
                print("second data sep.....")
                ck2 = _SC.separate_step_one(opt,Dr_2,run_parameter,Nshifts=5,Debug=isdebug['sep'])  

                
                print("4-1 Parameter Estimation - 1-level ----- Primarily for 2nd Order PGxy-----------")
                flag_temp = 0 #  0 not load  1 load
                # cfai_file = _U.save_path(f'cfai',run_parameter)   #make 'cfai' folder
                'get PGxy(3,5,3)'
                # if not cfai_file.exists():
                #     cfai_file.mkdir()
                
                if run_parameter['loadflag'] == 0:                #cal all
                    '(3,5,2)'
                    PGxy = _PE.parameter_estimationP2(opt,ck1,Debug=isdebug['est'],filename=filename+'5x5',run_parameter=run_parameter)
                elif run_parameter['loadflag'] == 1:
                    '(3,7,2)/(3,7,3)'
                    cfai,PGxy,_,_ = _PE.loadparameter(opt,file_name3,run_parameter)
                    if run_parameter['remove2']==1 or run_parameter['joint_deconvolution']==1: 
                        cfai02 = _PE.loadparameter4(opt,file_name3,run_parameter)
                    flag_temp = 1
                elif run_parameter['loadflag'] == 3:                #load PGXY, cal pz/cfai + not save
                    '(3,7,2)'
                    PGxy,_,_ = _PE.loadparameter2(opt,file_name3,run_parameter) 
                    
                elif run_parameter['loadflag'] == 4:                #try load PGxy，load cfai  [0+3]
                    try:  #load P
                        PGxy,_,_ = _PE.loadparameter2(opt,file_name3,run_parameter) 
                    except:
                        pass
                    if flag_temp==0: # 
                        PGxy = _PE.parameter_estimationP2(opt,ck1,Debug=isdebug['est'],filename=filename+'5x5',run_parameter=run_parameter)
                            
                elif run_parameter['loadflag'] == 5:                #try load PGXY，CFAI, else cal PGXY，CFAI 
                    try:   #flag_temp=1
                        cfai,PGxy,_,_ = _PE.loadparameter(opt,file_name3,run_parameter)
                        if run_parameter['remove2']==1 or run_parameter['joint_deconvolution']==1: 
                            cfai02 = _PE.loadparameter4(opt,file_name3,run_parameter)
                        flag_temp = 1
                    except:  #
                        pass
                    if flag_temp==0: 
                        PGxy = _PE.parameter_estimationP2(opt,ck1,Debug=isdebug['est'],filename=filename+'5x5',run_parameter=run_parameter)
                
                PGxy_2 = PGxy[:,[0,1,2],:]   #PGxy_2（3,3）
               
                'Calculate the second-order initial phase # Obtain the second-order initial phase as (3,3,3)'
                print("4-1 Parameter Estimation - 1-level ----- Primarily for 2nd Order cfai-----------")
                if flag_temp == 0 : #
                    cfai_02_ck1 = _PE.parameter_estimation_02(opt,ck1[[0,1,2,5,6,7,10,11,12]],PGxy_2,Norders=3,Debug=isdebug['est'],filename=filename+"-ck1",run_parameter=run_parameter)
                    cfai_02_ck2 = _PE.parameter_estimation_02(opt,ck2[[0,1,2,5,6,7,10,11,12]],PGxy_2,Norders=3,Debug=isdebug['est'],filename=filename+"-ck2",run_parameter=run_parameter)
                
                elif flag_temp == 1 :   #load get 2th order phase
                    cfai_02_ck1 = cfai[:,[0,1,2],:]   #load from file
                    cfai_02_ck2 = cfai02              #load from file
                    
                "Obtain CK 7 components (whether to eliminate initial phase before calculation) " #
                if run_parameter['remove2']==1:
                    cfai_021_ck1 = np.zeros((3,5,3))
                    cfai_021_ck2 = np.zeros((3,5,3))
                    for cfai_021_ck1_i in range(3): 
                        cfai_021_ck1[cfai_021_ck1_i,0:3] = cfai_02_ck1[cfai_021_ck1_i,0:3]
                        cfai_021_ck1[cfai_021_ck1_i,3:5] = cfai_02_ck1[cfai_021_ck1_i,1:3]/2
                            
                        cfai_021_ck2[cfai_021_ck1_i,0:3] = cfai_02_ck2[cfai_021_ck1_i,0:3]
                        cfai_021_ck2[cfai_021_ck1_i,3:5] = cfai_02_ck2[cfai_021_ck1_i,1:3]/2
                            
                    ck1_wofai = _WR.removefai(opt,ck1,cfai_021_ck1)
                    ck2_wofai = _WR.removefai(opt,ck2,cfai_021_ck2) #
                    
                    ck = _SC.GETck(opt,ck1_wofai,ck2_wofai,run_parameter) # ck 7
                    
                else:
                    ck = _SC.GETck(opt,ck1,ck2,run_parameter) #ck 7
                
                
                #Joint deconvolution
                if opt.sequence==55 and run_parameter['joint_deconvolution']==1: #
                    ck_02 = np.zeros_like(ck[0:9])
                    for iii in range(3):
                        for jjj in [0,1,2]:
                            if run_parameter['remove2']==1:  
                                ck_02[3*iii+jjj] = np.copy(ck2_wofai[iii*5+jjj])
                            else:
                                ck_02[3*iii+jjj] = np.copy(ck2[iii*5+jjj]) 
            else:
                raise('sequcen error')
            
            #todo
            if run_parameter['use_mask']==1:
                if run_parameter['meas'] ==1:
                    jieshu = [0]
                    jieshu2 = [0]
                else:
                    jieshu = [0,1,2,3,4,5,6]
                    jieshu2 = [0,1,2]
                    
                for iiii in range(3):
                    for jjjj in jieshu:
                        ck[iiii*7+jjjj] = ck[iiii*7+jjjj] * _U.pad_or_cut(opt.OTFmask3D,opt.nx,opt.ny,opt.nz+2*opt.Zpadding)
                    if run_parameter['joint_deconvolution']==1:
                        for kkkk in jieshu2:
                            ck_02[iiii*3+kkkk] = ck_02[iiii*3+kkkk] * _U.pad_or_cut(opt.OTFmask3D,opt.nx,opt.ny,opt.nz+2*opt.Zpadding)

            filename = p1_i.name[-8:-4] 
            assert len(ck.shape)==4               # (21,nz,nx,ny)

            #mark
            # if isdebug['sep']==1:
            #     _U.save_tif_v21(f'【】ck-[0123456]-{filename}-r.tif',abs( ck[[0,1,2,3,4,5,6]] ),run_parameter)
        
        
            del ck1,ck2
            if run_parameter['remove2']==1:
                del ck1_wofai,ck2_wofai
            ######################################5 [1 order parameter estimate]##############################
            print()
            print("4-2 Parameter Estimation 2-Level ----- Primarily for 1st Order cfai-----------")

            'PGxy_2(3,3,2)->(3,7,2)'
            PGxy = np.zeros((3,7,2))
            PGxy[:,0:3,:] = PGxy_2
            PGxy[:,3,:] = PGxy[:,1,:] / 2
            PGxy[:,4,:] = PGxy[:,1,:] / 2
            PGxy[:,5,:] = PGxy[:,2,:] / 2
            PGxy[:,6,:] = PGxy[:,2,:] / 2

            'get 1 order parameter'
            if flag_temp==0: 
                cfai_1jie7,_,_,_,_ = _PE.parameter_estimation3(opt,ck,PGxy,Debug=isdebug['est'],filename=filename+'2 level',run_parameter=run_parameter,jieshu = [3,4,5,6]) 
                cfai_1jie = cfai_1jie7[:,[3,4,5,6],:]
            elif flag_temp==1: 
                cfai_1jie = cfai[:,[3,4,5,6],:]
            
            'save,princt PGxy(3,7,2)、Cfai(3,7,3) '
            PGxy,cfai,cfai_save,cfai_02,cfai_02_save,PGz,P_long,plong = parameter_Finalsetting(opt,PGxy,cfai_02_ck1,cfai_1jie,cfai_02_ck2,run_parameter)

            # if flag_temp==0: #mark
            #     _PE.saveparameter(file_name3,PGxy,PGz,cfai_save,run_parameter)  
            #     if opt.sequence==55 and run_parameter['joint_deconvolution']==1:  
            #         _PE.saveparameter02(file_name3,cfai_02_save,run_parameter)         
                            
            same_region_fai.append(cfai)  
            same_region_pxy.append(PGxy)
            
            opt.plong = plong             
            
            run_parameter['PGz_avg'] = np.mean(abs(PGz[:,-4:None])).round(3)
            
            #####Parameter Estimation Completed#####
            if run_parameter['just_cfai']:
                break
            
            ########################################[Reconstruction]############################
            print()
            print(" 5 Prepare for Reconstruction--------")
            ###########
            print(" 5-1 making Ir -------")
            Irtest = _WR.IrTEST(opt,PGxy,PGz,kuodaz=opt.kuodapz,Zchushu=opt.kuodapz*(opt.nz+2*opt.Zpadding))

            directions = run_parameter['directions'] 
            BattStrFZs = run_parameter['attStrFZs']
            BattFWHMFZs = run_parameter['attFWHMFZs']
            BattStrs = run_parameter['attStrs']
            BattFWHMts = run_parameter['attFWHMs']
            
            for direction in directions:                           #【Which Directions to Reconstruct】
                for order_flag in run_parameter['orderForRe']:     #【Which Orders to Reconstruc】
                    for Method_c in run_parameter['Method_c']:        
                        for notch in run_parameter['notch']:        
                            if notch==0:   
                                run_parameter['attStrFZs'] = [0]
                                run_parameter['attFWHMFZs'] = [1]
                                run_parameter['attStrs'] = [0]
                                run_parameter['attFWHMs'] = [1]
                            else:         
                                run_parameter['attStrFZs'] = BattStrFZs
                                run_parameter['attFWHMFZs'] = BattFWHMFZs
                                run_parameter['attStrs'] = BattStrs
                                run_parameter['attFWHMs'] = BattFWHMts
                            
                            FZotf,FZotfmask = _WR.GetFZotf(opt,Irtest,order_flag=order_flag,run_parameter=run_parameter,PGxy=PGxy,PGz=PGz,direction=direction)
                            FZotf = abs(FZotf)
                            
                            if isdebug['mov']==1:
                                _U.save_tif_v21(r'FZotf-abs.tif',abs(FZotf[[0,1,2,3,4]]),run_parameter)
                                # _U.save_tif_v21(r'Msum_mask.tif',abs(Msum_mask),run_parameter)
                            
                            if run_parameter['joint_deconvolution']==0:
                                FZall = _WR.GetFZ(opt,ck,Irtest,order_flag=order_flag,run_parameter=run_parameter,direction=direction)
                            elif opt.sequence==55 and run_parameter['joint_deconvolution']==1:
                                FZall,FZall_02 = _WR.GetFZ_Joint(opt,ck,ck_02,Irtest,order_flag=order_flag,run_parameter=run_parameter,direction=direction)
                            
                            if isdebug['mov']==1:
                                _U.save_tif_v21(r'FZall.tif',abs(FZall[[0,1,2,3,4]]),run_parameter)
                            
                            # for notch0th in run_parameter['notch0ths']: 
                            run_parameter['notch0th'] = run_parameter['notch0ths'][0]   
                                # for attFWHM0 in run_parameter['attFWHM0s']:
                            run_parameter['attFWHM0'] = run_parameter['attFWHM0s'][0]
                                    # for notch1st_pang in run_parameter['notch1st_pangs']: 
                            run_parameter['notch1st_pang'] = run_parameter['notch1st_pangs'][0]
                                        # for attStrFZ in run_parameter['attStrFZs']:#
                            run_parameter['attStrFZ'] = run_parameter['attStrFZs'][0]
                                            # for attFWHMFZ in run_parameter['attFWHMFZs']:
                            run_parameter['attFWHMFZ'] = run_parameter['attFWHMFZs'][0]
                                                # for attStr in run_parameter['attStrs']:#
                            run_parameter['attStr'] = run_parameter['attStrs'][0]
                                                    # for attFWHM in run_parameter['attFWHMs']:
                            run_parameter['attFWHM'] = run_parameter['attFWHMs'][0]
                                                        

                            if run_parameter['joint_deconvolution']==0:
                                Wiener1 = _WR.GetWiener1(opt,_U.ToNumpy(FZall),_U.ToNumpy(FZotf),PGxy,PGz,cfai,1,order_flag,run_parameter,notch,direction=[0,1,2])
                            elif opt.sequence==55 and run_parameter['joint_deconvolution']==1:
                                Wiener1 = _WR.GetWiener1_Joint(opt,_U.ToNumpy(FZall),_U.ToNumpy(FZall_02),_U.ToNumpy(FZotf),PGxy,PGz,cfai,cfai_02,1,order_flag,run_parameter,notch,direction=[0,1,2])
                            
                            if isdebug['wie']==1:
                                _U.save_tif_v21('Wiener1-k-.tif',abs(Wiener1),run_parameter)
                            
                            for w2 in opt.W:
                                if  run_parameter['joint_deconvolution']==0:
                                    filename2 = filename
                                    Msum_all = _WR.GetWiener2(opt,_U.ToNumpy(Wiener1),_U.ToNumpy(FZotf),PGxy,PGz,cfai,w2,order_flag,run_parameter,notch,direction=[0,1,2])
                                elif opt.sequence==55 and run_parameter['joint_deconvolution']==1: 
                                    filename2 = filename+'J'
                                    Msum_all = _WR.GetWiener2_Joint(opt,_U.ToNumpy(Wiener1),_U.ToNumpy(FZotf),PGxy,PGz,cfai,cfai_02,w2,order_flag,run_parameter,notch,direction=[0,1,2])
                                
                                'Wiener'
                                Apo = _WF.bhs_3D_Z(opt,opt.plongxs*opt.plong,opt.fcxs*opt.fc,kuodaz=opt.kuodapz, Zjust=opt.Apoz )
                                
                                Wiener2 = Wiener1 / Msum_all
                                Wiener2 = Wiener2 * Apo
                                
                                _WR.Final_ifft(opt,_U.ToNumpy(Wiener2),Method_c,w2,notch,order_flag,direction,run_parameter=run_parameter,filename=filename2,p1_i_=p1_i_)
        
        if  run_parameter['just_cfai']==1 or run_parameter['fast']==1:
            break
    if run_parameter['just_cfai']: 
        print(r'parameter cal finnish')
        return 

    return 

   