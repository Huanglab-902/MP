import Utils as _U
from pathlib import Path
import Main_recon_MP as _MMP

def main(opt,run_parameter,savepath0,OTFone,FANWEI=[1]):
    isdebug = run_parameter['DEBUG']   
    ##############################【parameter】extra debug parameter
    opt.version ='V0.1'
    run_parameter['ver']=opt.version
    if run_parameter['meas'] == 1:
        opt.Apoz = 600
        opt.fcxs = 1.7                  # Axial range of the apodization function 1
        opt.plongxs = 1.7
        run_parameter['note']= 'ma60'
    elif run_parameter['meas'] == 2:
        opt.Apoz = 830
        opt.fcxs = 1.2                  
        opt.plongxs = 1.2 
        run_parameter['note']= ''
    elif run_parameter['meas'] == 0:
        opt.Apoz = 750
        opt.fcxs = 1.2                  
        opt.plongxs = 1.2 
        run_parameter['note']= 'simu'
    ##################################【run_parameter】

    loop_dir = Path(str(savepath0)).iterdir() 
    data_name = str(Path(str(savepath0)).name) 
    run_parameter['data_name'] = data_name 
    print(data_name)
    
    i= 0       
    for ldr_i in loop_dir:
        i+=1
        dir_name = ldr_i.parts[-1]  
                
        # if i not in FANWEI:           # skip 
        #     print(f'Loop-FILE skip2,i={i}',dir_name)
        #     continue
        
        if ldr_i.is_dir() != 1:  #
            print('Loop-FILE-skip 3',dir_name)
            continue
        if 'raw data' not in dir_name:
            print('Loop-FILE-skip 4',dir_name)
            continue

        savepath1 = ldr_i/'Dr/'/run_parameter['Dr_sub']
        savepath2 = OTFone
        
        savepath3 = ldr_i/'parameter/'/run_parameter['W_order'] #
        if not savepath3.parent.exists():
            savepath3.parent.mkdir()
            
        if not savepath3.exists():
            savepath3.mkdir()
        
        print('Loop-FILE-Process ',dir_name)
        run_parameter['dir_name'] = dir_name

        _MMP.main(opt,savepath1,savepath2,savepath3,run_parameter,isdebug=isdebug)