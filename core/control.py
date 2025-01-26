import tkinter
import tkinter.filedialog
from ui import Win
from tkinter import messagebox
import tkinter as tk
from pathlib import Path
import core.run1_488 as _rrr


class TextRedirector(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass
    
class Controller:
    # After importing the UI class, replace the following object type to enable IDE attribute hints
    ui: Win
    def __init__(self):
        pass
    def init(self, ui):
        """
        Obtain the UI instance and perform initialization configuration for the components
        """
        self.ui = ui
        
        #lable
        self.ui.tk_label_Data_reconstruction.configure(font=('Segoe UI ',15))
        self.ui.tk_label_condition.configure(font=('Segoe UI ',10,"bold"))
        self.ui.tk_label_condition.config(text=f"Select a folder and input Wiener parameter,then run it")
        #input
        self.ui.tk_input_input_folder_path.insert(0,r" select or copy path of raw data.....")
        # self.ui.tk_input_frames.insert(0,30)
        # self.ui.tk_input_time_point.insert(0,1)
        self.ui.tk_input_Wiener.insert(0,1e-4)
        # self.ui.tk_input_Channel.insert(0,488)



    def select_folder_path(self,evt): 
        pathh = Path.cwd()
        path = tkinter.filedialog.askdirectory(initialdir=pathh)
        if path:
            self.ui.tk_input_input_folder_path.delete(0,tkinter.END)
            self.ui.tk_input_input_folder_path.insert(0,f'{path}')
    

        
    def Run_All(self,evt):
        self.ui.tk_label_condition.config(text=f"Data is preprocessing...")
        # self.ui.tk_text_print.insert('0.0',"Data is preprocessing...\n")
        'path'
        data_path = self.ui.tk_input_input_folder_path.get()
        if data_path == '':
            messagebox.showerror(title='error!',message='please input data path ')

        ################################RECON########################################   
        para = dict()

        para['data_path']  = data_path

        'w, notch'
        w = float(self.ui.tk_input_Wiener.get())
        # Channel = float(self.ui.tk_input_Channel.get())
        Channel = 488
        notch = 0.6#self.ui.tk_scale_notch_low_strength.get()
        loadflag = 0#self.loadflag.get()
        fast = 0#self.fast.get()
        interpolation = 0#self.interpolation.get()
        
        self.ui.tk_label_condition.config(text=f"Data is being reconstructed")

        _rrr.Run_recon(para,w,Channel)  #main function

        self.ui.tk_label_condition.config(text=f"Data is reconstructed! ")

        messagebox.showinfo("Data is reconstructed!", f"results address:  {data_path}/Result")
    
        return 0