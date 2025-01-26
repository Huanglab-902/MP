
import random
from tkinter import *
from tkinter.ttk import *
class WinGUI(Tk):
    def __init__(self):
        super().__init__()
        self.__win()
        self.tk_frame_layout_pre = self.__tk_frame_layout_pre(self)
        self.tk_input_input_folder_path = self.__tk_input_input_folder_path( self.tk_frame_layout_pre) 
        self.tk_button_select_folder_path = self.__tk_button_select_folder_path( self.tk_frame_layout_pre) 
        self.tk_label_Wiener_ = self.__tk_label_Wiener_( self.tk_frame_layout_pre) 
        self.tk_input_Wiener = self.__tk_input_Wiener( self.tk_frame_layout_pre) 
        
        # self.tk_label_Channel_ = self.__tk_label_Channel_( self.tk_frame_layout_pre) 
        # self.tk_input_Channel = self.__tk_input_Channel( self.tk_frame_layout_pre) 
        
        self.tk_button_Run = self.__tk_button_Run( self.tk_frame_layout_pre) 
        self.tk_label_Data_reconstruction = self.__tk_label_Data_reconstruction(self)
        self.tk_label_condition = self.__tk_label_condition(self)
        
    def __win(self):
        self.title("3D-MP-SIM program")
        width = 528
        height = 280
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)
        
        self.resizable(width=False, height=False)
        
    def scrollbar_autohide(self,vbar, hbar, widget):
        def show():
            if vbar: vbar.lift(widget)
            if hbar: hbar.lift(widget)
        def hide():
            if vbar: vbar.lower(widget)
            if hbar: hbar.lower(widget)
        hide()
        widget.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Leave>", lambda e: hide())
        if hbar: hbar.bind("<Enter>", lambda e: show())
        if hbar: hbar.bind("<Leave>", lambda e: hide())
        widget.bind("<Leave>", lambda e: hide())
    
    def v_scrollbar(self,vbar, widget, x, y, w, h, pw, ph):
        widget.configure(yscrollcommand=vbar.set)
        vbar.config(command=widget.yview)
        vbar.place(relx=(w + x) / pw, rely=y / ph, relheight=h / ph, anchor='ne')
    def h_scrollbar(self,hbar, widget, x, y, w, h, pw, ph):
        widget.configure(xscrollcommand=hbar.set)
        hbar.config(command=widget.xview)
        hbar.place(relx=x / pw, rely=(y + h) / ph, relwidth=w / pw, anchor='sw')
    def create_bar(self,master, widget,is_vbar,is_hbar, x, y, w, h, pw, ph):
        vbar, hbar = None, None
        if is_vbar:
            vbar = Scrollbar(master)
            self.v_scrollbar(vbar, widget, x, y, w, h, pw, ph)
        if is_hbar:
            hbar = Scrollbar(master, orient="horizontal")
            self.h_scrollbar(hbar, widget, x, y, w, h, pw, ph)
        self.scrollbar_autohide(vbar, hbar, widget)
    def __tk_frame_layout_pre(self,parent):
        frame = Frame(parent,)
        frame.place(x=36, y=51, width=455, height=153)
        return frame
    def __tk_input_input_folder_path(self,parent):
        ipt = Entry(parent, )
        ipt.place(x=106, y=2, width=338, height=30)
        return ipt
    def __tk_button_select_folder_path(self,parent):
        btn = Button(parent, text="Select Folder", takefocus=False,)
        btn.place(x=13, y=2, width=88, height=30)
        return btn
    def __tk_label_Wiener_(self,parent):
        label = Label(parent,text="Wiener",anchor="center", )
        label.place(x=9, y=72, width=81, height=30)
        return label
    def __tk_input_Wiener(self,parent):
        ipt = Entry(parent, )
        ipt.place(x=106, y=72, width=73, height=30)
        return ipt
    
    def __tk_label_Channel_(self,parent):
        label = Label(parent,text="Channel",anchor="center", )
        label.place(x=9, y=94, width=81, height=30)
        return label
    def __tk_input_Channel(self,parent):
        ipt = Entry(parent, )
        ipt.place(x=106, y=94, width=73, height=30)
        return ipt
    
    def __tk_button_Run(self,parent):
        btn = Button(parent, text="Run", takefocus=False,)
        btn.place(x=238, y=64, width=205, height=60)
        return btn
    def __tk_label_Data_reconstruction(self,parent):
        label = Label(parent,text="Reconstruction",anchor="center", )
        label.place(x=58, y=11, width=388, height=30)
        return label
    def __tk_label_condition(self,parent):
        label = Label(parent,text="condition",anchor="center", )
        label.place(x=33, y=223, width=456, height=32)
        return label
    
class Win(WinGUI):
    def __init__(self, controller):
        self.ctl = controller
        super().__init__()
        self.__event_bind()
        self.__style_config()
        self.ctl.init(self)
    def __event_bind(self):
        self.tk_button_select_folder_path.bind('<Button-1>',self.ctl.select_folder_path)
        self.tk_button_Run.bind('<Button-1>',self.ctl.Run_All)
        pass
    def __style_config(self):
        pass
if __name__ == "__main__":
    win = WinGUI()
    win.mainloop()