import sys
import threading
import locale
import os
import time
os.environ['LANG'] = 'en_US.UTF-8'
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(module_path)

from core.ui import Win as MainWin
from core.control import Controller as MainUIController

def background_task():
    while running:
        time.sleep(1)

def on_closing():
    global running
    running = False  
    time.sleep(1)    
    app.destroy()  
    sys.exit()      
    
app = MainWin(MainUIController())
app.protocol("WM_DELETE_WINDOW", on_closing)

if __name__ == "__main__":
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    app.mainloop()