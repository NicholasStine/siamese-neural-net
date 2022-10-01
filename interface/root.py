from tkinter import *
from interface.grid import drawGrid

# Init the GUI root and utils
def start():
    root = Tk()
    padded_frame = Frame(root)
    padded_frame.grid(padx=20, pady=20)
    drawGrid(padded_frame)
    root.mainloop()