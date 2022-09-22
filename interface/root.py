import tkinter as tk
from interface.grid import drawGrid

# Init the GUI root and utils
def start():
    root = tk.Tk()
    drawGrid(root)
    root.mainloop()