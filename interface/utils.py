import tkinter as tk

# Setup the tkinter root window
def setupTkWindow():
    pass

# Packable item with tk.widget and optional side
class Packable():
    def __init__(self, widget, side):
        self.widget = widget
        self.side = side

# Store and pack all tkinter 
# elements that require _.pack()
class tkPack():
    def __init__(self):
        self.packable = []

    def append(self, tk_object, side=None):
        # Add the element to packable[]
        self.packable.append(Packable(tk_object, side))
        return tk_object

    # Reduce self.packable to the first n items
    # Target Length = (Total Number of tk.widgets) - (Number of tk.widgets to be updated)
    def popToLength(self, target_length):
        if (len(self.packable) > target_length):
            the_destroyables = self.packable[target_length:]
            for destroyable in the_destroyables:
                destroyable.widget.destroy()
            self.packable = self.packable[:target_length]
        

    # Call _.pack() on all cached
    # tkinter elements in self.packable
    def packThatThangUp(self):
        for tk_element in self.packable:
            side = tk_element.side
            if (side == None):
                tk_element.widget.pack()
            else:
                tk_element.widget.pack(side=side) 
            
    
