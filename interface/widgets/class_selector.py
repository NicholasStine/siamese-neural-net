from interface.auto_suggest.feature_tracker import combineFeatures
from tkinter import *

# A dropdown for viewing / altering the current
# selected feature's class in feature_viewer.py
class ClassSelector():
    def __init__(self, root, relief=SUNKEN):
        self.frame = Frame(root, relief=relief, borderwidth=3)
        
        # Display the editing feature
        self.feature_id = ''
        self.feature_label = Label(self.frame, text='No Feature Selected...')
        self.feature_label.pack()

        # A dropdown for choosing from all available unique labels
        self.class_selection = StringVar(self.frame)
        self.class_selection.set('No Features Detected')
        self.class_selection.trace('w', self.dropdownClick)
        self.class_selector = OptionMenu(self.frame, self.class_selection, *['Chagrin', 'Anton'])
        self.class_selector.pack()

        # Talk to the udder wiidgeets on feature change
        self.updateFeatures = None
        self.getCached = None
    
    def setLabels(self, labels):
        # Pass on empty labels list
        if (len(labels) == 0): return

        # Update the selected and selector
        self.class_selector.destroy()
        self.class_selector = OptionMenu(self.frame, self.class_selection, *labels)
        self.feature_label.configure(text=labels[0])
        self.feature_id = labels[0]
        
        # Re-pack le widget
        self.feature_label.pack()
        self.class_selector.pack()
        
    def featureClick(self, label):
        self.feature_label.configure(text=label)
        self.feature_id = label
        self.class_selection.set(label)

    # I don't actually use any of the callback args here
    def dropdownClick(self, selection, errection, deflection):
        if (self.feature_id == self.class_selection.get()): return
        keeper, remover = combineFeatures(self.class_selection.get(), self.feature_id)
        self.updateFeatures(self.getCached(), keeper, remover)

