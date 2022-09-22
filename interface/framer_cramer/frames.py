



# An incredibly iffy double linked frame node
class FrameNode():
    def __init__(self, image, str_index=0):
        self.image = image
        self.next = None
        self.prev = None
        self.str_index = str_index

    def __str__(self):
        return "{0} => {1} => {2}".format(
            "None" if self.prev == None else self.prev.str_index,
            self.str_index,
            "None" if self.next == None else self.next.str_index
        )

# a DoUbLe lInKeD LiSt of frames
class LinkedFrames():
    def __init__(self):
        self.root = None
        self.viewing = None
        self.str_index = 0

    def append(self, frame):
        if (self.root == None):
            self.root = FrameNode(frame)
            self.viewing = self.root
        else:
            self.str_index += 1
            self.root.next = FrameNode(frame, self.str_index)
            self.root.next.prev = self.root
            self.root = self.root.next

    def nextFrame(self):
        if (self.viewing.next != None): self.viewing = self.viewing.next
        return self.viewing.image
        
    def prevFrame(self):
        if (self.viewing.prev != None): self.viewing = self.viewing.prev
        return self.viewing.image

    def hasNext(self):
        if (self.viewing == None): return False
        return self.viewing.next != None

    def hasPrev(self):
        if (self.viewing == None): return False
        return self.viewing.prev != None
