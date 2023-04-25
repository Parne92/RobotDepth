import tkinter as tk
import _thread
import time
#from client import ClientSocket

globalVar=""

class DrawingStuff():
    def __init__(self, r, c=None):
        super(DrawingStuff, self).__init__()
        self.client = c
        self.root = r
        self.flag = True
        self.speed = .8
        r.title("TangoBot")
        self.canvasW = 800
        self.canvasH = 410
        self.c = tk.Canvas(self.root, width = self.canvasW, height=self.canvasH)
        self.c.pack()

    def arrow(self, key):
        print(key.keycode)
        if key.keycode == 38:
            self.speed += .2
        elif key.keycode== 40:
            self.speed -= .2
        print(self.speed)
        
    def changeFlag(self):
        for i in range(100):
            print (i)
        while(i != "Stop"):
            #i = input("Stop, up or down Eyeballs?")
            if i == "up":
                print('up')
                self.speed += .2
                print(self.speed)
            elif i == "down":
                self.speed -= .2
                
        self.flag = False
        
    def drawEyes(self):
        midRow = int(self.canvasH/2)
        midCol = int(self.canvasW/2)
        while(self.flag):
            self.c.create_oval(5, 5, midCol-5, self.canvasH-40, fill="#000000")
            self.c.create_oval(midCol+5, 5, self.canvasW, self.canvasH-40, fill="#000000")
            leftRow = int(midRow/2)+100
            leftCol = int(midCol/2)
            #start pupils
            self.c.create_oval(leftRow, leftCol, leftRow+100, leftCol+100, fill="#ffffff")
            self.c.create_oval(700, 220, 600, 320, fill="#ffffff")
            self.root.update()
            time.sleep(self.speed)

            self.c.create_oval(5, 5, midCol-5, self.canvasH-40, fill="#000000")
            self.c.create_oval(midCol+5, 5, self.canvasW, self.canvasH-40, fill="#000000")
            self.c.create_oval(leftRow, leftCol, leftRow-100, leftCol+100, fill="#ffffff")
            self.c.create_oval(500, 220, 600, 320, fill="#ffffff")
            self.root.update()
            time.sleep(self.speed)
           
    def drawTextbox(self):
     
        self.label = tk.Entry(self.root)
        self.label.pack()
        #self.label.insert(tk.END, "blank")

    def run(self):
        global globalVar
        print("New Thread")
##        while True:
##            time.sleep(1)
##            if True:
##                self.label.insert(0, globalVar)
##                print ("ggggg", globalVar)
##                globalVar = ""
##            time.sleep(1)
            
    def addButton(self):
        self.button = tk.Button(self.root, text="Send", width=25, fg="#ff0000", command=self.handleSend)
        self.button.pack()
            
    def handleSend(self):
        #self.client.sendData(self.label.get())
        print("Button Send:", self.label.get())

def myLoop(y):
    for i in range(250):
        print (i)
    i = input("Give me anything?")
    y.changeFlag()
    
def __main__():
    IP = '192.168.1.7'
    PORT = 5010
##    client = ClientSocket(IP, PORT)
##    client.start()
    root = tk.Tk()
   

    paint = DrawingStuff(root)
    root.bind('<Up>', paint.arrow)
    root.bind('<Left>', paint.arrow)
 
    try:
        #_thread.start_new_thread(paint.changeFlag,())
        _thread.start_new_thread(paint.drawEyes,())
    except:
       print ("Error: unable to start thread")
    #paint.drawEyes()
    print("Goodbye")
    paint.drawTextbox()
    paint.addButton()
    #paint.start()
    root.mainloop()    
    
__main__()
