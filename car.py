class Car(object):
    def __init__(self,name,x,y,vx):
        self.name = name
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = 0
        self.del_t = 0.01
        self.lastx = x
        self.lasty = y
        self.lastaction = 0
        self.state = []
        self.state_ = []
        self.ep_r = 0

    def reset(self,x,y,vx):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = 0
        self.lastx = x
        self.lasty = y
        self.lastaction = 0
        self.state = []
        self.state_ = []
        self.ep_r = 0

        return x,y,vx,self.vy

    def update_v(self,a):
        if(a==0 and self.vx > -4):
            self.vx -= 0.1
            self.vy = 0
        if(a==0 and self.vy !=0):
            self.vy = 0
        if(a==1 and self.vx < -2):
            self.vx += 0.1
            self.vy = 0
        if(a==2 and self.vy>-1):
            self.vx = self.vx
            self.vy -= 0.2
        if(a==3 and self.vy<1):
            self.vx =self.vx
            self.vy += 0.2

        self.lastaction = a

    def play(self):
        self.lastx = self.x
        self.lasty = self.y
        self.x = self.x + self.vx * self.del_t
        self.y = self.y + self.vy * self.del_t

        return self.x,self.y,self.vx,self.vy