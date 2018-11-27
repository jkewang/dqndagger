from car import Car
import nn
import random
import numpy as np
import math
import matplotlib.pyplot as plt

class Env(object):
    def __init__(self):
        self.Cars = []
        #generte 9 cars' position and velocity
        print("initialing")
        #left 3:
        left_rand_1 = float(random.randint(0,500))/100
        left_rand_2 = float(random.randint(0,500))/100
        while abs(left_rand_2-left_rand_1)<1:
            left_rand_2 = float(random.randint(0,500))/100
        left_rand_3 = float(random.randint(0,500))/100
        while(abs(left_rand_1-left_rand_3)<1 or abs(left_rand_2-left_rand_3)<1):
            left_rand_3 = float(random.randint(0,500))/100
            #print(left_rand_3)
        #center 3:
        center_rand_1 = float(random.randint(0,500))/100
        center_rand_2 = float(random.randint(0,500))/100
        while abs(center_rand_2-center_rand_1)<1:
            center_rand_2 = float(random.randint(0,500))/100
        center_rand_3 = float(random.randint(0,500))/100
        while(abs(center_rand_1-center_rand_3)<1 or abs(center_rand_2-center_rand_3)<1):
            center_rand_3 = float(random.randint(0,500))/100

        #right 3:
        right_rand_1 = float(random.randint(0,500))/100
        right_rand_2 = float(random.randint(0,500))/100
        while abs(right_rand_2-right_rand_1)<1:
            right_rand_2 = float(random.randint(0,500))/100
        right_rand_3 = float(random.randint(0,500))/100
        while(abs(right_rand_1-right_rand_3)<1 or abs(right_rand_2-right_rand_3)<1):
            right_rand_3 = float(random.randint(0,500))/100

        #initialize 9 cars:
        name_list = ["left_1","left_2","left_3","center_1","center_2","center_3","right_1","right_2","right_3"]
        init_x = [left_rand_1,left_rand_2,left_rand_3,center_rand_1,center_rand_2,center_rand_3,right_rand_1,right_rand_2,right_rand_3]
        init_y = [-4.2,-4.2,-4.2,0,0,0,4.2,4.2,4.2]
        init_v = [-2.5,-2.5,-2.5,-2,-2,-2,-1.5,-1.5,-1.5]
        for i in range(9):
            self.Cars.append(Car(name_list[i],init_x[i]+5,init_y[i],init_v[i]))
        print("init done!")

    def reset(self,car):
        success_reset = 0
        while success_reset == 0:
            success_reset = 1
            lane_number = random.randint(0,2)
            if lane_number == 0:
                init_y = -4.2
                init_v = -2.5
            elif lane_number == 1:
                init_y = 0
                init_v = -2
            else:
                init_y = 4.2
                init_v = -1.5

            init_x = float(random.randint(0,500)/100)+5

            for othercar in self.Cars:
                if othercar != car:
                    if abs(othercar.x - init_x) < 1 and abs(othercar.y - init_y) < 1.5:
                        success_reset = 0

        car = car.reset(init_x,init_y,init_v)

    def step(self):
        for car in self.Cars:
            car.state = self.perception(car)
            action = nn.choose_action(car.state[0],car.state[1])
            car.update_v(action)
        for car in self.Cars:
            car.play()

        for i in range(9):
            self.Cars[i].state_ = self.perception(self.Cars[i])
            reward,done = self.cal_reward(self.Cars[i])
            nn.store_transition(self.Cars[i].state[0],self.Cars[i].state[1],self.Cars[i].lastaction,reward,self.Cars[i].state_[0],self.Cars[i].state_[1],done)
            self.Cars[i].ep_r += reward
            self.Cars[i].done = done
        for i in range(9):
            if self.Cars[i].done == 1:
                print("ep_r:",round(self.Cars[i].ep_r,2),"epsilon:",nn.EPSILON)
                self.reset(self.Cars[i])

        if (nn.EPSILON<0.9):
            nn.EPSILON += 0.000002
        if (nn.MEMORY_COUNTER>nn.MEMORY_CAPACITY):
            nn.learn()

    def perception(self,car):

        #firstly calculate occupancy grids:
        LOW_X_BOUND = -4.5
        HIGH_X_BOUND = 4.5
        LOW_Y_BOUND = -5
        HIGH_Y_BOUND = 15
        OccMapState = np.zeros((20,7))
        for othercar in self.Cars:
            if othercar != car:
                relX = othercar.x - car.x
                relY = othercar.y - car.y
                if (relX>LOW_X_BOUND and relX<HIGH_X_BOUND) and (relY>LOW_Y_BOUND and relY<HIGH_Y_BOUND):
                    indexX = int((6+relX)/1.5-0.5)
                    indexY = int((15-relY)-0.5)
                    OccMapState[indexY,indexX] = 1.0

        OccMapState = OccMapState.reshape(-1)

        #next is vehicle state:
        VehicleState = np.zeros(40)
        for i in range(40):
            if i%4==0:
                VehicleState[i] = car.x
            elif i%4==1:
                VehicleState[i] = car.y
            elif i%4==2:
                VehicleState[i] = car.vx
            else:
                VehicleState[i] = car.vy

        state = [OccMapState,VehicleState]

        return state

    def cal_reward(self,car):
        collision = 0
        arrive = 0
        done = 0
        for othercar in self.Cars:
            if othercar != car:
                if abs(othercar.x-car.x)<1 and abs(othercar.y-car.y)<1.5:
                    collision = 1
                    break

        if car.x < -10:
            arrive = 1

        if collision == 1:
            reward = -10
            done = 1
        elif arrive == 1:
            reward = 10
            done = 1
        else:
            reward = -0.01

        return reward,done


if __name__ == '__main__':
    multi_env = Env()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.axis("equal")
    plt.ion()
    turns = 0
    while(True):
        turns += 1
        multi_env.step()
        if nn.EPSILON>0.85:
            plt.cla()
            vehicle_x = [multi_env.Cars[i].x for i in range(9)]
            vehicle_y = [multi_env.Cars[i].y for i in range(9)]
            lines = ax.scatter(vehicle_x,vehicle_y,c='b',marker="*",s=200)
            plt.pause(0.0001)
            try:
                ax.lines.remove(lines[0])
            except:
                pass
            plt.show()
