from car import Car
import nn
import random
import numpy as np
import math

class env(object):
    def __init__(self):
        self.Cars = []
        #generte 9 cars' position and velocity

        #left 3:
        left_rand_1 = float(random.randint(500))/100
        left_rand_2 = float(random.randint(500))/100
        while abs(left_rand_2-left_rand_1)<1:
            left_rand_2 = float(random.randint(500))/100
        left_rand_3 = float(random.randint(500))/100
        while(abs(left_rand_1-left_rand_3)<1 or (left_rand_2-left_rand_3)<1):
            left_rand_3 = float(random.randint(500)/100)

        #center 3:
        center_rand_1 = float(random.randint(500))/100
        center_rand_2 = float(random.randint(500))/100
        while abs(center_rand_2-center_rand_1)<1:
            center_rand_2 = float(random.randint(500))/100
        center_rand_3 = float(random.randint(500))/100
        while(abs(center_rand_1-center_rand_3)<1 or (center_rand_2-center_rand_3)<1):
            center_rand_3 = float(random.randint(500)/100)

        #right 3:
        right_rand_1 = float(random.randint(500))/100
        right_rand_2 = float(random.randint(500))/100
        while abs(right_rand_2-right_rand_1)<1:
            right_rand_2 = float(random.randint(500))/100
        right_rand_3 = float(random.randint(500))/100
        while(abs(right_rand_1-right_rand_3)<1 or (right_rand_2-right_rand_3)<1):
            right_rand_3 = float(random.randint(500)/100)

        #initialize 9 cars:
        name_list = ["left_1","left_2","left_3","center_1","center_2","center_3","right_1","right_2","right_3"]
        init_x = [left_rand_1,left_rand_2,left_rand_3,center_rand_1,center_rand_2,center_rand_3,right_rand_1,right_rand_2,right_rand_3]
        init_y = [0.5,0.5,0.5,1.5,1.5,1.5,2.5,2.5,2.5]
        init_v = [-2.5,-2.5,-2.5,-2,-2,-2,-1.5,-1.5,-1.5]
        for i in range(9):
            self.Cars.append(Car(name_list[i],init_x[i]+5,init_y[i],init_v[i]))

    def reset(self):
        self.Cars = []
        # generte 9 cars' position and velocity

        # left 3:
        left_rand_1 = float(random.randint(500)) / 100
        left_rand_2 = float(random.randint(500)) / 100
        while abs(left_rand_2 - left_rand_1) < 1:
            left_rand_2 = float(random.randint(500)) / 100
        left_rand_3 = float(random.randint(500)) / 100
        while (abs(left_rand_1 - left_rand_3) < 1 or (left_rand_2 - left_rand_3) < 1):
            left_rand_3 = float(random.randint(500) / 100)

        # center 3:
        center_rand_1 = float(random.randint(500)) / 100
        center_rand_2 = float(random.randint(500)) / 100
        while abs(center_rand_2 - center_rand_1) < 1:
            center_rand_2 = float(random.randint(500)) / 100
        center_rand_3 = float(random.randint(500)) / 100
        while (abs(center_rand_1 - center_rand_3) < 1 or (center_rand_2 - center_rand_3) < 1):
            center_rand_3 = float(random.randint(500) / 100)

        # right 3:
        right_rand_1 = float(random.randint(500)) / 100
        right_rand_2 = float(random.randint(500)) / 100
        while abs(right_rand_2 - right_rand_1) < 1:
            right_rand_2 = float(random.randint(500)) / 100
        right_rand_3 = float(random.randint(500)) / 100
        while (abs(right_rand_1 - right_rand_3) < 1 or (right_rand_2 - right_rand_3) < 1):
            right_rand_3 = float(random.randint(500) / 100)

        # initialize 9 cars:
        name_list = ["left_1", "left_2", "left_3", "center_1", "center_2", "center_3", "right_1", "right_2", "right_3"]
        init_x = [left_rand_1, left_rand_2, left_rand_3, center_rand_1, center_rand_2, center_rand_3, right_rand_1,
                  right_rand_2, right_rand_3]
        init_y = [-4.2, -4.2, -4.2, 0, 0, 0, 4.2, 4.2, 4.2]
        init_v = [-2.5, -2.5, -2.5, -2, -2, -2, -1.5, -1.5, -1.5]
        for i in range(9):
            self.Cars.append(Car(name_list[i], init_x[i] + 5, init_y[i], init_v[i]))

    def step(self):
        for car in self.Cars:
            car.state = self.perception(car)
            action = nn.choose_action(car.state)
            car.update_v(action)
        for car in self.Cars:
            car.play()

        for car in self.Cars:
            car.state_ = self.perception(car)
            reward,done = self.cal_reward(car)
            nn.store_trasition(car.state,car.lastaction,reward,car.state_,done)

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

        #next is vehicle state:
        VehicleState = [0.0 for i in range(4)]
        VehicleState[0] = car.x
        VehicleState[1] = car.y
        VehicleState[2] = car.vx
        VehicleState[3] = car.vy

        state = [OccMapState,VehicleState]

        return state

    def cal_reward(self,car):
        collision = 0
        arrive = 0
        done = 0
        for othercar in self.Cars:
            if othercar != car:
                if abs(othercar.x-car.x)<1.5 and abs(othercar.y-car.y)<2.8:
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
            reward = 0.1

        return reward,done