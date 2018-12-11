from car import Car
import actor_critic
import random
import numpy as np
import math
import matplotlib.pyplot as plt

ac = actor_critic.Actor_Critic()

class Env(object):
    def __init__(self):
        self.Cars = []
        self.Waiting_cars = []
        self.car_nums = 9
        self.Waiting_car_nums = 0
        self.used_car_nums = 9
        self.r_list = []
        #generte 9 cars' position and velocity
        print("initialing")
        #left 3:
        left_rand_1 = float(random.randint(0,1600))/100
        left_rand_2 = float(random.randint(0,1600))/100
        while abs(left_rand_2-left_rand_1)<4:
            left_rand_2 = float(random.randint(0,1600))/100
        left_rand_3 = float(random.randint(0,1600))/100
        while(abs(left_rand_1-left_rand_3)<4 or abs(left_rand_2-left_rand_3)<4):
            left_rand_3 = float(random.randint(0,1600))/100
            #print(left_rand_3)
        #center 3:
        center_rand_1 = float(random.randint(0,1600))/100
        center_rand_2 = float(random.randint(0,1600))/100
        while abs(center_rand_2-center_rand_1)<4:
            center_rand_2 = float(random.randint(0,1600))/100
        center_rand_3 = float(random.randint(0,1600))/100
        while(abs(center_rand_1-center_rand_3)<4 or abs(center_rand_2-center_rand_3)<4):
            center_rand_3 = float(random.randint(0,1600))/100

        #right 3:
        right_rand_1 = float(random.randint(0,1600))/100
        right_rand_2 = float(random.randint(0,1600))/100
        while abs(right_rand_2-right_rand_1)<4:
            right_rand_2 = float(random.randint(0,1600))/100
        right_rand_3 = float(random.randint(0,1600))/100
        while(abs(right_rand_1-right_rand_3)<4 or abs(right_rand_2-right_rand_3)<4):
            right_rand_3 = float(random.randint(0,1600))/100

        #initialize 9 cars:
        name_list = ["left_1","left_2","left_3","center_1","center_2","center_3","right_1","right_2","right_3"]
        init_x = [left_rand_1,left_rand_2,left_rand_3,center_rand_1,center_rand_2,center_rand_3,right_rand_1,right_rand_2,right_rand_3]
        init_y = [-4.2,-4.2,-4.2,0,0,0,4.2,4.2,4.2]
        init_v = [(-2.5+random.random()) for i in range(self.car_nums)]
        for i in range(self.car_nums):
            self.Cars.append(Car(name_list[i],init_x[i],init_y[i],init_v[i]))
        print("init done!")

    def reset(self,car,FROMCARS):
        success_reset = 0
        try_times = 0
        while success_reset == 0:
            try_times += 1
            success_reset = 1
            lane_number = random.randint(0,2)
            if lane_number == 0:
                init_y = -4.2
                init_v = -2.5+random.random()
            elif lane_number == 1:
                init_y = 0
                init_v = -2.5+random.random()
            else:
                init_y = 4.2
                init_v = -2.5+random.random()

            init_x = float(random.randint(0,1600)/100)

            for othercar in self.Cars:
                if othercar != car:
                    if abs(othercar.x - init_x) < 2.8 and abs(othercar.y - init_y) < 1.5:
                        success_reset = 0

            if try_times >= 30:
                if FROMCARS==1:
                    print("start_zone is full!")
                    self.Waiting_cars.append(car)
                    self.Cars.remove(car)
                    self.Waiting_car_nums += 1
                    self.car_nums -= 1

                    return 0
                else:
                    return 0

        car.reset(init_x,init_y,init_v)
        return 1


    def step(self):
        for car in self.Cars:
            car.state = self.perception(car)
            action = ac.choose_action(car.state)
            #print(action)
            car.update_v(action)
        for car in self.Cars:
            car.play()

        for i in range(self.car_nums):
            self.Cars[i].state_ = self.perception(self.Cars[i])
            reward,done = self.cal_reward(self.Cars[i])
            ac.store_transition(self.Cars[i].state[0],self.Cars[i].state[1],self.Cars[i].lastaction,reward,self.Cars[i].state_[0],self.Cars[i].state_[1],done)
            self.Cars[i].ep_r += reward
            self.Cars[i].done = done
        for car in self.Cars:
            if car.done == 1:
                print("ep_r:",round(car.ep_r,2),"epsilon:",ac.EPSILON)
                print("now_cars:",self.car_nums)
                print("waiting_cars:",self.Waiting_car_nums)
                success = self.reset(car,FROMCARS=1)

        if (ac.EPSILON<0.9):
            ac.EPSILON += 0.00002
        if (ac.pointer>ac.MEMORY_CAPACITY):
            ac.learn()

        CAR_IN_START = 0
        for car in self.Cars:
            if (car.x > 0) and (car.x<16):
                CAR_IN_START += 1

        if CAR_IN_START < 6:
            self.add_car()

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
        outline = 0
        arrive = 0
        done = 0
        for othercar in self.Cars:
            if othercar != car:
                if abs(othercar.x-car.x)<2.8 and abs(othercar.y-car.y)<1.5:
                    collision = 1
                    break
        if (car.y > 6) or (car.y < -6):
            outline = 1

        if car.x < -30:
            arrive = 1

        if collision == 1:
            reward = -10
            done = 1
        elif arrive == 1:
            reward = 10
            done = 1
        elif outline == 1:
            reward = -5
            done = 1
        else:
            reward = -0.03 + (-0.01)*car.vx

        return reward,done

    def add_car(self):
        print("adding~")
        if self.Waiting_cars == []:
            success_reset = 0
            while success_reset == 0:
                success_reset = 1
                lane_number = random.randint(0, 2)
                if lane_number == 0:
                    init_y = -4.2
                    init_v = -2.5 + random.random()
                elif lane_number == 1:
                    init_y = 0
                    init_v = -2.5 + random.random()
                else:
                    init_y = 4.2
                    init_v = -2.5 + random.random()
                init_x = float(random.randint(0, 1600) / 100)
                for othercar in self.Cars:
                    if abs(othercar.x - init_x) < 2.8 and abs(othercar.y - init_y) < 1.5:
                        success_reset = 0

            self.car_nums += 1
            self.used_car_nums += 1
            name = "new_car_" + str(self.used_car_nums-9)
            car = Car(name, init_x, init_y, init_v)
            self.Cars.append(car)
        else:
            success = self.reset(self.Waiting_cars[0],FROMCARS=0)
            if success == 1:
                self.Cars.append(self.Waiting_cars[0])
                self.car_nums += 1
                self.Waiting_cars.remove(self.Waiting_cars[0])
                self.Waiting_car_nums -= 1

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
        #print(multi_env.car_nums)
        if ac.EPSILON>0.9:
            lenth = len(multi_env.r_list)
            xx = range(lenth)
            plt.plot(xx,multi_env.r_list)
            plt.show()
            plt.pause(1000)

            """
            #ac.saver()

            plt.cla()
            vehicle_x = [car.x for car in multi_env.Cars]
            vehicle_y = [car.y for car in multi_env.Cars]
            lines = ax.scatter(vehicle_x,vehicle_y,c='b',marker="*",s=200)
            plt.pause(0.0001)
            try:
                ax.lines.remove(lines[0])
            except:
                pass
            plt.show()
            """