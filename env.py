from car import Car
import nn
import random


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
        init_y = [0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5]
        init_v = [-2.5, -2.5, -2.5, -2, -2, -2, -1.5, -1.5, -1.5]
        for i in range(9):
            self.Cars.append(Car(name_list[i], init_x[i] + 5, init_y[i], init_v[i]))

    def step(self):
        for car in self.Cars:
            state = self.perception(car)
            action = nn.choose_action(state)
            car.update_v(action)
        for car in self.Cars:
            car.play()

    def perception(self,car):
        state = []
        return state
