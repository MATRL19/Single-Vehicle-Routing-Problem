import numpy as np
import random
import os, os.path
import numpy as np
from math import pow
import pandas as pd
import csv
from math import pow ,sqrt
import random
from sklearn.ensemble import RandomForestRegressor

class Tiling:
    def __init__(self, xStart, xEnd, yStart, yEnd, zStart, zEnd, binNumber, expandRate):

        self.binNumber = binNumber
        self.tiles = np.zeros([self.binNumber, self.binNumber, self.binNumber])

        expandRate = expandRate
        ExpandRatio = random.randint(0, expandRate)

        self.startx= xStart - (ExpandRatio*(xEnd-xStart)/100)
        self.endx= xEnd + ((expandRate-ExpandRatio)*(xEnd-xStart)/100)

        self.starty= yStart - (ExpandRatio*(yEnd-yStart)/100)
        self.endy= yEnd + ((expandRate-ExpandRatio)*(yEnd-yStart)/100)

        self.startz = zStart - (ExpandRatio*(zEnd-zStart)/100)
        self.endz = zEnd - ((expandRate-ExpandRatio)*(zEnd-zStart)/100)

        self.stepx = (self.endx- self.startx)/self.binNumber
        self.stepy = (self.endy- self.starty)/self.binNumber
        self.stepz = (self.endz- self.startz)/self.binNumber

    def update(self, tileX, tiley, tilez, value):

        xBound= self.startx
        yBound = self.starty
        zBound = self.startz

        xind = -1
        yind = -1
        zind = -1

        while xBound < tileX:
            xBound += self.stepx
            xind += 1

        while yBound < tiley:
            yBound += self.stepy
            yind += 1

        while zBound < tilez:
            zBound += self.stepz
            zind += 1

        xind = np.clip(xind, 0, self.binNumber)
        yind = np.clip(yind, 0, self.binNumber)
        zind = np.clip(zind, 0, self.binNumber)

        self.tiles[xind, yind, zind] *= 0.5 + 0.5 * value

    def showTiling(self):
        print(self.tiles)
        print('$$$$$$$$$$$$$$$$$')
        print(self.startx, self.endx, self.stepx)
        print(self.starty, self.endy, self.stepy)
        print(self.startz, self.endz, self.stepz)

    def getValue(self, tileX, tiley, tileZ):
        xBound= self.startx
        yBound = self.starty
        zBound = self.startz
        xind = -1
        yind = -1
        zind = -1


        while xBound < tileX:
            xBound += self.stepx
            xind += 1

        while yBound < tiley:
            yBound += self.stepy
            yind += 1

        while zBound < tileZ:
            zBound += self.stepz
            zind +=1

        xind = np.clip(xind, 0, self.binNumber)
        yind = np.clip(yind, 0, self.binNumber)
        zind = np.clip(zind, 0, self.binNumber)

        print('$$$$$$$4')
        print(self.tiles.shape)
        print(yind)

        return self.tiles[xind, yind, zind]

    def position(self, tileX, tiley, tileZ):
        xBound= self.startx
        yBound = self.starty
        zBound = self.startz
        xind = -1
        yind = -1
        zind = -1

        while xBound < tileX:
            xBound += self.stepx
            xind += 1

        while yBound < tiley:
            yBound += self.stepy
            yind += 1

        while zBound < tileZ:
            zBound += self.stepz
            zind += 1

        xind = np.clip(xind, 0, self.binNumber)
        yind = np.clip(yind, 0, self.binNumber)
        zind = np.clip(zind, 0, self.binNumber-1)

        tmp = np.zeros([self.binNumber*self.binNumber*self.binNumber])
        tmp[zind*self.binNumber*self.binNumber+xind*self.binNumber+yind] = 1
        tmp2 = zind*self.binNumber*self.binNumber+xind*self.binNumber+yind
        return tmp, tmp2

    def reset(self, binNumber):
        self.binNumber = binNumber
        self.tiles = np.zeros([self.binNumber, self.binNumber, self.binNumber])

class Tile:
    def __init__(self, xStartBound, xEndBound, yStartBound, yEndBound, zStartBound, zEndBound, numOfTilings, binNumber, expandRate):

        self.xStart = xStartBound
        self.xEnd = xEndBound

        self.yStart = yStartBound
        self.yEnd = yEndBound

        self.zStart = zStartBound
        self.zEnd = zEndBound

        self.binNumber = binNumber
        self.expandRate = expandRate

        self.numOfTilings = numOfTilings

        self.tilings = []

    def generate(self):
        for i in range(1, self.numOfTilings+1):
            tmp= Tiling(self.xStart, self.xEnd, self.yStart, self.yEnd, self.zStart, self.zEnd, self.binNumber, self.expandRate)
            self.tilings.append(tmp)

    def update(self, stateX, stateY, stateZ, stateValue):
        for obj in self.tilings:
            obj.update(stateX, stateY, stateZ, stateValue)

    def showTilings(self):
        for obj in self.tilings:
            print(obj.tiles)

    def getValue(self, stateX, stateY, stateZ):
        tmp = []
        for obj in self.tilings:
            tmp.append(obj.getValue(stateX, stateY, stateZ))
        return tmp

    def position(self, stateX, stateY, stateZ):
        tmp =[]
        tmp3 = []
        for obj in self.tilings:
            tmp2 = obj.position(stateX, stateY, stateZ)
            tmp.append(tmp2[0])
            tmp3.append(tmp2[1])
        return tmp, tmp3

    def reset(self):
        for obj in self.tilings:
            obj.reset()


a = Tile(-100, 100, -100, 100, -100, 100, 4, 10, 25)

a.generate()

tmp, tmp3 = a.position(0, 25, 49)
print(np.shape(tmp))


class DataPreProcessing:
    def __init__(self, datafile):
        self.datafile = datafile

    def _read_text_file(self):
        flist = open(self.datafile).readlines()[:]
        listData = [s.rstrip('\n') for s in flist]
        return listData

    def _numberOfCities(self):
        listdata = self._read_text_file()
        numberOfCities = len(listdata)
        return numberOfCities

    def _transform_to_list_ndarray(self):
        listData = self._read_text_file()
        for i in range(len(listData)):
            listData[i] = np.fromstring(listData[i], dtype=float, sep=' ')

        return listData

    def extract_coordinates(self):
        Coordiantes = []
        listData = self._transform_to_list_ndarray()
        for i in range(len(listData)):
            Coordiantes.append(np.array(listData[i][1:3]))
        return Coordiantes

    def extract_single_x_coordinate(self, customerID):
        coordinate = self.extract_coordinates_single_customer(customerID)
        X_coordiante = coordinate[0]
        return X_coordiante

    def extract_single_y_coordinate(self, customerID):
        coordinate=self.extract_coordinates_single_customer(customerID)
        Y_coordiante = coordinate[1]
        return Y_coordiante
    
    def extract_all_x_coordinate(self):
        X_coordiantes = []
        coordinates = self.extract_coordinates()

        for i in range(len(coordinates)):
            X_coordiantes.append(np.array(coordinates[i][0]))
        X_coordinateslist = []

        for i in range(len(X_coordiantes)):
            X_coordinateslist.append(float(X_coordiantes[i]))    
        return X_coordinateslist               

    def extract_all_y_coordinate(self):
        Y_coordiantes = []
        coordinates = self.extract_coordinates()

        for i in range(len(coordinates)):
            Y_coordiantes.append(np.array(coordinates[i][1]))
        Y_coordinateslist = []

        for i in range (len(Y_coordiantes)):    
            Y_coordinateslist.append(float(Y_coordiantes[i]))    
        return Y_coordinateslist

    def extract_coordinates_single_customer(self, customer_ID):
        coordinates = self.extract_coordinates()
        return coordinates[customer_ID-1]

    def compute_distance_between_2_customers(self,customer_ID1, customer_ID2):
        coordinate_customer_ID1 =  self.extract_coordinates_single_customer(customer_ID1)
        coordinate_customer_ID2 = self.extract_coordinates_single_customer(customer_ID2)
        distance = sqrt((pow((coordinate_customer_ID1[0]-coordinate_customer_ID2[0]), 2))+(pow((coordinate_customer_ID1[1]-coordinate_customer_ID2[1]), 2)))
        return distance

    def extract_ID_customer_from_coordinates(self, coordinates):
        all_coordinates=self.extract_coordinates()
        for i in range(len(all_coordinates)):
            if coordinates[0] == all_coordinates[i][0] and coordinates[1] == all_coordinates[i][1]:
                return i+1
                break
        return None
    
    def compute_distance_Matrix(self):
        D = np.zeros(shape=(self._numberOfCities(), self._numberOfCities()))

        for i in range(len(D)):
            for j in range(len(D)):
                if i == 0 and j == 0:
                    D[i][j] = self.compute_distance_between_2_customers(self._numberOfCities(), self._numberOfCities())
                if i == 0 and j != 0:
                    D[i][j] = self.compute_distance_between_2_customers(self._numberOfCities(), j)
                if j == 0 and i != 0:
                    D[i][j] = self.compute_distance_between_2_customers(i, self._numberOfCities())
                else:    
                    D[i][j] = self.compute_distance_between_2_customers(i, j)
        return np.negative(D)

    def print_circuit_broad(self,X_coordinates,Y_coordinates):
        Customers_X_coordinates=X_coordinates[0:len(X_coordinates)-1]
        Customers_Y_coordinates=Y_coordinates[0:len(Y_coordinates)-1]
        Depot_X_coordinate=X_coordinates[len(X_coordinates)-1]
        Depot_Y_coordinate=Y_coordinates[len(Y_coordinates)-1]
        
        
cwd = os.getcwd()
path = cwd + "/Test/j2_1_4.TXT"
dataFile = DataPreProcessing(path)
Matrix_Distance = dataFile.compute_distance_Matrix()


def Boltzman_policy(Q, state, tempertaure_param, R_matrix):
    Not_Possible_Actions = np.argwhere(np.isnan(R_matrix[state, :]))
    p = np.array([Q[(state, x)]/tempertaure_param for x in range(len(Q))])
    pi_actions = np.exp(p) / np.sum(np.exp(p))

    for x in np.nditer(Not_Possible_Actions):
        pi_actions[x] = 0

    IndexList = np.where(pi_actions == np.amax(pi_actions))
    pi_actions[np.random.choice(IndexList[0])] += 1 - sum(pi_actions)
    action = np.random.choice(range(len(Q)), p=pi_actions)
    return action


def epsilonGreedy(epsilon, s, Q, A, w, fn_approx, x_coordinates, y_coordinates, Matrix_Distance):   
    if fn_approx == "ANN":
        training_data = pd.read_csv('training.csv',index_col=False, header=None)
        training_data = np.array(training_data)
        model = RandomForestRegressor(n_estimators=400, oob_score=True, random_state=43)
        model.fit(training_data[:, 1:], training_data[:, 0])
    TileCoding = Tile(-100, 100, -100, 100, -100, 100, numOfTilings, binNumber, 25)
    TileCoding.generate()
    p = np.random.random_sample()

    if p > epsilon:
        max_Q = min(Q[s, A[:, 0]])
        greedy_actions = [i for i, j in enumerate(Q[s, :]) if j == max_Q and i in A[:, 0].tolist()]
        a = random.choice(greedy_actions)
        MAX = -9999

        for i in A[:, 0].tolist():
            if i == len(Matrix_Distance):
                i = 0
            x, tmp3 = TileCoding.position(x_coordinates[i], y_coordinates[i], len(A[:, 0].tolist())-1)
            new_data = np.zeros((1, 1+2*numOfTilings*binNumber*binNumber*binNumber))
            for r in range(1, 1+numOfTilings*binNumber*binNumber*binNumber):
                new_data[0][r] = np.array(x).reshape(1, numOfTilings*binNumber*binNumber*binNumber)[0][r-1]
            for r in range(1+numOfTilings*binNumber*binNumber*binNumber, 1+2*numOfTilings*binNumber*binNumber*binNumber):
                new_data[0][r] = np.array(w).reshape(1, numOfTilings*binNumber*binNumber*binNumber)[0][r-1-numOfTilings*binNumber*binNumber*binNumber]
            if fn_approx == "linear":
                if Matrix_Distance[s][i] + np.matmul(np.array(w).reshape(1, numOfTilings*binNumber*binNumber*binNumber), np.array(x).reshape(1, numOfTilings*binNumber*binNumber*binNumber).T) > MAX:
                    MAX = Matrix_Distance[s][i] + np.matmul(np.array(w).reshape(1, numOfTilings*binNumber*binNumber*binNumber), np.array(x).reshape(1, numOfTilings*binNumber*binNumber*binNumber).T)
                    a = i
            if fn_approx == "ANN":
                if Matrix_Distance[s][i] + model.predict(new_data[0, 1:].reshape(1, -1)) > MAX:
                    MAX = Matrix_Distance[s][i] + model.predict(new_data[0, 1:].reshape(1, -1))
                    a = i
            if a == 0:
                a = len(Matrix_Distance)
    else:
        greedy_actions = []
        probabilities = []
        total = 0

        for i in A[:, 0].tolist():
            if i == len(Matrix_Distance):
                i = 0
            total += np.exp(Matrix_Distance[s][i])
                 
        for i in A[:, 0].tolist():
            if i == len(Matrix_Distance):
                i = 0
            probabilities.append(np.exp(Matrix_Distance[s][i]) / total)
            if i == 0:
                greedy_actions.append(len(Matrix_Distance))
            else:
                greedy_actions.append(i)
        a = np.random.choice(greedy_actions, 1, probabilities)[0]
    return a


class Agent:
    def __init__(self, temperature, method, start_gamma, alpha, path, w):
        self.method = method
        self.dataFile = DataPreProcessing(path)
        self.n_cities = self.dataFile._numberOfCities()-1
        self.n_actions = self.dataFile._numberOfCities()
        self.gamma = start_gamma
        self.alpha = alpha
        self.temperature = temperature
        self.coordinate_depot = self.dataFile.extract_coordinates_single_customer(self.n_cities+1)
        self.counter = 0
        self.q = np.zeros(shape=(self.n_cities+2, self.n_actions+1))
        self.visited_cities = np.zeros(shape =self.n_cities+2)
        self.track_transitions = []
        self.totalTravelCost = []
        self.epsilon = 0.1
        self.eli = np.zeros([numOfTilings, binNumber*binNumber*binNumber])
        self.w = w

    def init_R_Matrix(self): 
        R = self.dataFile.compute_distance_Matrix()
        R = np.row_stack((R, np.full((self.n_cities+1), np.nan)))
        R = np.column_stack((R, np.full((self.n_cities+2), np.nan)))
        np.fill_diagonal(R, np.nan)
        R[self.n_cities+1][0] = 0
        return R

    def dont_go_back(self, R_matrix, state):
        R_matrix[0:-1, state] = np.nan
        return R_matrix
    
    def possible_actions(self, R_matrix, state):
        Actions = np.argwhere(~np.isnan(R_matrix[state,:]))
        return Actions
    
    def track_transition_sequence(self, action):
        self.track_transitions.append([action])
        return self.track_transitions
    
    def go_next_state(self, action):
        s_prime=action
        return s_prime
    
    def update_visted_states(self, state):
        self.visited_cities[state] = 1
        return self.visited_cities
    
    def update_RL_method(self, state, action, next_state, R_matrix):
        TileCoding = Tile(-100, 100, -100, 100, -100, 100, numOfTilings, binNumber, 25)
        TileCoding.generate()
        
        if self.method == 'q_learning':
            if sum(self.visited_cities) < self.n_cities:
                A_next = self.possible_actions(R_matrix, next_state)
                self.q[state, action] = self.q[state, action] + self.alpha * (R_matrix[state, action] + self.gamma * max(self.q[next_state, A_next])[0] - self.q[state, action])
                return self.q
            else:
                return self.q
           
        elif self.method == 'sarsa':
            if sum(self.visited_cities) < self.n_cities:
                A_next = self.possible_actions(R_matrix, next_state)
                self.q[state, action] = self.q[state, action] + self.alpha * (R_matrix[state, action] + self.gamma * self.q[next_state, A_next][0] - self.q[state, action])
                
                return self.q
            else:
                return self.q
          
        else:
            raise Exception("Invalid method provided")

    def totalCost(self, R_matrix, state, action):
        self.totalTravelCost.append(R_matrix[state,action])
        return self.totalTravelCost

    def terminationTheTrajectory(self):

        if sum(self.visited_cities) == self.n_cities+2:
            return True
        return False

    def iteration(self, episodeNumber):
        TileCoding = Tile(-100, 100, -100, 100, -100, 100, numOfTilings, binNumber, 25)
        TileCoding.generate()
        w = self.w
        episodeNumber = 1
        R_backup= self.dataFile.compute_distance_Matrix()
        graph = []
        lamda = 0.85

        for j in range(episodeNumber):
            self.totalTravelCost = []
            self.visited_cities= np.zeros(shape=self.n_cities+2)
            self.track_transitions = []
            start_state = 0
            state = start_state
            self.update_visted_states(state)
            self.track_transition_sequence(state)
            self.eli *= 0
            R_init = self.init_R_Matrix()
            x_coordinates = self.dataFile.extract_all_x_coordinate()
            y_coordinates = self.dataFile.extract_all_y_coordinate()
            Matrix_Distance = self.dataFile.compute_distance_Matrix()
            conter=0
            record_data = np.zeros((len(Matrix_Distance), 1+2*numOfTilings*binNumber*binNumber*binNumber))
            new_data = np.zeros((len(Matrix_Distance), 1+2*numOfTilings*binNumber*binNumber*binNumber))
            
            if j >= 0:
                training_data = pd.read_csv('training.csv',index_col=False, header=None)
                training_data = np.array(training_data)
                for l in range(len(training_data)):
                    training_data[l, 1+numOfTilings*binNumber*binNumber*binNumber:1+2*numOfTilings*binNumber*binNumber*binNumber] = np.array(w).reshape(1, numOfTilings*binNumber*binNumber*binNumber)
                model = RandomForestRegressor(n_estimators=400, oob_score=True, random_state=43)
                model.fit(training_data[:,1:], training_data[:, 0])

            while ~self.terminationTheTrajectory():
                if j < 0:
                    fn_approx = "linear"
                else:
                    fn_approx = "ANN"
                R_updated=self.dont_go_back(R_init, state)
                possible_actions = self.possible_actions(R_updated, state)
                self.eli *= self.gamma*lamda
                s, tmp3 = TileCoding.position(x_coordinates[state], y_coordinates[state], len(possible_actions[:, 0].tolist()))

                for n in range(len(tmp3)):
                    self.eli[n][tmp3[n]] += 1
                    
                if self.method == 'q_learning':
                    action_epsilonGreeedy = epsilonGreedy(0.1, state, self.q, possible_actions, w, fn_approx, x_coordinates, y_coordinates, Matrix_Distance)
                else:
                    action_epsilonGreeedy = epsilonGreedy(self.epsilon, state, self.q, possible_actions, w, fn_approx, x_coordinates, y_coordinates, Matrix_Distance)
                action = action_epsilonGreeedy
                st = state
                act = action
                if state == len(Matrix_Distance):
                    st = 0
                if action == len(Matrix_Distance):
                    act = 0
                s, tmp3 = TileCoding.position(x_coordinates[st], y_coordinates[st], len(possible_actions[:, 0].tolist()))
                a1, tmp3 = TileCoding.position(x_coordinates[act], y_coordinates[act], len(possible_actions[:, 0].tolist())-1)
                for r in range(1, 1+numOfTilings*binNumber*binNumber*binNumber):
                    record_data[conter][r] = np.array(s).reshape(1, numOfTilings*binNumber*binNumber*binNumber)[0][r-1]
                for r in range(1+numOfTilings*binNumber*binNumber*binNumber, 1+2*numOfTilings*binNumber*binNumber*binNumber):
                    record_data[conter][r] = np.array(w).reshape(1, numOfTilings*binNumber*binNumber*binNumber)[0][r-1-numOfTilings*binNumber*binNumber*binNumber]
                for r in range(1, 1+numOfTilings*binNumber*binNumber*binNumber):
                    new_data[conter][r] = np.array(a1).reshape(1, numOfTilings*binNumber*binNumber*binNumber)[0][r-1]
                for r in range(1+numOfTilings*binNumber*binNumber*binNumber, 1+2*numOfTilings*binNumber*binNumber*binNumber):
                    new_data[conter][r] = np.array(w).reshape(1, numOfTilings*binNumber*binNumber*binNumber)[0][r-1-numOfTilings*binNumber*binNumber*binNumber]
                
                if j < 0:
                    delta = (Matrix_Distance[st][act] + np.matmul(self.gamma*np.array(w).reshape(1, numOfTilings*binNumber*binNumber*binNumber), np.array(a1).reshape(1,numOfTilings*binNumber*binNumber*binNumber).T)\
                             - np.matmul(np.array(w).reshape(1, numOfTilings*binNumber*binNumber*binNumber), np.array(s).reshape(1, numOfTilings*binNumber*binNumber*binNumber).T))
                else:
                    delta = Matrix_Distance[st][act] + self.gamma*model.predict(new_data[conter, 1:].reshape(1, -1)) - model.predict(record_data[conter, 1:].reshape(1, -1))
                w += self.alpha*delta*self.eli
                nextState = action_epsilonGreeedy
                self.totalCost(R_updated, state, action_epsilonGreeedy)
                self.update_RL_method(state, action_epsilonGreeedy, nextState, R_updated)
                R_init = R_updated
                state = nextState
                self.track_transition_sequence(state) 
                self.update_visted_states(state)
                conter += 1

                if sum(self.visited_cities) == self.n_cities+1:
                    for k in range(self.n_cities+1):
                        R_init[ k,self.n_cities+1] = R_backup[k, 0]

            print("Total Cost of this trajectory is:", sum(self.totalTravelCost))
            graph.append(-sum(self.totalTravelCost))
            old_cost = sum(self.totalTravelCost)
            cost_check = [old_cost]

            for i in range(self.n_cities+1):
                b = self.track_transitions[i+1][0]
                a = self.track_transitions[i][0]
                if b == len(Matrix_Distance):
                    b = 0         
                if i + 1 == len(Matrix_Distance):
                    break
                old_cost = old_cost - Matrix_Distance[a][b]
                cost_check.append(old_cost)

            for i in range(len(Matrix_Distance)):
                record_data[i][0] = cost_check[i]

            print(self.track_transitions)
            cwd = os.getcwd()
            pd.DataFrame(record_data).to_csv(cwd+"training.csv", mode='a', header=None, index=None)
            pd.DataFrame(w).to_csv("weight.csv", mode='w', header=None, index=None)
            print(self.epsilon)
            if j % 5 == 0:
                self.epsilon = self.epsilon*(random.random())
                self.temperature *= 0.95

        #######TEST######
        print(graph)
        print("TEST**************TEST*************TEST")
        cwd = os.getcwd()
        pathTest, dirs, filesTest = next(os.walk(cwd+"/Test/"))

        for k in range(len(filesTest)):
            fullPath = pathTest+filesTest[k]
            path = fullPath
            print(' Test file path', path)
            self.dataFile = DataPreProcessing(path)
            x_coordinates = self.dataFile.extract_all_x_coordinate()
            y_coordinates = self.dataFile.extract_all_y_coordinate()
            Matrix_Distance = self.dataFile.compute_distance_Matrix()
            self.n_cities = self.dataFile._numberOfCities()-1
            self.totalTravelCost = []
            self.visited_cities= np.zeros(shape=self.n_cities+2)
            self.track_transitions = []
            start_state = 0
            state = start_state
            R_init = self.init_R_Matrix()
            self.update_visted_states(state)
            self.track_transition_sequence(state)
            conter = 0

            while ~self.terminationTheTrajectory():
                print(conter)
                conter += 1
                R_updated = self.dont_go_back(R_init, state)
                possible_actions = self.possible_actions(R_updated, state)
                if self.method == 'q_learning':
                    action_epsilonGreeedy = epsilonGreedy(0.1, state, self.q, possible_actions, w, fn_approx, x_coordinates, y_coordinates, Matrix_Distance)
                else:
                    action_epsilonGreeedy = epsilonGreedy(self.epsilon, state, self.q, possible_actions, w, fn_approx, x_coordinates, y_coordinates, Matrix_Distance)
                nextState = action_epsilonGreeedy
                self.totalCost(R_updated, state, action_epsilonGreeedy)
                self.update_RL_method(state, action_epsilonGreeedy, nextState, R_updated)
                R_init = R_updated
                state = nextState
                self.update_visted_states(state)
                self.track_transition_sequence(state)
                if sum(self.visited_cities) == self.n_cities+1:
                    for k in range(self.n_cities+1):
                        R_init[k, self.n_cities+1] = R_backup[k, 0]
            print("transition_states", self.track_transitions)
            print("Total Cost of this trajectory is:", sum(self.totalTravelCost))
            print("visited_states", self.visited_cities)
            print("%%%%%%%%%%%%%%END OF ONE TEST FILE %%%%%%%%%%%%%%%%%%%%%%%%%%")


numOfTilings = 4
binNumber = 10
w = pd.read_csv('weight.csv',index_col=False, header=None)
w = np.array(w)
tracking = np.zeros((len(Matrix_Distance), len(Matrix_Distance)))
cwd = os.getcwd()
pathTr, dirs, filesTr = next(os.walk(cwd+"/Train/"))
epoques = 10
for e in range(epoques):
    for k in range(len(filesTr)):
        fullPath = pathTr+filesTr[k]
        path = fullPath
        print('file path', path)
        w = pd.read_csv('weight.csv', index_col=False, header=None)
        w = np.array(w)
        Q_LearningAgent = Agent(temperature=10, method='sarsa', start_gamma=1, alpha=0.025, path=path, w=w)
        Q_LearningAgent.iteration(10)
