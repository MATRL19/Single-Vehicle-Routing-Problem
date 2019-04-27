import random
import math
import numpy as np
from matplotlib import pyplot as plt
import Tile
import os, os.path
import numpy as np
from math import pow
import pandas as pd
import csv
from math import pow, sqrt
from time import sleep
from collections import defaultdict
from scipy import sparse
from numpy import array


binNumber= 10 #number of divisions in each direction of tilings
tilingNumber = 4 #number of tilings
expandRate = 25 #the overlap of tilings


class DataPreProcessing:

    def __init__(self, datafile):
        self.datafile = datafile

    def _read_text_file(self):
        flist = open(self.datafile).readlines()[:]
        listData=[s.rstrip('\n') for s in flist]
        return listData

    def _numberOfCities(self):
        listdata=self._read_text_file()
        numberOfCities=len(listdata)
        return  numberOfCities

    def _transform_to_list_ndarray(self):
        listData=self._read_text_file()
        for i in range (len(listData)):
            listData[i]=np.fromstring(listData[i], dtype=float, sep=' ')

        return listData

    def extract_coordinates(self):
        Coordiantes=[]
        listData=self._transform_to_list_ndarray()
        for i in range(len(listData)):
            Coordiantes.append(np.array(listData[i][1:3]))
        return Coordiantes
    
    def extract_single_x_coordinate(self, customerID):
        coordinate=self.extract_coordinates_single_customer(customerID)
        X_coordiante=coordinate[0]
        return X_coordiante

    def extract_single_y_coordinate(self, customerID):
        coordinate=self.extract_coordinates_single_customer(customerID)
        Y_coordiante=coordinate[1]
        return Y_coordiante

    def extract_all_x_coordinate(self):
        X_coordiantes=[]
        coordinates=self.shift_coordinates()
        for i in range(len(coordinates)):
            X_coordiantes.append(np.array(coordinates[i][0]))
        X_coordinateslist=[]
        for i in range (len(X_coordiantes)):
            X_coordinateslist.append(float(X_coordiantes[i]))
        return X_coordinateslist


    def extract_all_x_coordinate_modified(self, ID_depot):
        size=self._numberOfCities()+1
        x_coordiantes=[0]*size
        x_coordiantes[0]=float(self.extract_single_x_coordinate(ID_depot))
        x_coordiantes[1:]=self.extract_all_x_coordinate()
        return x_coordiantes

    def extract_all_y_coordinate(self):
        Y_coordiantes=[]
        coordinates=self.shift_coordinates()
        for i in range(len(coordinates)):
            Y_coordiantes.append(np.array(coordinates[i][1]))
        Y_coordinateslist=[]
        for i in range (len(Y_coordiantes)):
            Y_coordinateslist.append(float(Y_coordiantes[i]))
        return Y_coordinateslist

    def extract_all_y_coordinate_modified(self, ID_depot):
        size=self._numberOfCities()+1
        y_coordiantes=[0]*size
        y_coordiantes[0]=float(self.extract_single_y_coordinate(ID_depot))
        y_coordiantes[1:]=self.extract_all_y_coordinate()
        return y_coordiantes


    def extract_coordinates_single_customer(self,customer_ID):
        coordinates=self.shift_coordinates()
        return coordinates[(customer_ID)-1]

    def compute_distance_between_2_customers(self,customer_ID1,customer_ID2):
        coordinate_customer_ID1= self.extract_coordinates_single_customer(customer_ID1)
        coordinate_customer_ID2= self.extract_coordinates_single_customer(customer_ID2)
        distance=sqrt((pow((coordinate_customer_ID1[0]-coordinate_customer_ID2[0]),2))+(pow((coordinate_customer_ID1[1]-coordinate_customer_ID2[1]),2)))
        return distance


    def extract_ID_customer_from_coordinates(self, coordinates):
        all_coordinates=self.shift_coordinates()
        for i in range(len(all_coordinates)):
            if coordinates[0] == all_coordinates[i][0] and coordinates[1] == all_coordinates[i][1]:
                return i+1
                break
        return None

    def extract_ID_depot(self):
        all_coordinates=self.extract_coordinates()
        ID_depot=len(all_coordinates)
        return ID_depot


    def shift_coordinates(self):
        ID_depot=self.extract_ID_depot()
        coordinates=self.extract_coordinates()
        coordiantes_depot=coordinates[ID_depot-1]
        x_coordinate_depot=coordiantes_depot[0]
        y_coordinate_depot=coordiantes_depot[1]
        for i in range(len(coordinates)):
            coordinates[i][0]=coordinates[i][0]-(x_coordinate_depot)
            coordinates[i][1]=coordinates[i][1]-(y_coordinate_depot)

        return coordinates



    def compute_distance_Matrix(self):
        D = np.zeros(shape=(self._numberOfCities(),self._numberOfCities()))
        for i in range(len(D)):
            for j in range (len(D)):
                if i==0 and j==0:
                    D[i][j] = self.compute_distance_between_2_customers(self._numberOfCities(),self._numberOfCities())
                if i==0 and j!=0:
                    D[i][j]=self.compute_distance_between_2_customers(self._numberOfCities(),j)
                if j==0 and i!=0:
                    D[i][j] = self.compute_distance_between_2_customers(i,self._numberOfCities())
                else:
                    D[i][j] = self.compute_distance_between_2_customers(i,j)
        return np.negative(D)


    def print_circuit_broad(self,X_coordinates,Y_coordinates):
        Customers_X_coordinates=X_coordinates[0:len(X_coordinates)-1]
        Customers_Y_coordinates=Y_coordinates[0:len(Y_coordinates)-1]
        Depot_X_coordinate=X_coordinates[len(X_coordinates)-1]
        Depot_Y_coordinate=Y_coordinates[len(Y_coordinates)-1]
        plt.figure(figsize=(20,10))
        plt.scatter(Customers_X_coordinates, Customers_Y_coordinates, c='b')
        plt.scatter(Depot_X_coordinate, Depot_Y_coordinate, c='r')

        for i in range (len(X_coordinates)):
            plt.annotate('$%d$' %(i+1), (X_coordinates[i],Y_coordinates[i]), horizontalalignment='left',verticalalignment='bottom')

        plt.grid(True)
        plt.xlabel('X_coordinates',fontsize=20)
        plt.ylabel('Y_coordinates',fontsize=20)
        plt.title('Cities Representation, TSP',fontsize=20)

        plt.yticks(fontsize=22)
        plt.xticks(fontsize=22)

    def compute_distance_Matrix(self):
        D=np.zeros(shape=(self._numberOfCities(),self._numberOfCities()))
        for i in range(len(D)):
            for j in range (len(D)):
                if i==0 and j==0:
                    D[i][j] = self.compute_distance_between_2_customers(self._numberOfCities(),self._numberOfCities())
                if i==0 and j!=0:
                    D[i][j] = self.compute_distance_between_2_customers(self._numberOfCities(),j)
                if j==0 and i!=0:
                    D[i][j] = self.compute_distance_between_2_customers(i,self._numberOfCities())
                else:
                    D[i][j] = self.compute_distance_between_2_customers(i,j)
        return np.negative(D)



class SemiGradientSarSa:

    def __init__(self, alpha, temperature):
        self.alpha = alpha
        self.temperature = temperature


    def Boltzman_policy( self, state, tempertaure_param, w, dictBinCustomerID, listOfVisitedCustomers):
        listActions = self.PossibleActions(dictBinCustomerID)
        approximatedQ = []

        for i in range(len(listActions)):
            approximatedQ.append((np.array(w).reshape(1, 400)@np.array(listActions[i]).reshape(400, 1))[0][0])

        p = np.array([ approximatedQ[x]/tempertaure_param for x in range(len(approximatedQ))])
        pi_actions = np.exp(p) / np.sum(np.exp(p))
        max_action = np.argmax(pi_actions)
        tmpcounter = 1
        
        while(1 == 1):
            if max_action in listOfVisitedCustomers:
                pi_actions[max_action] = 0
                if sum(pi_actions) == 0:
                    max_action  = random.randint(0, len(pi_actions)-1)
                else:
                    max_action = np.argmax(pi_actions)
                tmpcounter += 1
            else:
                return max_action

        return max_action

    def Dict(self, dataFile, tilling):
        dict_bin_customerID = dict()
        dict_coor_customerID = dict()
        customersCoordsList = dataFile.shift_coordinates()

        for i in range(0, 11):
            dict_coor_customerID[i] = customersCoordsList[i]
            dict_bin_customerID[i] = tilling.binCoordinates(customersCoordsList[i][0], customersCoordsList[i][1])
        
        return dict_bin_customerID, dict_coor_customerID

    def InitialState(self, dictBinCustomerID):
        state = dictBinCustomerID.get(len(dictBinCustomerID)-1)
        return state, len(dictBinCustomerID)-1

    def ReturnKeyOfThisValue(self, state, dictBinCustomerID):
        print(state[1])
        listOfKeys = []
        print(listOfKeys)
        return listOfKeys

    def PossibleActions(self, dictBinCustomerID):
        actionList = []
        for i in range(len(dictBinCustomerID)-1):
            actionList.append(dictBinCustomerID.get(i))

        return actionList

    def reward(self, step, S, Sprime, numberOfCustomers, dataFile):
        if step == 1:
            return dataFile.compute_distance_between_2_customers(numberOfCustomers ,Sprime)
        elif step > numberOfCustomers:
            return dataFile.compute_distance_between_2_customers(S, numberOfCustomers)
        
        return dataFile.compute_distance_between_2_customers(S, Sprime)



def main(alpha, gamma, lamda, seedi, epsilon, tempertaure_param):
    random.seed(seedi)
    w = [random.uniform(-0.001, 0.001)] * np.ones([tilingNumber,binNumber*binNumber])*epsilon
    stateSeq = []
    tiling = Tile.Tile(-200, 200, -200, 200, tilingNumber, binNumber, expandRate)
    tiling.generate()
    cwd = os.getcwd()
    path, dirs, files = next(os.walk(cwd+"/Train/"))
    comulativeReward = 0
    for epoch in range(100):
        print("Epoch:")
        print(epoch)
        comulativeReward = 0
        
        for k in range(0,len(files)): 
            stateSeq = []
            policySarsa = SemiGradientSarSa(0.1, 2)
            fullPath = path+files[k]
            dataFile=DataPreProcessing(fullPath)
            dictBinCustomerID, dictCoorCustomerID = policySarsa.Dict(dataFile, tiling)
            s, stateID = policySarsa.InitialState(dictBinCustomerID)
            stateSeq.append(stateID)
            action = policySarsa.Boltzman_policy(s, tempertaure_param, w, dictBinCustomerID, stateSeq)
            singleFileDistance = 0

            for step in range(1, dataFile._numberOfCities()):

                reward = policySarsa.reward(step, stateID, action, dataFile._numberOfCities()-1, dataFile)
                sprime_binary=dictBinCustomerID.get(action)
                stateSeq.append(action)

                if step != dataFile._numberOfCities()-1:
                    actionprime = policySarsa.Boltzman_policy(sprime_binary, tempertaure_param, w, dictBinCustomerID, stateSeq)
                    ssecondprime_binary=dictBinCustomerID.get(actionprime)
                    qhat = np.array(w).reshape(1,400)@np.array(sprime_binary).reshape(400, 1)
                    qhatprime = np.array(w).reshape(1,400)@np.array(ssecondprime_binary).reshape(400, 1)
                    w=w+(alpha*(reward+(gamma * qhatprime)-qhat))*sprime_binary
                    singleFileDistance += reward
                    stateID = action 
                    action = actionprime

            reward = policySarsa.reward(12, stateID, 11, dataFile._numberOfCities()-1, dataFile)
            sBinaryDepot, stateDepotID = policySarsa.InitialState(dictBinCustomerID)
            stateSeq.append(stateDepotID)
            qfinalhat = np.array(w).reshape(1,400)@np.array(sBinaryDepot).reshape(400, 1)
            w=w+(alpha*(reward - qfinalhat))*sBinaryDepot
            singleFileDistance += reward
            comulativeReward += singleFileDistance

        print("comulative reward is:")
        print(comulativeReward)

        #############END OF TRAIN
    path, dirs, files = next(os.walk(cwd+"Test/"))
    
    for k in range(0,len(files)):
        stateSeq = []
        policySarsa = SemiGradientSarSa(0.1, 2)
        fullPath = path+files[k]
        dataFile=DataPreProcessing(fullPath)
        dictBinCustomerID, dictCoorCustomerID = policySarsa.Dict(dataFile, tiling)
        s, stateID = policySarsa.InitialState(dictBinCustomerID)
        stateSeq.append(stateID)
        action = policySarsa.Boltzman_policy(s, tempertaure_param, w, dictBinCustomerID, stateSeq)
        singleFileDistance = 0
        
        for step in range(1, dataFile._numberOfCities()):
            reward = policySarsa.reward(step, stateID, action, dataFile._numberOfCities()-1, dataFile)
            sprime_binary=dictBinCustomerID.get(action)
            stateSeq.append(action)

            if step != dataFile._numberOfCities()-1:
                actionprime = policySarsa.Boltzman_policy(sprime_binary, tempertaure_param, w, dictBinCustomerID, stateSeq)
                singleFileDistance += reward
                stateID = action 
                action = actionprime

        reward = policySarsa.reward(12, stateID, 11, dataFile._numberOfCities()-1, dataFile)
        sBinaryDepot, stateDepotID = policySarsa.InitialState(dictBinCustomerID)
        stateSeq.append(stateDepotID)
        qfinalhat = np.array(w).reshape(1,400)@np.array(sBinaryDepot).reshape(400, 1)
        w=w+(alpha*(reward - qfinalhat))*sBinaryDepot
        singleFileDistance += reward
        print("fileName:")
        print(files[k])
        print("sequence")
        print(stateSeq)
        print("total distance:")
        print(singleFileDistance)
        comulativeReward += singleFileDistance

    print("comulative reward is:")
    print(comulativeReward)
    ##########END OF TEST
    return stateSeq,comulativeReward

listreward=[]
listtemp=[1]

for temp in listtemp:
    print("temp=", temp)
    _, comulativereward = main(0.01, 0.8, 1, 2, 0.05, temp)
    listreward.append(comulativereward)

print(listreward)
