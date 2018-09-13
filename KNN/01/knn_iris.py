import math
import operator
from sklearn import datasets
import numpy as np
import pandas

def getData(filename) :
        data = (pandas.DataFrame(pandas.read_csv(filename))).values       
        for i in range(len(data)) :
                if i % 4 != 0 :
                        traningData.append(data[i])
                else :
                        testdata.append(data[i])
        
        
def edistance(data1, data2, n) :
        sum = 0
        for i in range(n) :
                sum += pow((data1[i]-data2[i]), 2)           
        return math.sqrt(sum)
 
def getKneighbors(traningData, testdata, k):
        distance = []       
        n = len(testdata) - 1
        for i in  range(len(traningData)) :              
                distance.append((traningData[i], edistance(traningData[i], testdata, n)))
        distance.sort(key=operator.itemgetter(1))       
        
        neighbors = []
        for j in range(k) :
                neighbors.append(distance[j])
        return neighbors

def getClass(neighbors) :
        flag = {}       
        for data in neighbors :
                t = data[0][-1]               
                if t in flag :
                        flag[t] += 1
                else :
                        flag[t] = 1            
        r = sorted(flag.items(), key=lambda obj: obj[1],reverse=True)   
        return (r[0][0])

traningData = []
testdata = []
        
if __name__ == "__main__" : 
        getData("iris.csv")       
        for testd in testdata :
                print(testd, end = ' ')
                print(": ",getClass(getKneighbors(traningData, testd, 3)))
        
        

   


        


