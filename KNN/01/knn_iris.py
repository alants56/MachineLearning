import math
import operator
import pandas


def getdata(filename):
        data = (pandas.DataFrame(pandas.read_csv(filename))).values       
        for i in range(len(data)):
                if i % 4 != 3:
                        traningData.append(data[i])
                else:
                        testdata.append(data[i])
        
        
def edistance(data1, data2, n):
        t = 0
        for i in range(n):
                t += pow((data1[i]-data2[i]), 2)
        return math.sqrt(t)


def getkneighbors(traning, test, k):
        distance = []       
        n = len(test) - 1
        for i in range(len(traning)):
                distance.append((traning[i], edistance(traning[i], test, n)))
        distance.sort(key=operator.itemgetter(1))       
        
        neighbors = []
        for j in range(k):
                neighbors.append(distance[j])
        return neighbors


def getclass(neighbors):
        flag = {}       
        for data in neighbors:
                t = data[0][-1]               
                if t in flag:
                        flag[t] += 1
                else:
                        flag[t] = 1    
                        
        r = sorted(flag.items(), key=lambda obj: obj[1], reverse=True)
        return r[0][0]


traningData = []
testdata = []
        
if __name__ == "__main__":
        getdata("iris.csv")
        for testd in testdata:
                print(testd, end=' ')
                print(": ", getclass(getkneighbors(traningData, testd, 3)))
        
        

   


        


