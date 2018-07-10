import numpy as np
import math
import pandas

def max_num(a,b) :
        if a > b :
                return a
        else :
                return b

def max_pooling(A, n) :
        B = np.zeros((n,n))
        for i in range(n) :
                for j in range(n) :
                        a = max_num(A[2*i] [ j*2],  A[2*i] [ j*2+1])
                        b = max_num(A[2*i+1] [ j*2], A[2*i+1 ] [j*2+1])
                        B[i,j] = max_num(a,b)
        return B
        
def converlution(A, B, n, s) :
        C = np.zeros((n,n))
        for i in range(n) :
                for j in range(n) :
                        C[i][j] = np.sum(A[i:i+s, j:j+s] * B)
        return C
     
def sigmod(x) :
        c = math.e**(-x)
        return 1 / (1 + c)
        
def f_sigmod(y) :
        for i in range(len(y)) :
                y[i] = sigmod(y[i])
        return y
        
def cnetwork(x, w, b) :
        y = np.zeros(10)
        for i in range(10) :
                for h in range(144) :
                        y[i] += x[h] * w[h][i] 
                y[i] += b[i]        
        y = f_sigmod(y)
        return y
 
def max_array(y) :
        n = len(y)
        t = y[0]
        j = 0
        for i in range(n) :
                if t < y[i] :
                        t = y[i]
                        j = i
        return j               
         
def update_w(w, step, z, y, r) :
        d = y - r
        for i in range(144) :
                for h in range(10) :
#                        w[i][h] = w[i][h] - step * d[h] * x[i]*(sigmod(y[h])*(1-sigmod(y[h]))) 
                        w[i][h] = w[i][h] - step * d[h] * x[i]
        return w

def update_b(b, step, y, r) :
        d = y - r
        for i in range(10) :
#                b[i] = b[i] - step * d[i]*(sigmod(y[i])*(1-sigmod(y[i]))) 
                b[i] = b[i] - step * d[i]
        return b
        
if __name__ == "__main__" :
        
        step1 = 0.01
        step2 = 0.01
        data = pandas.DataFrame(pandas.read_csv("train.csv"))
        Y = data.values[:, 0]
        X = data.values[:, 1:]
        num = len(Y) 
        
    
        f1 = np.array([[0,1,0],[0,1,0],[0,1,0]])
        f2 = np.array([[0,0,0],[1,1,1],[0,0,0]])
        f3 = np.array([[1,0,0],[0,1,0],[0,0,1]])   
        f4 = np.array([[0,0,1],[0,1,0],[1,0,0]])   
     
        w = np.random.random((144, 10))
        b = np.random.random(10)
        
        print("Training ...")
        for j in range(3) :
                print(j)
                for i in range(num) :
                        A = max_pooling(X[i,:].reshape((28,28)), 14)
                        B1 = converlution(A, f1, 12, 3)
                        B2 = converlution(A, f2, 12, 3)
                        B3 = converlution(A, f3, 12, 3)
                        B4 = converlution(A, f4, 12, 3)
                        C1 = max_pooling(B1, 6)
                        C2 = max_pooling(B2, 6)
                        C3 = max_pooling(B3, 6)
                        C4 = max_pooling(B4, 6)
                        x = np.append(np.append(C1.flatten(),C2.flatten()), np.append(C3.flatten(),C4.flatten()))
                        y = cnetwork(x, w, b)                            
                        r = np.zeros(10)
                        r[Y[i]] = 1
                        b = update_b(b, step1, y,r)
                        w = update_w(w, step2, x, y,r)
                step1 = step1/10
                step2 = step2/10
                
        error = 0
        for j in range(num) :
                A = max_pooling(X[j,:].reshape((28,28)), 14)
                B1 = converlution(A, f1, 12, 3)
                B2 = converlution(A, f2, 12, 3)
                B3 = converlution(A, f3, 12, 3)
                B4 = converlution(A, f4, 12, 3)
                C1 = max_pooling(B1, 6)
                C2 = max_pooling(B2, 6)
                C3 = max_pooling(B3, 6)
                C4 = max_pooling(B4, 6)
                x = np.append(np.append(C1.flatten(),C2.flatten()), np.append(C3.flatten(),C4.flatten()))
                y = cnetwork(x, w, b)                                          
                if Y[j] != max_array(y) :
                        error += 1
                        
                        
        print("error:")
        print(error)
        print("Scored:")                       
        print(1- error/num) 

         
        print("Testing ...")
        test = pandas.DataFrame(pandas.read_csv("test.csv"))
        T = test.values
        num = len(T)
        id = []
        label = []
        for k in range(num) :
                A = max_pooling(T[k,:].reshape((28,28)), 14)
                B1 = converlution(A, f1, 12, 3)
                B2 = converlution(A, f2, 12, 3)
                B3 = converlution(A, f3, 12, 3)
                B4 = converlution(A, f4, 12, 3)
                C1 = max_pooling(B1, 6)
                C2 = max_pooling(B2, 6)
                C3 = max_pooling(B3, 6)
                C4 = max_pooling(B4, 6)
                x = np.append(np.append(C1.flatten(),C2.flatten()), np.append(C3.flatten(),C4.flatten()))
                y = cnetwork(x, w, b)                                          
                id.append(k+1)
                label.append(max_array(y))
        
        save = pandas.DataFrame({'ImageId': id, 'Label': label})
        save.to_csv('submission.csv',index = False,)
             
        
                
#output: 
#      train data scored: 0.8666
#      test data scored:  0.8597  
                
           
        
         

        
