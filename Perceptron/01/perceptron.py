class perceptron (object) :
        def __init__(self, w, b, lr) :
                self.w = w
                self.b = b
                self.lr = lr               
        def train(self, x, y) :
                n = 0
                while(True) :
                        num = 0
                        for i in range(len(x)) :
                                r = self.test(x[i])   
                                if r == y[i] :
                                        num += 1
                                        continue
                                else :
                                        self.update_w(x[i],y[i])
                                        self.update_b(y[i])
                                        n += 1                                    
                        if len(x) == num :
                                print("N = ", n)
                                return

        def test(self, x) :
                return self.sign(x*self.w + self.b)
                
        def update_w(self, x, y) :
                self.w += x * y * self.lr
                
        def update_b(self, y) :
                self.b += y * self.lr
        
        def output_S(self) :
                print("S: ", self.w, "* x + ", self.b, " = 0")
                
        def sign(self, x) :
                if x > 0 :
                        return 1
                else :
                        return -1

if __name__ == '__main__'    :
        x = [ 1,  3,   6,  5,  4]
        y = [-1, -1,  1,  1, -1]
        per = perceptron(0,0,1)  
        per.train(x,y)
        per.output_S()
        print("Predict:")
        print("x = 1 ", "y = ",per.test(1))
        print("x = 7 ", "y = ",per.test(7))
         