import pandas as pd


def get_dic(data) :
    dic = {}
    for d in data :
        for x in d :
            if x in dic :
                dic[x] += 1
            else : 
                dic[x] = 1
    return dic


def get_frequent_itemsets_1(data, min_sup) :
    dic = get_dic(data)
    dic = [[{x},dic[x]] for x in dic]
    return list(filter(lambda x :  x[1] >= min_sup, dic))

def get_next_set(C, length) :
    T = []
    n = len(C)
    for i in range(n) :
        for j in range(i, n) :
            t = C[i]|C[j]
            if len(t) == length and not t in T:
                T.append(t)
    return T

def apriori(data, min_support = 0.1) :
    data = [set(x) for x in data]
    currentlen = 1
    min_sup = len(data)*min_support


    frequent_itemsets = get_frequent_itemsets_1(data, min_sup)

    C = [x[0] for x in frequent_itemsets]
    
    while True :
        C = get_next_set(C,currentlen+1)

        if not C :
            break

        freq_items_0 = [[c, 0] for c in C]
        for items in data :
            for fi in freq_items_0 :
                if fi[0] == fi[0]&items :
                    fi[1] += 1

        freq_items_1 = list(filter(lambda x :  x[1] >= min_sup, freq_items_0))

        frequent_itemsets += freq_items_1
        
        C = [x[0] for x in freq_items_1]
        
        currentlen += 1

    return frequent_itemsets

def output_fi(frequent_itemsets, num) :
    print(''.ljust(5), 'support'.ljust(25), 'itemset'.ljust(100))
    for i in range(len(frequent_itemsets)) :
        print(str(i+1).ljust(5),str(frequent_itemsets[i][1]/num).ljust(25),str(frequent_itemsets[i][0]).ljust(100))

def name_eval(s) :
    try :
        s = eval(s)
    except :
        pass
    return s

def load_data_csv(Path) :
    dataframe = pd.DataFrame(pd.read_csv(Path))
    dataframe = dataframe.iloc[:, 0: -1]
    name, values = dataframe.columns.values.tolist(), dataframe.values.tolist()
    name = [name_eval(s) for s in name]
    data = [list(filter(lambda x:  value[name.index(x)] is 't', name))  for value in values]
    return data

def main() :
    data = load_data_csv('supermarket.csv')
    frequent_itemsets =apriori(data, min_support = 0.3)
    output_fi(frequent_itemsets, len(data))
    

if __name__ == '__main__':
    main()
