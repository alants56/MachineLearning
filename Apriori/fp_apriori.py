import pandas as pd
from mlxtend.frequent_patterns import apriori


data = pd.DataFrame(pd.read_csv("supermarket.csv"))
#clean
data.replace(['?','t'], [0, 1], inplace=True)

#split
data = data.iloc[:, 0: -1]

#frequent intemsets mining
frequent_itemsets = apriori(data, min_support=0.3, use_colnames=True)
print(frequent_itemsets)