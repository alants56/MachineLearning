import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

train_set = (pandas.DataFrame(pandas.read_csv("train.csv"))).values
test_set = (pandas.DataFrame(pandas.read_csv("test.csv"))).values

X_train = train_set[:, 1:]
y_train = train_set[:, 0]

print(y_train)

classifier = KNeighborsClassifier(n_neighbors = 10)
classifier.fit(X_train, y_train)
label = classifier.predict(test_set)

print("finish!")


save = pandas.DataFrame({'Label': label})
save.to_csv('submission.csv',index = False,)        