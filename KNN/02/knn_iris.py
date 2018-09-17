import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

dataset = (pandas.DataFrame(pandas.read_csv("iris.csv"))).values
X = dataset[:, 0:3]
y = dataset[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(y_test == y_pred)
