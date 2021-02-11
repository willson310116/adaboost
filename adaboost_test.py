import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

from adaboost import Adaboost

def accuracy(y_true, y_pred):
	accuracy = np.sum(y_true == y_pred) / len(y_true)
	return accuracy

data = datasets.load_breast_cancer()
X = data.data
y = data.target

y[y == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Adaboost classification with 5 weak classifiers

acc_list = []
dim = int(input("# of features\n> "))
# for _ in range(1,data.feature_names.shape[0]):
for _ in range(1,dim):
	old = time.time()
	clf = Adaboost(n_clf=_)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	acc = accuracy(y_test, y_pred)
	acc_list.append(acc)
	new = time.time()
	print(f"Adaboost accuracy with {_} clfs: {round(acc,3)} -------- {round((new-old),3)} sec")

plt.style.use("fivethirtyeight")
plt.title("Adaboost Test")
# plt.plot(list(range(1,data.feature_names.shape[0])),acc_list)
plt.plot(list(range(1,dim)),acc_list)
plt.xlabel("n_clf")
plt.ylabel("Accuracy")
# plt.legend()
plt.tight_layout()
plt.show()
