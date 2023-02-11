import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

clf = svm.SVC(gamma=0.001, C=100)
clf.fit(X_train, y_train)

print(clf.predict(X_test[0:1]))
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

clf = svm.SVC(gamma=0.001, C=100)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"Was: {accuracy * 100 :0.1f}% correct")
