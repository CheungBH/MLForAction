from sklearn.model_selection import cross_val_score
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

# ==================Bagging 元估计器=============
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
scores = cross_val_score(bagging, X, y)
print('Bagging准确率：', scores.mean())