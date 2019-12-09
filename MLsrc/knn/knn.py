from sklearn.neighbors import KNeighborsClassifier
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
data=[]
traffic_feature=[]
traffic_target=[]
csv_file = csv.reader(open('../RFData.csv'))
for content in csv_file:
    content=list(map(float,content))
    if len(content)!=0:
        data.append(content)
        traffic_feature.append(content[0:6])
        traffic_target.append(content[-1])
print('data=',data)
print('traffic_feature=',traffic_feature)
print('traffic_target=',traffic_target)
feature_train, feature_test, target_train, target_test = train_test_split(traffic_feature, traffic_target, test_size=0.3,random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(feature_train,target_train)
predict_results=knn.predict(feature_test)
print(accuracy_score(predict_results, target_test))
conf_mat = confusion_matrix(target_test, predict_results)
print(conf_mat)
print(classification_report(target_test, predict_results))
