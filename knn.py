#Importing the necessary packages
import numpy as np
from collections import Counter
#for the datasets
from sklearn import datasets
from sklearn.model_selection import train_test_split

#for proving our algo
from sklearn.neighbors import KNeighborsClassifier

#impleting the K-Nearest neighbours algorithm
class KNN:
    def __init__(self,n_neighbors=3):
        self.n_neighbors=n_neighbors

    def fit(self,X,y):
        self.X_train=X
        self.y_train=y

    def euclidean_distance(self,x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2)) #measures the euclidean distance between points


    def predict(self,X):
        predicted_labels=[self._predict(x) for x in X]
        self.predictions=np.array(predicted_labels)
        return self.predictions # returns the prediction of the model

    def _predict(self,x):
        distances=[self.euclidean_distance(x,x_train) for x_train in self.X_train]
        neighbors=np.argsort(distances)[:self.n_neighbors] #arranges the value based on closeness and takes first N neighbors
        neighbors_labels=[self.y_train[i] for i in neighbors] #gets the label to the values
        predicted_value=Counter(neighbors_labels).most_common(1) #gets the most common value
        return predicted_value[0][0]

    def score(self,X_test,y_test):
        predictions=self.predict(X_test)
        accuracy=np.sum(predictions==y_test)/len(y_test) #gets the accuracy of the y_test
        print("Accurracy:{}%".format(accuracy*100)) #prints the result
        return accuracy #also returns the value of the accuracy






#time to test

iris= datasets.load_iris()
X,y= iris.data, iris.target


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=1234)

#Instantiating the two KneighborsClassifier and KNN
clf_real=KNeighborsClassifier(n_neighbors=3)
clf=KNN(n_neighbors=3)

#fitting the model
clf_real.fit(X_train,y_train)
clf.fit(X_train,y_train)
predictions_real=clf_real.predict(X_test)
predictions=clf.predict(X_test)

#predicting
score_real=clf_real.score(X_test,y_test)

score=clf.score(X_test,y_test)# it works! and gives the same result for accuracy

# Do they give the same accuracy?
assert(score==score_real)