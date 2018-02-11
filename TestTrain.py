import csv
import pandas
import sklearn   #we use the model_selection module in the sklearn library
from sklearn import model_selection
from sklearn import svm   #for SVM
from sklearn.ensemble import AdaBoostClassifier #for Adaboost
from sklearn.tree import DecisionTreeClassifier #for Adaboost
from sklearn.neighbors.nearest_centroid import NearestCentroid #for K Nearest Neighbour Centroid
from sklearn.ensemble import RandomForestClassifier #for Random Forest Classifier
from sklearn.neural_network import MLPClassifier #for MLP Classifier
from sklearn.neighbors import KNeighborsClassifier #for KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier #for Gaussian Process Classifier
from sklearn.naive_bayes import GaussianNB #for Gaussian Naive Bayes
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis #for Quadratic Discriminant Analysis
from sklearn.linear_model import LogisticRegression #for Logistic Regression
import matplotlib
import scipy
import sys


#Iris-setosa=0, Iris-versicolor=1, Iris-virginica=2

def model_runner(model,name):
	model_name=model
	model_name.fit(X_train,Y_train)
	resultTest=model_name.predict(X_validation)

	write=pandas.DataFrame({'Class':resultTest})
	write.to_csv(name,index=False)

	#print(resultTest)





#training
train=pandas.read_csv('IrisTrain.csv')
temp1=['SepalLength(cm)','SepalWidth(cm)','PetalLength(cm)','PetalWidth(cm)']
X_train=train[temp1]
temp2=['Class']
#print(X_train)
Y_train=train[temp2]

#testing
test=pandas.read_csv('IrisTest.csv')
X_validation=test[temp1]

#Calling models
#Model Number 1 SVM
svm1=svm.SVC()
name="SVM.csv"
model_runner(svm1,name)

#Model Number 2 Adaboost
ada1=AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),n_estimators=300)
name="AdaBoostClassifier.csv"
model_runner(ada1,name)

#Model Number 3 K Nearest Neighbour
knn1=NearestCentroid()
name="NearestCentroid.csv"
model_runner(knn1,name)

#Model Number 4 Random Forest Classifier
rfc1=RandomForestClassifier(max_depth=2,random_state=0)
name="RandomForestClassifier.csv"
model_runner(rfc1,name)

#Model Number 5 MLP Classifier
mlp1=MLPClassifier(hidden_layer_sizes=(100,), activation="relu",solver='adam', alpha=0.0001,batch_size='auto', learning_rate="constant",learning_rate_init=0.001,power_t=0.5, max_iter=200, shuffle=True,random_state=0, tol=1e-4,verbose=False, warm_start=False, momentum=0.9,nesterovs_momentum=True, early_stopping=False,validation_fraction=0.1, beta_1=0.9, beta_2=0.999,epsilon=1e-8)
name="MLPClassifier.csv"
model_runner(mlp1,name)  #This model runs with a convergence warning

#Model Number 6 K Neighbour Classifier
knc1=KNeighborsClassifier(n_neighbors=3)
name="KNeighborsClassifier.csv"
model_runner(knc1,name)

#Model Number 7 Gaussian Process Classifier
gpc1=GaussianProcessClassifier(kernel=None, n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None, n_jobs=1)
name="GaussianProcessClassifier.csv"
model_runner(gpc1,name)  #Remove optimizer=’fmin_l_bfgs_b’ and multi_class=’one_vs_rest’

#Model Number 8 Gaussian Naive Bayes
gnb1=GaussianNB()
name="GaussianNB.csv"
model_runner(gnb1,name)

#Model Number 9 Quadratic Discriminant Analysis
qda1=QuadraticDiscriminantAnalysis()
name="QuadraticDiscriminantAnalysis.csv"
model_runner(qda1,name)

#Model Number 10 Logistic Regression
lor1=LogisticRegression()
name="LogisticRegression.csv"
model_runner(lor1,name)