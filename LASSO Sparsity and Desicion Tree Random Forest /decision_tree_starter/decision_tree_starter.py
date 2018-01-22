import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

eps = 1e-5  # a small number


class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def entropy(y):
        # TODO implement entropy function
        uni_y = np.unique(y)
        prob_y = []
        for i in range(np.size(uni_y)):
            prob_y.append(list(y).count(uni_y[i])/np.size(y)) 
        return - np.dot(np.asarray(prob_y),np.log(np.asarray(prob_y)))

    @staticmethod
    def information_gain(X, y, thresh):
        # TODO implement information gain function
        # X0, y0, X1, y1 = self.split(X, y, 0, thresh)
        # X0, y0, X1, y1 = DecisionTree.split(X = X, y = y, idx = 0, thresh=thresh)
        idx0 = np.where(X[:] < thresh)[0]
        idx1 = np.where(X[:] >= thresh)[0]
        X0, X1 = X[idx0], X[idx1]
        y0, y1 = y[idx0], y[idx1]

        prob_X0 = float(np.size(X0))/(np.size(X0) + np.size(X1))
        prob_X1 = float(np.size(X1))/(np.size(X0) + np.size(X1))
        return DecisionTree.entropy(y) - (prob_X0*DecisionTree.entropy(y0) + prob_X1*DecisionTree.entropy(y1))
        # np.random.rand()

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:,idx] < thresh)[0]
        idx1 = np.where(X[:,idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            thresh = np.array([np.linspace(np.min(X[:, i]) + eps,
                                           np.max(X[:, i]) - eps, num=10) for i
                               in range(X.shape[1])])
            for i in range(X.shape[1]):
                gains.append([self.information_gain(X[:, i], y, t) for t in
                              thresh[i, :]])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(np.argmax(gains),
                                                          gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            X0, y0, X1, y1 = self.split(X, y, idx=self.split_idx,
                                        thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(max_depth=self.max_depth-1,
                                         feature_labels=self.features)
                self.left.fit(X0, y0)
                self.right = DecisionTree(max_depth=self.max_depth-1,
                                          feature_labels=self.features)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode[0]
        return self

    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(X, idx=self.split_idx,
                                                 thresh=self.thresh)
            # print(self.split_idx,self.thresh,'\n')
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat


class BaggedTrees(BaseEstimator, ClassifierMixin):

    def __init__(self, params=None, n=200, sample_percent = 0.2):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.sample_percent = sample_percent
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params) for i in
            range(self.n)]
        self.accu = []
        self.bestAccu = 0
        self.bestTreeInd = None

    def fit(self, X, y):
        # TODO implement function
        for j in range(self.n):
            sub_ind = np.random.choice(range(X.shape[0]), int(self.sample_percent*X.shape[0]), replace = True)
            X_new = X[sub_ind, :]
            y_new = y[sub_ind]
            for i in range(int(1/self.sample_percent)):
                sub_ind = np.random.choice(range(X.shape[0]), int(self.sample_percent*X.shape[0]), replace = True)
                X_new = np.concatenate((X_new, X[sub_ind, :]),axis = 0)
                y_new = np.concatenate((y_new, y[sub_ind]),axis = 0)
            self.decision_trees[j].fit(X_new, y_new)
            self.accu.append(np.sum(self.decision_trees[j].predict(X) == y)/y.shape[0])
        self.bestAccu = np.max(self.accu)
        self.bestTreeInd = np.argmax(self.accu)
        return self
    

    def predict(self, X):
        # TODO implement function
        return self.decision_trees[self.bestTreeInd].predict(X)
        


class RandomForest(BaggedTrees):

    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        # TODO implement function
        self.params = params
        self.n = n
        #self.sample_percent = sample_percent
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params) for i in
            range(self.n)]
        self.accu = []
        self.bestAccu = 0
        self.bestTreeInd = None
        self.m = m
        self.featInd = []
    
    def fit(self, X, y):
        for j in range(self.n):
            sub_ind = np.random.choice(range(X.shape[0]), X.shape[0], replace = True)
            sub_ind_feat = np.random.choice(range(X.shape[1]), self.m, replace = True)
            X_temp = X[sub_ind, :]
            X_new = X_temp[:, sub_ind_feat]
            y_new = y[sub_ind]
            # for i in range(int(1/self.sample_percent)):
            '''for i in range(X.shape[0]-1):
                sub_ind = np.random.choice(range(X.shape[0]), 1, replace = True)
                # sub_ind = np.random.choice(range(X.shape[0]), int(self.sample_percent*X.shape[0]), replace = True)
                X_temp = X[sub_ind, :]
                X_new = np.concatenate((X_new, X_temp[:, sub_ind_feat]),axis = 0)
                y_new = np.concatenate((y_new, y[sub_ind]),axis = 0)'''
            self.featInd.append(sub_ind_feat)
            self.decision_trees[j].fit(X_new, y_new)
            X_accur = X[:, sub_ind_feat]
            self.accu.append(np.sum(self.decision_trees[j].predict(X_accur) == y)/y.shape[0])
        self.bestAccu = np.max(self.accu)
        self.bestTreeInd = np.argmax(self.accu)
        
        return self

    def predict(self, X):
        # TODO implement function
        X_new = X[:, self.featInd[self.bestTreeInd]]
        return self.decision_trees[self.bestTreeInd].predict(X_new)



class BoostedRandomForest(RandomForest):

    def __init__(self, params = None, M_tree = 200, m = 2):
        if params is None:
            params = {}
        # TODO implement function
        self.params = params
        self.M = M_tree
        self.m = m
        # self.decision_trees = DecisionTreeClassifier(random_state=0, **self.params)
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params) for i in
            range(self.M)]
        self.featInd = []



    def fit(self, X, y):
        self.w = np.ones(X.shape[0]) / X.shape[0]  # Weights on data
        self.a = np.zeros(self.M)  # Weights on decision trees
        
        # TODO implement function
        for i in range(self.M):
            # generate samples using weight
            sub_ind = np.random.choice(range(X.shape[0]), X.shape[0], replace = True, p = self.w/np.sum(self.w))
            sub_ind_feat = np.random.choice(range(X.shape[1]), self.m, replace = True)
            self.featInd.append(sub_ind_feat)
            X_temp = X[sub_ind, :]
            X_new = X_temp[:, sub_ind_feat]
            y_new = y[sub_ind]
            self.decision_trees[i].fit(X_new, y_new)
            X_test = X[:, sub_ind_feat]
            e = np.dot(self.decision_trees[i].predict(X_test) == y, self.w)/np.sum(self.w)
            self.a[i] = np.log((1-e)/e)/2
            # print(e)
            ind_0 = np.where((self.decision_trees[i].predict(X_test) == y) == False)
            ind_1 = np.where((self.decision_trees[i].predict(X_test) == y) == True)
            self.w[ind_0] = self.w[ind_0] * np.exp(-self.a[i])
            self.w[ind_1] = self.w[ind_1] * np.exp(self.a[i])

        return self

    def predict(self, X):
        # TODO implement function
        pred_0 = np.zeros(X.shape[0])
        pred_1 = np.zeros(X.shape[0])
        for i in range(self.M):
            pred_0 = pred_0 + self.a[i]*(self.decision_trees[i].predict(X[:,self.featInd[i]]) == np.zeros(X.shape[0]))
            pred_1 = pred_1 + self.a[i]*(self.decision_trees[i].predict(X[:,self.featInd[i]]) == np.ones(X.shape[0]))

        
        temp = pred_0 - pred_1
        temp[temp >= 0] = 1
        temp[temp < 0] = 0
        return temp

        #pass


if __name__ == "__main__":
    #dataset = "spam"
    dataset = "titanic"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100
    if dataset == "titanic":
        # Load titanic data
        # path_train = 'datasets/titanic/titanic_training.csv'
        path_train = 'titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=np.float32)
        features = data[0, 1:]  # features = all columns except survived
        y = data[1:, 0]  # label = survived

        #change the original data
        data[np.isnan(data)]=0
        # class_names = ["Died", "Survived"]

        # TODO implement preprocessing of Titanic dataset
        X_pre = np.asarray([data[1:,1],data[1:,2],data[1:,4],data[1:,5],data[1:,7],data[1:,3],data[1:,6]])
        a = OneHotEncoder(categorical_features = np.array([0,1,2,3,4]))
        a.fit(X_pre.T)

        
        X = a.transform(X_pre.T).toarray()


        path_test = 'titanic_testing_data.csv'
        data_test = genfromtxt(path_test, delimiter=',', dtype=np.float32)
        data_test[np.isnan(data_test)]=0
        Z_pre = np.asarray([data_test[1:,0],data_test[1:,1],data_test[1:,3],data_test[1:,4],data_test[1:,6],data_test[1:,2],data_test[1:,5]])
        Z = a.transform(Z_pre.T).toarray()



    elif dataset == "spam":
        features = ["pain", "private", "bank", "money", "drug", "spam",
                    "prescription", "creative", "height", "featured", "differ",
                    "width", "other", "energy", "business", "message",
                    "volumes", "revision", "path", "meter", "memo", "planning",
                    "pleased", "record", "out", "semicolon", "dollar", "sharp",
                    "exclamation", "parenthesis", "square_bracket", "ampersand"]
        assert len(features) == 32

        # Load spam data
        # path_train = 'datasets/spam_data/spam_data.mat'
        path_train = 'spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]
    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)

    print("\n\nPart 0: constant classifier\n")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # Basic decision tree
    print("\n\nPart (a-b): simplified decision tree\n")

    kf = KFold(n_splits=3, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        dt = DecisionTree(max_depth = 3, feature_labels=features)
        dt.fit(X_train, y_train)
        print(dt.split_idx, dt.thresh)
        print('training accuracy :')
        print(np.sum(dt.predict(X_train) == y_train)/y_train.shape[0])
        print('validation Accuracy :')
        print(np.sum(dt.predict(X_test) == y_test)/y_test.shape[0],'\n')
    # print("Predictions", dt.predict(Z)[:100])
    # print(Z.shape)

    
    # TODO implement and evaluate parts c-h
    # d part bagged tree
    print("\n\nPart (d): Bagged decision tree\n")

    kf = KFold(n_splits=3, shuffle=True)

    for train_index, test_index in kf.split(X):
        dt = BaggedTrees(params=params,sample_percent = 0.3)
        dt.fit(X_train, y_train)

        # calculate common split on the root
        feature_split = []
        for i in range(dt.n):
            feature_split.append((dt.decision_trees[i].tree_.feature[0],dt.decision_trees[i].tree_.threshold[0]))
        result = dict((i, feature_split.count(i)) for i in feature_split)
        maximum = max(result, key=result.get) 
        print(maximum, result[maximum])

        print('training Accuracy :')
        print(dt.bestAccu)
        print('validation Accuracy :')
        print(np.sum(dt.predict(X_test) == y_test)/y_test.shape[0],'\n')


    #print(dt.accu)
    
    # np.savetxt('submission.txt',dt.predict(Z)[:].astype(np.int8))

    


    # f) random forest
    print("\n\nPart (f): Random Forest decision tree\n")

    kf = KFold(n_splits=3, shuffle=True)

    for train_index, test_index in kf.split(X):
        dt = RandomForest(params=params, n=200, m=15)
        dt.fit(X_train, y_train)
    
        
        feature_split = []
        for i in range(dt.n):
            feature_split.append((dt.decision_trees[i].tree_.feature[0],dt.decision_trees[i].tree_.threshold[0]))
        result = dict((i, feature_split.count(i)) for i in feature_split)
        maximum = max(result, key=result.get) 
        print(maximum, result[maximum])

        print('training Accuracy:')
        print(dt.bestAccu)
        print('validation Accuracy:')
        print(np.sum(dt.predict(X_test) == y_test)/y_test.shape[0],'\n')

    
    # np.savetxt('submission.txt',dt.predict(Z)[:].astype(np.int8))
    
    
    
    
    
    # h) Boosted random forest
    print("\n\nPart (h): Boosted Random Forest decision tree\n")

    kf = KFold(n_splits=3, shuffle=True)

    for train_index, test_index in kf.split(X):
        dt = BoostedRandomForest(params=params, M_tree = 8, m = 10)
        dt.fit(X_train, y_train)
    
        # calculate common split on the root
        feature_split = []
        for i in range(dt.M):
            feature_split.append((dt.decision_trees[i].tree_.feature[0],dt.decision_trees[i].tree_.threshold[0]))
        result = dict((i, feature_split.count(i)) for i in feature_split)
        maximum = max(result, key=result.get) 
        print(maximum, result[maximum])

        print('training Accuracy:')
        print(np.sum(dt.predict(X_train) == y_train)/y_train.shape[0])
        print('validation Accuracy:')
        print(np.sum(dt.predict(X_test) == y_test)/y_test.shape[0],'\n')


    

