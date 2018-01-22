from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor


ridge_regression_pipeline = Pipeline(
        [
            # Apply scaling to Ridge Regression
            # ('scale', StandardScaler()),

            ('ridge', Ridge())
        ]
    )

lasso_regression_pipeline = Pipeline(
        [
            # Apply scaling to Lasso Regression
            # ('scale', StandardScaler()),

            ('lasso', Lasso())
        ]
    )

k_nearest_neighbors_regression_pipeline = Pipeline(
        [
            # Apply scaling to k Nearest Neighbors Regression
            #('scale', StandardScaler()),

            ('knn', KNeighborsRegressor())
        ]
    )


svm_classification_pipeline = Pipeline(
        [
            # Apply PCA to SVM Classification
            ('pca', PCA()),

            # Apply scaling to SVM Classification
            ('scale', StandardScaler()),

            ('svm', SVC())
        ]
    )


k_nearest_neighbors_classification_pipeline = Pipeline(
        [
            # Apply scaling to k Nearest Neighbors Classification
            # ('scale', StandardScaler()),

            ('knn', KNeighborsClassifier())
        ]
    )
