# Importing Packages
# Basic Libraries
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.stats.multitest as smt
from itertools import repeat
# Machine Learning related Libraries
from sklearn.model_selection import cross_validate
# Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# Yellowbrick Viz
from yellowbrick.target import FeatureCorrelation


# Calculating univariate Statistics
def unistats(A, B):
    df_stats = pd.DataFrame(columns=["Fold Change", "log2FoldChange", "p_value", "logpvalue", "t-stat"],
                            index=[A.index])
    count = 0
    for index, row in df_stats.iterrows():
        name = A.iloc[count].name
        fold_change = np.sqrt(
            ((np.mean(10 ** (A.iloc[count]))) / (np.mean(10 ** (B.iloc[count])))) ** 2)  # Caution!! -->
        t_test, p_value = stats.ttest_ind(A.iloc[count], B.iloc[count])
        log_p_value = -np.log10(p_value)
        log_FC = np.log2(fold_change)
        df_stats.loc[name] = [fold_change, log_FC, p_value, log_p_value, t_test]
        count = count + 1
    ## Multiple Testing
    raw_p_values = df_stats.p_value.values
    a = smt.multipletests(raw_p_values, method='fdr_bh')
    se = pd.Series(a[1].tolist())
    df_stats["corrected_pvalues"] = se.values
    return df_stats


def stats_framing(A, B, df):
    df_stats = unistats(A, B)  # Calculating univariate Statistics
    X, y = settingVars(A, B)  # Setting up variables
    df_stats_ml = pearsoncorrelation(X, y, df, df_stats)  # Pearson Correlation
    featurescoresMI = mutualinformation(X, y)  # Mutual Information - Classification
    df_stats_ml = pd.concat([df_stats_ml, featurescoresMI], axis=1)  # Combining stats
    return df_stats_ml


# Preparing Variables
def settingVars(A, B):
    Xa = A.T
    Xb = B.T
    y_list = []
    y_list.extend(repeat(0, len(Xa)))
    y_list.extend(repeat(1, len(Xb)))
    y = pd.DataFrame(y_list)
    #X = Xa.append(Xb)
    X = pd.concat([Xa, Xb])
    y.set_index(X.index, inplace=True)
    y = y.squeeze()

    return (X, y)


# Pearson Correlation
def pearsoncorrelation(X, y, df, df_stats):
    visualizer = FeatureCorrelation(labels=X.columns, sort=True)
    visualizer.fit(X, y)  # Fit the data to the visualizer
    featurescores = pd.DataFrame(visualizer.scores_, visualizer.features_, ['Pearson Correlation Scores'])
    a = df.index.get_level_values(0)
    df_stats.index = a
    df_stats_ml = pd.concat([df_stats, featurescores], axis=1)
    return df_stats_ml


# Mutual Information - Classification
def mutualinformation(X, y):
    visualizer = FeatureCorrelation(method='mutual_info-classification', labels=X.columns, sort=True)
    visualizer.fit(X, y)
    featurescoresMI = pd.DataFrame(visualizer.scores_, visualizer.features_, ['Mutual Information Scores'])
    return featurescoresMI


## ML Algorithmns
# Bayes Naive Classifier
def BNC(X, y):
    gnb_cv = GaussianNB()
    gnb_cv = gnb_cv.fit(X, y)
    cv_results = cross_validate(gnb_cv, X, y, cv=12)
    gnb_cv_score = np.mean(cv_results['test_score'])
    return gnb_cv_score


# k-nearest Neighbour
def KNN(X, y):
    knn_cv = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                  metric='minkowski', metric_params=None, n_jobs=None, )
    knn_cv = knn_cv.fit(X, y)
    knn_results = cross_validate(knn_cv, X, y, cv=10)
    knn_cv_score = np.mean(knn_results['test_score'])
    return knn_cv_score


# Random Forest Classification
def RFC(X, y):
    rf_cv = RandomForestClassifier(random_state=42)
    rf_cv = rf_cv.fit(X, y)
    rf_results = cross_validate(rf_cv, X, y, cv=10)
    rf_cv_score = np.mean(rf_results['test_score'])
    return rf_cv_score


# Support Vector Machine
def lSVC(X, y):
    svc_cv = SVC(kernel='linear')
    svc_cv = svc_cv.fit(X, y)
    svc_results = cross_validate(svc_cv, X, y, cv=10)
    svc_cv_score = np.mean(svc_results['test_score'])
    return svc_cv_score
