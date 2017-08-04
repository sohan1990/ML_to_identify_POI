import sys
sys.path.append("../tools/")
import pickle
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from tester import test_classifier


## Function to get a particular value out of dictionary of dictionaries
def dict_values(dict, item):
    new_list = []
    for name in dict:
        new_list.append(dict[name][item]) if not isinstance(dict[name][item], str) \
            else new_list.append(0)
    return new_list

## Data load and pre-process
def data_load_process():
    # Load the dictionary containing the data set
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    # visualizing data
    print len(data_dict)

    salary = dict_values(data_dict, 'salary')
    bonus = dict_values(data_dict, 'bonus')
    plt.scatter(salary, bonus)
    plt.title('Salary vs bonus')
    plt.ylabel('bonus ')
    plt.xlabel('salary')
    plt.show()

    # Removing outliers and re-visualizing
    # From scatter plot above we find 'TOTAL' to be an outlier
    outliers = ['TOTAL']
    # checking for more outliers
    outliers.append('THE TRAVEL AGENCY IN THE PARK')
    # checking for all missing data for an entry
    for name in data_dict:
        flag = 0
        for item in data_dict[name]:
            if data_dict[name][item] != 'NaN':
                if item != 'poi':
                    flag = 1
        if flag == 0:
            outliers.append(name)
    # removing outliers
    for outlier in outliers:
        data_dict.pop(outlier, 0)
    #print data_dict

    salary = dict_values(data_dict, 'salary')
    bonus = dict_values(data_dict, 'bonus')
    poi = dict_values(data_dict, 'poi')
    plt.scatter(salary, bonus, s = 100, c = poi)
    plt.title('Salary vs bonus after removing outlier')
    plt.ylabel('bonus ')
    plt.xlabel('salary')
    plt.show()

    poi_count = 0
    for person in data_dict:
        if data_dict[person]['poi']:
            poi_count = poi_count + 1

    print 'POI count = {}'.format(poi_count)
    print 'NON-POI count = {}'.format(len(data_dict) - poi_count)
    return data_dict

## looking at features
def look_at_features(data_dict):
    features = data_dict['METTS MARK'].keys()
    print 'The features present are: {}'.format(features)
    print 'Total no of features = {}'.format(len(features))
    feature_list = dict()
    for feature in features:
        feature_list[feature] = 0
        for name in data_dict:
            if data_dict[name][feature] == 'NaN':
                feature_list[feature] = feature_list[feature] + 1
    features_list = sorted(feature_list.items(), key = lambda x:x[1], reverse=True)
    print features_list
    return 0

## Create new feature(s)
def add_new_features(features_list, data_dict):

    new_features = ['ratio_of_to_poi_messages',
                          'ratio_of_from_poi_messages',
                          'total_poi_messages',
                          'salary_square',
                          'bonus_square',
                          'total_stock_value_square',
                          'incentive_cube',
                          'bonus_salary_ratio']
    features_list = features_list + new_features

    for name in data_dict:
        # To make all non numeric values into numeric
        if isinstance(data_dict[name]['from_this_person_to_poi'], str):
            data_dict[name]['from_this_person_to_poi'] = 0
        if isinstance(data_dict[name]['to_messages'],str):
            data_dict[name]['to_messages'] = 0
        if isinstance(data_dict[name]['from_poi_to_this_person'], str):
            data_dict[name]['from_poi_to_this_person'] = 0
        if isinstance(data_dict[name]['from_messages'], str):
            data_dict[name]['from_messages'] = 0

        # Creating ratio_of_to_poi_messages
        # try/except is to ensure that infinite values get dealt with correctly
        try:
            ratio_of_to_poi_messages = float(data_dict[name]['from_this_person_to_poi']) /\
                                       float(data_dict[name]['from_messages'])
        except:
            ratio_of_to_poi_messages = 0

        # Creating ratio_of_from_poi_messages
        # try/except is to ensure that infinite values get dealt with correctly
        try:
            ratio_of_from_poi_messages = float(data_dict[name]['from_poi_to_this_person']) /\
                                         float(data_dict[name]['to_messages'])
        except:
            ratio_of_from_poi_messages = 0

        # Creating total_poi_messages
        data_dict[name]['total_poi_messages'] = data_dict[name]['from_this_person_to_poi'] +\
                                                data_dict[name]['from_poi_to_this_person']
        # Inserting the new features into the dictionary
        data_dict[name]['ratio_of_to_poi_messages'] = ratio_of_to_poi_messages
        data_dict[name]['ratio_of_from_poi_messages'] = ratio_of_from_poi_messages


        if(data_dict[name]['salary'] != 'NaN'):
            data_dict[name]['salary_square'] = data_dict[name]['salary'] ** 2
        else:
            data_dict[name]['salary_square'] = 0
        if (data_dict[name]['bonus'] != 'NaN'):
            data_dict[name]['bonus_square'] = data_dict[name]['bonus'] ** 2
        else:
            data_dict[name]['bonus_square'] = 0
        if (data_dict[name]['total_stock_value'] != 'NaN' and data_dict[name]['total_stock_value'] > 0):
            data_dict[name]['total_stock_value_square'] = data_dict[name]['total_stock_value'] ** 2
        else:
            data_dict[name]['total_stock_value_square'] = 0
        if (data_dict[name]['long_term_incentive'] != 'NaN' and data_dict[name]['long_term_incentive'] > 0):
            data_dict[name]['incentive_square'] = data_dict[name]['long_term_incentive'] ** 2
        else:
            data_dict[name]['incentive_square'] = 0
        if (data_dict[name]['long_term_incentive'] != 'NaN' and data_dict[name]['long_term_incentive'] > 0):
            data_dict[name]['incentive_cube'] = data_dict[name]['long_term_incentive'] ** 3
        else:
            data_dict[name]['incentive_cube'] = 0
        if (data_dict[name]['salary'] != 'NaN' and data_dict[name]['bonus'] != 'NaN'):
            data_dict[name]['bonus_salary_ratio'] = data_dict[name]['bonus'] / data_dict[name]['salary']
        else:
            data_dict[name]['bonus_salary_ratio'] = 0

    return features_list, data_dict

## Adding all the classifier functions
def classifier_list():

    clf_list = []
    # NAIVE BAYES
    #from sklearn.naive_bayes import GaussianNB
    #clf_NB = GaussianNB()
    #clf_NB_params = {}
    #clf_list.append((clf_NB, clf_NB_params))

    # DECISION TREE
    clf_tree = DecisionTreeClassifier()
    clf_tree_params = {"min_samples_split": [2,5,10],
                      "criterion": ["gini", "entropy"],
                       "random_state": [47]}
    clf_list.append((clf_tree, clf_tree_params))

    # LOGISTIC REGRESSION
    clf_logreg = LogisticRegression()
    clf_logreg_params = {"C": [0.05, 0.5, 1, 10, 10**2,10**5,10**10, 10**20],
                         "tol": [10**-1, 10**-5, 10**-10],
                         "random_state": [47]}
    clf_list.append((clf_logreg, clf_logreg_params))

    # ADABOOST
    clf_ada = AdaBoostClassifier()
    clf_ada_params = {"n_estimators": [20, 30, 40, 50],
                      #"learning_rate": [0.1, 0.5, 1, 2, 5],
                      "algorithm": ["SAMME", "SAMME.R"],
                      "random_state": [47]}
    clf_list.append((clf_ada, clf_ada_params))

    # RANDOM FOREST
    clf_rforest = RandomForestClassifier()
    clf_rforest_params = {"n_estimators": [20, 30, 40, 50],
                          "criterion": ["gini", "entropy"],
                          "min_samples_split": [2, 5, 10],
                          "random_state": [47]}
    clf_list.append((clf_rforest, clf_rforest_params))

    # SVM
    clf_svm = LinearSVC()
    clf_svm_params = {"C": [1, 5, 10, 15, 20],
                     "tol": [0.1, 0.01, 0.001, 0.0001],
                     "dual": [False],
                     "random_state" : [47]}
    clf_list.append((clf_svm, clf_svm_params))

    """"""
    return clf_list

## Using pipeline to add PCA to all classifiers. Also updating the parameter list
def add_pca(clf_list):

    pca = PCA()
    pca_params = {'pca__n_components': [5, 10, 15, 20],
                  "pca__random_state": [47]}
    delimiter = '('
    clf_list_new = []
    for classifier in clf_list:
        clf, params = classifier
        name = str(clf).split(delimiter)
        new_param = {}
        for param, values in params.iteritems():
            new_param[name[0] + '__' + param] = values

        clf_pipeline = Pipeline([('pca', pca), (name[0], clf)])
        #print clf_pipeline

        new_param.update(pca_params)
        #print new_param

        clf_list_new.append((clf_pipeline, new_param))
    return clf_list_new

## Applying GridSearch to the classifier list to select best parameters
def classify(clf_list, features_train, labels_train):

    best_clf_list = []
    for classifier, params in clf_list:
        clf_rev = GridSearchCV(classifier, params)
        clf_rev.fit(features_train, labels_train)
        try:
            best_clf_list.append(clf_rev.best_estimator_)
        except:
            print "no best estimator available for this classifier"
    return best_clf_list

## Function to evaluate the accuracy, recall and precision of the classifiers
def evaluate(clf_list, features_test, labels_test):

    new_list = []
    for clf in clf_list:
        pred = clf.predict(features_test)
        recall = recall_score(labels_test, pred)
        precision = precision_score(labels_test, pred)
        f1 = f1_score(labels_test, pred)
        acc = accuracy_score(labels_test, pred)
        new_list.append((clf, recall, precision, acc, f1))
    return new_list

## using tester to determine classifier
def evaluate2(clf_list, my_dataset, features_list):
    for clf in clf_list:
        test_classifier(clf, my_dataset, features_list)
    return 1

### Sort the classifier list according to their f1-score
def sort_clf(clf_list):
    clf_list_sorted = sorted(clf_list, key = lambda x:x[1], reverse=True)
    return clf_list_sorted

### Best classifier tuning
def best_classifier():
    clf = []
    clf_tree = DecisionTreeClassifier()
    clf_tree_params = {"min_samples_split": [2, 5, 10],
                       "criterion": ["gini", "entropy"],
                       "random_state": [46]}
    clf.append((clf_tree, clf_tree_params))
    return clf

