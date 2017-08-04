### IMPORTING ALL THE LIBRARIES AT ONE PLACE

#!/usr/bin/python

#import sys
#sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.model_selection import train_test_split

### Functions custom made for this project are imported
from functions import dict_values, data_load_process, look_at_features, add_new_features
from functions import classifier_list, add_pca, classify, evaluate, evaluate2, sort_clf
from functions import best_classifier


### MAIN BODY

## Selecting features to use
poi = ['poi']

financial_features = ['salary',
                      'deferral_payments',
                      'total_payments',
                      'loan_advances',
                      'bonus',
                      'deferred_income',
                      'total_stock_value',
                      'expenses',
                      'exercised_stock_options',
                      'other',
                      'long_term_incentive',
                      'restricted_stock',
                      'director_fees']

email_features = ['to_messages',
                  'from_poi_to_this_person',
                  'from_messages',
                  'from_this_person_to_poi',
                  'shared_receipt_with_poi']

features_list = poi + financial_features + email_features

## loading the data set
data_dict = data_load_process()
#print data_dict

## Taking a closer look at the features
look_at_features(data_dict)

## Adding new features
features_list, data_dict = add_new_features(features_list, data_dict)
#print features_list

"""
# Manually removing features, according to the feature importance values of Decision Trees
# to improve performance

features_list = ['poi', 'salary', 'bonus', 'total_stock_value', 'from_poi_to_this_person',
 'from_this_person_to_poi', 'shared_receipt_with_poi', 'ratio_of_to_poi_messages',
  'salary_square', 'incentive_cube']
"""

"""
# Another feature list that gives better recall and precision, but worse accuracy
features_list = ['poi', 'ratio_of_to_poi_messages', 'ratio_of_from_poi_messages','bonus_salary_ratio', 'salary']
"""

## Store to my_data set for easy export below.
my_dataset = data_dict

## Format data to get a list
data = featureFormat(my_dataset, features_list, sort_keys = True)

## Extract features and labels from data set for local testing
labels, features = targetFeatureSplit(data)

## Splitting into train test sets
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

## Creating the classifier list
#clf_list = classifier_list()

"""
After selecting Decision Tree as the best classifier, after looking
at all the classifiers from the above classifier list, we make another
function with the same parameter values for the Decision Tree to further
tune the classifier. The below line is to be uncommented only after
testing for the other classifiers in the above function call of classifier_list()
"""
clf_list = best_classifier()

## Using Principal component analysis to scale down the number of features
clf_list = add_pca(clf_list)

## Using GridSearch to select best parameters for each classifier
clf_list = classify(clf_list, features_train, labels_train)

"""
Evaluate the precision, recall and f1 scores of the classifiers with the best parameters
The sort function is used to sort the classifiers after using evaluate() to get their
performances.
This is no longer being used, as this did not use k-fold cross validation and hence
gave more optimistic results.
NOTE: The following codes were replaced by call to the evaluate2() function
"""
"""
clf_list = evaluate(clf_list, features_test, labels_test)
clf_list_sorted = sort_clf(clf_list)
print clf_list_sorted, '\n'
clf =  clf_list_sorted[0][0]
print clf
"""

## Evaluating the classifiers using provided tester function in tester.py
evaluate2(clf_list, my_dataset, features_list)
clf = clf_list[0]

## Dump your classifier, data set, and features_list
dump_classifier_and_data(clf, my_dataset, features_list)
