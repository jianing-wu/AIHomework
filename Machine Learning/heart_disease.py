import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics


def read_data(path='data/heart_2020_cleaned.csv', split=slice(50000)):
    # DO NOT CHANGE THIS CODE
    df = pd.read_csv(path)
    to_replace_dict = {
        'No, borderline diabetes': '0',
        'Yes (during pregnancy)': '1'
    }
    df = df[df.columns].replace(to_replace_dict)
    df['HeartDisease'] = df['HeartDisease'].replace({'Yes': 1, 'No': 0})
    df = df[split].reset_index(drop=True)  # Only use first 50000 datapoints to make script run faster
    return df


def split_data(df):
    # DO NOT CHANGE THIS CODE
    features = df.drop(columns=['HeartDisease'], axis=1)
    target = df['HeartDisease']
    X_train, X_val, y_train, y_val = train_test_split(features, target, shuffle=True, test_size=.1, random_state=42)

    return X_train, X_val, y_train, y_val


def evaluate_model(model, x_test, y_test):
    # DO NOT CHANGE THIS CODE
    # Predict Test Data
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(x_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}


def print_eval_results(res, model_name=""):
    # DO NOT CHANGE THIS CODE
    if model_name:
        print(f"\nResults for {model_name} are:")
    else:
        print("\nResults are:")
    print(f"Accuracy: {res['acc']}")
    print(f"Precision: {res['prec']}")
    print(f"Recall: {res['rec']}")
    print(f"F1: {res['f1']}")
    print(f"AUC: {res['auc']}")
    print("\n")


def process_data(df):
    """
    Accepts a data frame containing the data to use
    Returns a processed dataframe.
    The returned dataframe must contain the column 'HeartDisease' as originally in the data.
    """
    "*** Optional Code CHANGES Here ***"

    # Make binary variables binary numbers
    to_replace_dict = {
        'Yes': 1,
        'No': 0,
        'Male': 1,
        'Female': 0,
    }
    df = df[df.columns].replace(to_replace_dict)
    df['Diabetic'] = df['Diabetic'].astype(int)

    # Standardize numerical distributions
    numerical_cols = ['MentalHealth', 'BMI', 'PhysicalHealth', 'SleepTime']
    Scaler = StandardScaler()
    df[numerical_cols] = Scaler.fit_transform(df[numerical_cols])

    # Encode categorical data as one hot encodings
    enc = OneHotEncoder()
    # Encoding categorical features
    categorical_cols = df[['AgeCategory', 'Race', 'GenHealth']]
    encoded_cat_cols = pd.DataFrame(enc.fit_transform(categorical_cols).toarray())
    # Linking the encoded columns with the df
    df = pd.concat([df, encoded_cat_cols], axis=1)

    # Dropping the original categorical features
    df = df.drop(columns=['AgeCategory', 'Race', 'GenHealth'], axis=1)

    return df


def run_and_eval_knn(X_train, X_val, y_train, y_val):
    """Runs KNN Model"""
    "*** Optional Code Changes Here ***"
    knn = KNeighborsClassifier(n_neighbors=1)
    print("Fitting KNN Classifier...")
    knn.fit(X_train, y_train)
    print("Done")
    knn_eval = evaluate_model(knn, X_val, y_val)
    print_eval_results(knn_eval, model_name="KNN")
    return knn


def run_and_eval_decision_tree(X_train, X_val, y_train, y_val):
    """Runs Decision Tree Model"""
    "*** Optional Code Changes Here ***"
    clf = tree.DecisionTreeClassifier(random_state=0)
    print("Fitting Decision Tree Classifier...")
    clf.fit(X_train, y_train)
    print("Done")

    # Evaluate Model
    clf_eval = evaluate_model(clf, X_val, y_val)
    print_eval_results(clf_eval, model_name="Decision Tree")
    return clf


def run_and_eval_logistic(X_train, X_val, y_train, y_val):
    """Runs Logistic Regression Model"""
    "*** Optional Code Changes Here ***"
    clf = LogisticRegression(random_state=0, max_iter=500, verbose=False)
    print("Fitting Logistic Regression...")
    clf.fit(X_train, y_train)
    print("Done")

    # Evaluate Model
    clf_eval = evaluate_model(clf, X_val, y_val)
    print_eval_results(clf_eval, model_name="Logistic Regression")
    return clf


def run_and_eval_neural_network(X_train, X_val, y_train, y_val):
    """Runs Logistic Regression Model"""
    "*** Optional Code Changes Here ***"
    clf = MLPClassifier(random_state=0,
                        hidden_layer_sizes=(10,),
                        activation="relu",
                        solver='sgd',
                        learning_rate='constant',
                        learning_rate_init=0.01,
                        batch_size=200,
                        momentum=.9,
                        max_iter=100,  # This is equivalent to num epochs
                        verbose=False)
    print("Fitting Neural Netwok...")
    clf.fit(X_train, y_train)
    print("Done")

    # Evaluate Model
    clf_eval = evaluate_model(clf, X_val, y_val)
    print_eval_results(clf_eval, model_name="Neural Network")
    return clf


def run_student_model(X_train, X_val, y_train, y_val, X_test=None):
    """
    This is where you develop and train your sklearn model

    If X_test is not None, this should return the predictions for those data

    This question is open ended and lets you decide how to move forward.

    See the examples above for ideas to get started.

    Areas to think about:
        dealing with unbalanced data
        better training of models
        better processing of data
        different models
        choosing the hyperparameters (Maybe look into sklearn.model_selection.GridSearchCV)
        different ways of optimizing models

    """
    # "*** Make Changes Here ***"

    # params = {
    #     'hidden_layer_sizes': [(10,10),(20,20)],
    #     'activation': ['logistic'],
    #     'solver': ['sgd'],
    #     'learning_rate': ['constant'],
    #     'learning_rate_init': [0.01, 0.05],
    #     'batch_size': [100, 200],
    #     'max_iter': [100, 200]
    # }
    model = MLPClassifier(random_state=0,
                        hidden_layer_sizes=(20,5),
                        activation="relu",
                        solver='sgd',
                        learning_rate='constant',
                        learning_rate_init=0.01,
                        batch_size=200,
                        momentum=.9,
                        max_iter=100,  # This is equivalent to num epochs
                        verbose=False)

    # clf = GridSearchCV(model, params)
    # clf.fit(X_train, y_train)
    model.fit(X_train, y_train)
    eval_results = evaluate_model(model, X_val, y_val)
    print_eval_results(eval_results, model_name="Student Model")

    if X_test is not None:
        # y_pred = model.predict(X_test)

        # Possibly you may want to use the predicted probabilites and an alternative threshold (default is 0.5)
        # y_probs = model.predict_proba(X_test)
        y_pred = (model.predict_proba(X_test)[:,1] >= 0.2).astype(bool)

        return y_pred


def main():
    data = read_data()
    data = process_data(data)
    X_train, X_val, y_train, y_val = split_data(data)
    run_and_eval_knn(X_train, X_val, y_train, y_val)
    run_and_eval_decision_tree(X_train, X_val, y_train, y_val)
    run_and_eval_logistic(X_train, X_val, y_train, y_val)
    run_and_eval_neural_network(X_train, X_val, y_train, y_val)

    # run_student_model(X_train, X_val, y_train, y_val)


if __name__ == '__main__':
    main()



