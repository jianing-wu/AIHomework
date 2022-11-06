import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing


def run_data_processing(infile='data/survey.csv', outfile='data/survey.npz'):
    """
    Runs data processing on the mental health survey data

    infile: a filepath to the csv file to load data from
    outfile: a filepath to store the npz file.

    This function must store the output as a numpy npz collection with two files `X` and `y`.
    X is a size N x d array where N is the number of datapoints and d is the feature dimension
    y is a size N array containing just the labels for each row.

    The y value must come from the 'treatment' column.


    """
    # reading in CSV's from a file path
    train_df = pd.read_csv(infile)

    # dealing with missing data
    # Let’s get rid of the variables "Timestamp",“comments”, “state” just to make our lives easier.
    train_df = train_df.drop(['comments'], axis=1)
    train_df = train_df.drop(['state'], axis=1)
    train_df = train_df.drop(['Timestamp'], axis=1)

    defaultInt = 0
    defaultString = 'NaN'
    defaultFloat = 0.0

    # Create lists by data tpe
    intFeatures = ['Age']
    stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                     'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                     'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                     'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                     'seek_help']
    floatFeatures = []

    # Clean the NaN's
    for feature in train_df:
        if feature in intFeatures:
            train_df[feature] = train_df[feature].fillna(defaultInt)
        elif feature in stringFeatures:
            train_df[feature] = train_df[feature].fillna(defaultString)
        elif feature in floatFeatures:
            train_df[feature] = train_df[feature].fillna(defaultFloat)
        else:
            print('Error: Feature %s not recognized.' % feature)

    train_df['Gender'] = train_df['Gender'].str.lower()

    # Made gender groups # What fairness considerations should you make here?
    male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
    female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

    for (row, col) in train_df.iterrows():

        if str.lower(col.Gender) in male_str:
            train_df['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

        elif str.lower(col.Gender) in female_str:
            train_df['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

        else:
            train_df['Gender'].replace(to_replace=col.Gender, value='nonbinary_or_not_listed', inplace=True)

    # complete missing age with mean
    train_df['Age'].fillna(train_df['Age'].median(), inplace = True)

    # Fill with media() values < 18 and > 120
    s = pd.Series(train_df['Age'])
    s[s<18] = train_df['Age'].median()
    train_df['Age'] = s
    s = pd.Series(train_df['Age'])
    s[s>120] = train_df['Age'].median()
    train_df['Age'] = s

    # Ranges of Age
    train_df['age_range'] = pd.cut(train_df['Age'], [0,20,30,65,100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)


    # There are only 0.014% of self employed so let's change NaN to NOT self_employed
    # Replace "NaN" string from defaultString
    train_df['self_employed'] = train_df['self_employed'].replace([defaultString], 'No')

    # There are only 0.20% of self work_interfere so let's change NaN to "Don't know
    # Replace "NaN" string from defaultString
    train_df['work_interfere'] = train_df['work_interfere'].replace([defaultString], 'Don\'t know')

    labelDict = {}
    for feature in train_df:
        le = preprocessing.LabelEncoder()
        le.fit(train_df[feature])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        train_df[feature] = le.transform(train_df[feature])
        # Get labels
        labelKey = 'label_' + feature
        labelValue = [*le_name_mapping]
        labelDict[labelKey] = labelValue

    for key, value in labelDict.items():
        print(key, value)

    # Get rid of 'Country'
    train_df = train_df.drop(['Country'], axis=1)

    # Scaling Age
    scaler = preprocessing.MinMaxScaler()
    train_df['Age'] = scaler.fit_transform(train_df[['Age']])

    # define X and y
    feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
    X = train_df[feature_cols]

    X_feats = np.array(X)

    # DO NOT CHANGE ANY LINES BELOW THIS
    y = train_df.treatment
    y_feats = np.array(y)
    np.savez(outfile, x=X_feats, y=y_feats.astype('float').reshape(-1,1))

def main():
    run_data_processing()

if __name__ == '__main__':
    main()
