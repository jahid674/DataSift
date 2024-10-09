import pandas as pd
import numpy as np
import us
from sklearn.model_selection import train_test_split
from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage, ACSMobility, ACSTravelTime, ACSHealthInsurance


def load_acs_data(dataset_name, year='2018', states=None, horizon='1-Year'):
    if states is None:
        states = [state.abbr for state in us.states.STATES]

    acs_data_source = ACSDataSource(survey_year=year, horizon=horizon, survey='person')
    data = acs_data_source.get_data(states=states, download=True)
    
    dataset_mapping = {
        'income': (ACSIncome, 'Income'),
        'employment': (ACSEmployment, 'Employment'),
        'public_coverage': (ACSPublicCoverage, 'PublicCoverage'),
        'mobility': (ACSMobility, 'Mobility'),
        'travel_time': (ACSTravelTime, 'TravelTime')
    }
    
    if dataset_name not in dataset_mapping:
        raise ValueError(f"Invalid dataset name. Choose from {', '.join(dataset_mapping.keys())}.")

    dataset_class, label_name = dataset_mapping[dataset_name]
    features, labels, _ = dataset_class.df_to_numpy(data)
    df = pd.DataFrame(features, columns=dataset_class.features)
    
    df[label_name] = labels
    df['RAC1P'] = df['RAC1P'].apply(lambda x: 0 if x == 2 else 1)
    df['SEX'] = df['SEX'].apply(lambda x: 1 if x == 1 else 0)
    df[label_name] = df[label_name].apply(lambda x: 1 if x else 0)
    test_data, train_data = train_test_split(df, test_size=0.8, random_state=42)
    return train_data, test_data

def get_target_sensitive_attribute(dataset_name):
    attributes = {
        'income': ('Income', 'SEX', 0.0, 1.0),
        'employment': ('Employment', 'RAC1P', 0.0, 1.0),
        'public_coverage': ('PublicCoverage', 'SEX', 0.0, 1.0),
        'mobility': ('Mobility', 'RAC1P', 0.0, 1.0),
        'travel_time': ('TravelTime', 'RAC1P', 0.0, 1.0),
        'adult': ('income', 'gender', 0.0, 1.0),
        'credit': ('SeriousDlqin2yrs', 'age', 0.0, 1.0)
    }
    
    if dataset_name not in attributes:
        raise ValueError(f"Invalid dataset name. Choose from {', '.join(attributes.keys())}.")

    Target_attribute, Sensitive_attribute, protected_group, privileged_group = attributes[dataset_name]
    return Target_attribute, Sensitive_attribute, protected_group, privileged_group


def load_adult(sample=False):
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital', 'occupation',\
            'relationship', 'race', 'gender', 'capgain', 'caploss', 'hours', 'country', 'income']
    if sample:
        df_train = pd.read_csv('adult-sample-train-10pc', names=cols, sep=",")
        df_test = pd.read_csv('adult-sample-test-10pc', names=cols, sep=",")
    else:
        df_train = pd.read_csv('adult.data', names=cols, sep=",")
        df_test = pd.read_csv('adult.test', names=cols, sep=",")

    df_train = process_adult(df_train)
    df_test = process_adult(df_test)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    '''X_train = df_train.drop(columns='income')
    y_train = df_train['income']

    X_test = df_test.drop(columns='income')
    y_test = df_test['income']'''
    return df_train, df_test

def process_adult(df):
    # replace missing values (?) to nan and then drop the columns
    df['country'] = df['country'].replace('?',np.nan)
    df['workclass'] = df['workclass'].replace('?',np.nan)
    df['occupation'] = df['occupation'].replace('?',np.nan)
    # dropping the NaN rows now
    df.dropna(how='any',inplace=True)
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
    df['age'] = df['age'].apply(lambda x : 1 if x >= 45 else 0) # 1 if old, 0 if young
    df['workclass'] = df['workclass'].map({'Never-worked': 0, 'Without-pay': 1, 'State-gov': 2, 'Local-gov': 3, 'Federal-gov': 4, 'Self-emp-inc': 5, 'Self-emp-not-inc': 6, 'Private': 7}).astype(int)
    df['education'] = df['education'].map({'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6, '12th': 7, 'HS-grad':8, 'Some-college': 9, 'Bachelors': 10, 'Prof-school': 11, 'Assoc-acdm': 12, 'Assoc-voc': 13, 'Masters': 14, 'Doctorate': 15}).astype(int)
    df['marital'] = df['marital'].map({'Married-civ-spouse': 2, 'Divorced': 1, 'Never-married': 0, 'Separated': 1, 'Widowed': 1, 'Married-spouse-absent': 2, 'Married-AF-spouse': 2}).astype(int)
    df['relationship'] = df['relationship'].map({'Wife': 1 , 'Own-child': 0 , 'Husband': 1, 'Not-in-family': 0, 'Other-relative': 0, 'Unmarried': 0}).astype(int)
    df['race'] = df['race'].map({'White': 1, 'Asian-Pac-Islander': 0, 'Amer-Indian-Eskimo': 0, 'Other': 0, 'Black': 0}).astype(int)
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0}).astype(int)
    # process hours
    df.loc[(df['hours'] <= 40), 'hours'] = 0
    df.loc[(df['hours'] > 40), 'hours'] = 1
    df = df.drop(columns=['fnlwgt', 'education.num', 'occupation', 'country', 'capgain', 'caploss'])
    df = df.reset_index(drop=True)
    return df


def preprocess_credit(df, preprocess=True):
    # Fill missing values
    df['MonthlyIncome'].fillna(df['MonthlyIncome'].mean(), inplace=True)
    df['NumberOfDependents'].fillna(df['NumberOfDependents'].median(), inplace=True)
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower_bound, upper_bound)
    
    remove_outliers(df, 'MonthlyIncome')
    remove_outliers(df, 'RevolvingUtilizationOfUnsecuredLines')
    remove_outliers(df, 'DebtRatio')
    
    df['age'] = df['age'].apply(lambda x : 1 if x >= 35 else 0)   
    return df

def load_credit(preprocess=True):
    train = pd.read_csv("cs-training.csv")
    train_data = train.drop("Unnamed: 0", axis=1)
    train_data = preprocess_credit(train_data, preprocess=True)   
    train_data = train_data.reset_index(drop=True)
    test_data, train_data = train_test_split(train_data, test_size=0.8, random_state=42)
    return train_data, test_data




def load_data(dataset_name):
    if dataset_name == 'adult':
        return load_adult(sample=False)
    elif dataset_name == 'credit':
        return load_credit(preprocess = True)
    elif dataset_name in ['income', 'employment', 'public_coverage', 'mobility', 'travel_time']:
        return load_acs_data(dataset_name, year='2018', states=['CA'], horizon='1-Year')
    else:
        raise ValueError("Dataset not supported. Choose from 'adult', 'german', or ACS datasets.")

