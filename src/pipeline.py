import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def data_prep():
    car_data = fetch_ucirepo(id=19)
    df = car_data.data.original
    targets = car_data.data.targets

    # Encoding data
    category = [
        ['low', 'med', 'high', 'vhigh'],  # buying
        ['low', 'med', 'high', 'vhigh'],  # maint
        ['2', '3', '4', '5more'],         # doors
        ['2', '4', 'more'],               # persons
        ['small', 'med', 'big'],          # lug_boot
        ['low', 'med', 'high'],           # safety
        ['unacc', 'acc', 'good', 'vgood'] # target: evaluation
    ]

    encoder = OrdinalEncoder(categories=category).set_output(transform='pandas')

    df = encoder.fit_transform(df)

    # Split data
    # train: 60%, valid: 20%, test: 20%
    train, temp = train_test_split(df, train_size=0.6, shuffle=True, random_state=42, stratify=targets)
    valid, test = train_test_split(temp, train_size=0.5, shuffle=True, random_state=42, stratify=temp.iloc[:, -1])
    
    
    smote = SMOTE(random_state=42)
    
    X_train, y_train = smote.fit_resample(train.iloc[:, :-1], train.iloc[:, -1])
    X_valid, y_valid = valid.iloc[:, :-1], valid.iloc[:, -1]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

    return X_train, y_train, X_valid, y_valid, X_test, y_test
