import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def get_f1_score(param1_value, param2_value, param3_value, param4_value):
    df = pd.read_csv('data/heart.csv')

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(
        n_estimators=param1_value,
        max_depth=param2_value,
        max_features=param3_value,
        criterion=param4_value
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    return f1, X, y, X_train, X_test, y_train, y_test, model

