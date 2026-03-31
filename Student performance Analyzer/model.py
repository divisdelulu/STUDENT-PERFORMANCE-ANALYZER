import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

def train_models():
    data = pd.read_csv("dataset.csv")

    X = data[['hours', 'attendance', 'previous_marks']]
    y_reg = data['final_marks']

    # Classification target
    y_class = [1 if i >= 50 else 0 for i in y_reg]

    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2)

    # Regression Model
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

    # Classification Model
    clf_model = LogisticRegression()
    clf_model.fit(X_train, y_class[:len(X_train)])

    return reg_model, clf_model