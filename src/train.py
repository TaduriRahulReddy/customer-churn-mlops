

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load data
df = pd.read_csv("data/Customer-Churn.csv")

# drop ID column
df = df.drop("customerID", axis=1)

# encode target
df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

# convert categorical features
df = pd.get_dummies(df)



X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# start mlflow experiment
mlflow.set_experiment("customer_churn")

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    # log parameters
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)

    # log metric
    mlflow.log_metric("accuracy", accuracy)

    # log model
    mlflow.sklearn.log_model(model, "model")

# save model locally
joblib.dump(model, "models/churn_model.pkl")

print("Model trained and saved.")