import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
import dagshub
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse

def train_model(n_estimators, learning_rate):
    dagshub.init(repo_owner='OselC', repo_name='Liver-Disease-Detection', mlflow=True)

    # Load Data
    df = pd.read_csv("liver_patient_preprocessing.csv")

    # Split data
    X = df.drop(columns=["Selector"])
    y = df["Selector"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Autolog
    mlflow.autolog()

    # MLflow Tracking
    mlflow.set_experiment("XGBoost_Liver_Disease_Tracking")

    with mlflow.start_run():
        model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        model.fit(x_train, y_train)

        # Evaluation
        model.score(x_test, y_test)

        # Confusion Matrix
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # Feature Importance
        plt.figure(figsize=(10,6))
        pd.Series(model.feature_importances_, index=X.columns).nlargest(10).plot(kind='barh')
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")

        mlflow.last_active_run()

        print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()

    train_model(args.n_estimators, args.learning_rate)