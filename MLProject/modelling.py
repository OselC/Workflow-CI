import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def train_model():
    # Load Data
    df = pd.read_csv("../MLProject/liver_patient_preprocessing.csv")

    # Split data
    X = df.drop(columns=["Selector"])
    y = df["Selector"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Autolog
    mlflow.autolog()

    # MLflow Tracking
    mlflow.set_experiment("XGBoost_Liver_Disease_Tracking")

    with mlflow.start_run():
        model = XGBClassifier()
        model.fit(x_train, y_train)

        # Evaluation
        model.score(x_test, y_test)

        mlflow.last_active_run()

        print("Training completed!")

if __name__ == "__main__":
    train_model()