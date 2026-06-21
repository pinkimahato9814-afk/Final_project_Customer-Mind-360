import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("cleaned_dataset.csv")


# Data Cleaning
if df["Churned"].dtype == "object":
    df["Churned"] = df["Churned"].map({
        "Yes": 1,
        "No": 0,
        "Churned": 1,
        "Not Churned": 0
    })

df = df.dropna()


def predict_churn():
    numerical_features = [
        "Customer_Service_Calls",
        "Cart_Abandonment_Rate",
        "Pages_Per_Session",
        "Session_Duration_Avg",
        "Email_Open_Rate",
        "Mobile_App_Usage",
        "Login_Frequency",
        "Total_Purchases",
        "Days_Since_Last_Purchase"
    ]

    categorical_features = [
        "Signup_Quarter"
    ]

    features = numerical_features + categorical_features
    target = "Churned"

    X = df[features]
    Y = df[target]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("knn", KNeighborsClassifier(n_neighbors=7))
        ]
    )

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    cr = classification_report(Y_test, Y_pred, output_dict=True)
    cm = confusion_matrix(Y_test, Y_pred)

    return features, target, model, Y_pred, cr, cm