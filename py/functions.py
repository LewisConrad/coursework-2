import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# loading the csv file
def load_missions_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)




def preprocess_missions(df: pd.DataFrame):

    #drop irrelevant columns
    df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], errors="ignore")

    #converting to binary ( success = 1, anything else = 0)
    df["y"] = (df["Mission_Status"].astype(str).str.strip().str.lower() == "success").astype(int)

    #converting date and time to just the launch year
    df["LaunchYear"] = pd.to_datetime(df["Date"], errors="coerce").dt.year

    #cleaning prices
    df["Price"] = (
        df["Price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    #setting features and making copies
    X = df[["Organisation", "Location", "Rocket_Status", "Price", "LaunchYear"]].copy()
    y = df["y"].copy()

    #joining x and y together and dropping rows that are missing essential data
    data = pd.concat([X, y.rename("y")], axis=1).dropna(subset=["LaunchYear", "Price", "y"])
    X = data.drop(columns=["y"])
    y = data["y"]

    #one-hot encoding
    X_encoded = pd.get_dummies(X, columns=["Organisation", "Location", "Rocket_Status"], drop_first=True)

    #drop any remaining NaNs 
    data2 = pd.concat([X_encoded, y.rename("y")], axis=1).dropna()
    X_clean = data2.drop(columns=["y"])
    y_clean = data2["y"]

    return X_clean, y_clean


def train_test(
    X_clean: pd.DataFrame,
    y_clean: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
    ):

    #train and test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y_clean,
        test_size=0.2,
        random_state=42,
        stratify=y_clean
        )

    #fitting scaler on training data and transforming training and test data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
