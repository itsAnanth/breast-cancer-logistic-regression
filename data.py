from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def getData():
    data = load_breast_cancer()
    X, y, features, labels = data.data, data.target, data.feature_names, data.target_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # print(X_train_scaled)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, features, labels