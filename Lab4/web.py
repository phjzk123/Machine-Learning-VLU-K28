import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTreeClass
from RandomForest import RandomForest

# Hàm tính độ chính xác
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true) * 100

# Đọc dữ liệu
file_path = 'drug200.csv'
data = pd.read_csv(file_path)

# Biến đổi dữ liệu định tính sang định lượng
data['Sex'] = data['Sex'].map({'M': 0, 'F': 1})
data['BP'] = data['BP'].map({'HIGH': 2, 'NORMAL': 1, 'LOW': 0})
data['Cholesterol'] = data['Cholesterol'].map({'HIGH': 1, 'NORMAL': 0})
data['Drug'] = data['Drug'].map({'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'DrugY': 4})

# Tạo tập X và y
X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = data['Drug']

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Giao diện Streamlit
st.title("Drug Classification with RandomForest and DecisionTree")

# Chọn mô hình
model_choice = st.selectbox("Choose Model", ("Decision Tree", "Random Forest"))

# Nhập dữ liệu từ người dùng
age = st.slider("Age", min_value=0, max_value=100, value=50)
sex = st.selectbox("Sex", ["M", "F"])
bp = st.selectbox("Blood Pressure (BP)", ["LOW", "NORMAL", "HIGH"])
cholesterol = st.selectbox("Cholesterol", ["NORMAL", "HIGH"])
na_to_k = st.slider("Na_to_K ratio", min_value=5.0, max_value=40.0, value=10.0)

# Chuyển đổi input từ người dùng thành dạng số
sex = 0 if sex == "M" else 1
bp = {"LOW": 0, "NORMAL": 1, "HIGH": 2}[bp]
cholesterol = 0 if cholesterol == "NORMAL" else 1

# Tạo dataframe từ dữ liệu nhập vào
input_data = pd.DataFrame([[age, sex, bp, cholesterol, na_to_k]], columns=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])

# Hiển thị dữ liệu người dùng nhập
st.write("### Input Data:")
st.write(input_data)

# Chạy mô hình
if model_choice == "Decision Tree":
    model = DecisionTreeClass(min_samples_split=2, max_depth=10)
    model.fit(X_train, y_train)
elif model_choice == "Random Forest":
    model = RandomForest(n_trees=3, n_features=4)
    model.fit(X_train, y_train)

# Dự đoán
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"### Predicted Drug: {prediction[0]}")

# Hiển thị độ chính xác của mô hình cho cả hai mô hình
# Huấn luyện Decision Tree
decision_tree_model = DecisionTreeClass(min_samples_split=2, max_depth=10)
decision_tree_model.fit(X_train, y_train)
y_pred_dt = decision_tree_model.predict(X_test)
decision_tree_accuracy = accuracy(y_test.values, y_pred_dt)

# Huấn luyện Random Forest
random_forest_model = RandomForest(n_trees=3, n_features=4)
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)
random_forest_accuracy = accuracy(y_test.values, y_pred_rf)

# Hiển thị độ chính xác
st.write(f"### Decision Tree Accuracy: {decision_tree_accuracy:.2f}%")
st.write(f"### Random Forest Accuracy: {random_forest_accuracy:.2f}%")
