import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

# Đọc dữ liệu
df = pd.read_csv("~/Documents/thuchanhhocmayungdung/lab3 (1)/Education.csv")
text, label = df['Text'], df['Label']

# Chia dữ liệu thành train/test
X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.2, random_state=42)

# Chuyển đổi dữ liệu văn bản thành đặc trưng số
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Khởi tạo mô hình KNN
knn_model = KNeighborsClassifier(n_neighbors=3)  # Bạn có thể thay đổi số lượng k tại đây

# Huấn luyện mô hình KNN
knn_model.fit(X_train_vect, y_train)

# Nhập dữ liệu từ người dùng
user_input = st.text_input("Tôi sẽ phân tích tâm trạng của bạn", "")

# Dự đoán với mô hình KNN
if user_input:
    user_input_vect = vectorizer.transform([user_input])
    ans = knn_model.predict(user_input_vect)

    # Xuất kết quả dự đoán
    if ans[0] == "positive":
        st.write("Tâm trạng của bạn là tốt")
    else:
        st.write("Tâm trạng của bạn là không tốt")

# Hiển thị tên của bạn ở góc
st.sidebar.markdown("Nguyễn Văn Phú - 2274802010661")
