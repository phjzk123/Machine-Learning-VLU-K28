import streamlit as st
import numpy as np
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv("~/Documents/thuchanhhocmayungdung/Lab2/Data/Education.csv")
text, label = df['Text'], df['Label']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.2, random_state=42)

# Chuyển đổi dữ liệu văn bản thành đặc trưng số sử dụng TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Chuyển đổi dữ liệu sang mảng
X_train_vect = X_train_vect.toarray()
X_test_vect = X_test_vect.toarray()

# Khởi tạo hai mô hình Naive Bayes
bernoulli_model = BernoulliNB()
multinomial_model = MultinomialNB()

# Huấn luyện mô hình
bernoulli_model.fit(X_train_vect, y_train)
multinomial_model.fit(X_train_vect, y_train)

# Nhập dữ liệu từ người dùng
title = st.text_input("Tôi sẽ phân tích tâm trạng của bạn", "")

# Lựa chọn mô hình
model_choice = st.radio("Chọn mô hình để phân tích tâm trạng:", ("BernoulliNB", "MultinomialNB"))

# Dự đoán với mô hình đã chọn khi người dùng nhập dữ liệu
if title:
    user_input = vectorizer.transform([title]).toarray()  # Chuyển đổi dữ liệu nhập của người dùng

    if model_choice == "BernoulliNB":
        ans = bernoulli_model.predict(user_input)
    else:
        ans = multinomial_model.predict(user_input)

    # Xuất kết quả dự đoán
    st.write("Tâm trạng của bạn là:", "tốt" if ans[0] == "positive" else "không tốt")

# Hiển thị đánh giá mô hình khi người dùng nhấn nút
if st.button('Hiển thị đánh giá mô hình'):
    if model_choice == "BernoulliNB":
        y_pred = bernoulli_model.predict(X_test_vect)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
    else:
        y_pred = multinomial_model.predict(X_test_vect)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
    
    st.write(f"Độ chính xác của mô hình {model_choice}: {accuracy:.2f}")
    st.text(f"Báo cáo phân loại cho mô hình {model_choice}:")
    st.text(report)

# Hiển thị tên của bạn ở góc sidebar
st.sidebar.markdown("Nguyễn Văn Phú - 2274802010661")
