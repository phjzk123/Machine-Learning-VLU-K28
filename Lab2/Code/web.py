import streamlit as st
import numpy as np
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Đọc dữ liệu
df = pd.read_csv("~/Documents/thuchanhhocmayungdung/Lab2/Data/Education.csv")
text, label = df['Text'], df['Label']

# Chia dữ liệu thành train/test
X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.2, random_state=42)

# Chuyển đổi dữ liệu văn bản thành đặc trưng số
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

X_train_vect = X_train_vect.toarray()
X_test_vect = X_test_vect.toarray()

# Khởi tạo hai mô hình Naive Bayes
bernoulli_model = BernoulliNB()
multinomial_model = MultinomialNB()

# Huấn luyện mô hình
bernoulli_model.fit(X_train_vect, y_train)
multinomial_model.fit(X_train_vect, y_train)

# Dự đoán với mô hình Bernoulli và Multinomial
y_pred_bernoulli = bernoulli_model.predict(X_test_vect)
y_pred_multinomial = multinomial_model.predict(X_test_vect)

# Lựa chọn mô hình
model_choice = st.radio("Chọn mô hình để phân tích tâm trạng:", ("BernoulliNB", "MultinomialNB"))

# Nhập dữ liệu từ người dùng
title = st.text_input("Tôi sẽ phân tích tâm trạng của bạn", "0")

# Dự đoán với mô hình đã chọn
user_input = vectorizer.transform([title])

if model_choice == "BernoulliNB":
    ans = bernoulli_model.predict(user_input)
else:
    ans = multinomial_model.predict(user_input)

# Xuất kết quả dự đoán
st.write("Tâm trạng của bạn là", "tốt" if ans == "positive" else "không tốt")
# Hiển thị độ chính xác và báo cáo phân loại
if st.button('Hiển thị đánh giá mô hình'):
    if model_choice == "BernoulliNB":
        accuracy = accuracy_score(y_test, y_pred_bernoulli)
        report = classification_report(y_test, y_pred_bernoulli)
    else:
        accuracy = accuracy_score(y_test, y_pred_multinomial)
        report = classification_report(y_test, y_pred_multinomial)
    
    st.write(f"Độ chính xác của mô hình {model_choice}: {accuracy}")
    st.text(f"Báo cáo phân loại cho mô hình {model_choice}:")
    st.text(report)
