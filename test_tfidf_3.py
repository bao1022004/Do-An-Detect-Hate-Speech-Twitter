import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score
from data_handler import get_data
import re

def clean_text(text):
    # Loại bỏ link, icon và ký tự đặc biệt
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.lower()
    return text

def run_tfidf_test():
    # 1. Lấy dữ liệu từ data_handler
    print("Đang tải dữ liệu...")
    raw_data = get_data()
    
    texts = [clean_text(d['text']) for d in raw_data]
    labels = [d['label'] for d in raw_data]
    
    X = np.array(texts)
    y = np.array(labels)

    # 2. Thiết lập TF-IDF Vectorizer với ngram (1, 3)
    # Tăng max_features lên một chút vì cụm 3 từ sẽ tạo ra rất nhiều đặc trưng mới
    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, stop_words='english')

    # 3. Sử dụng K-Fold để đánh giá
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    fold = 1
    accuracies = []

    print("\n--- Bắt đầu huấn luyện TF-IDF (1,3-grams) + Logistic Regression ---")

    for train_index, test_index in kf.split(X):
        X_train_raw, X_test_raw = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Chuyển đổi văn bản
        X_train_tfidf = tfidf.fit_transform(X_train_raw)
        X_test_tfidf = tfidf.transform(X_test_raw)

        # Sử dụng Logistic Regression với class_weight='balanced' để xử lý mất cân bằng nhãn
        model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
        model.fit(X_train_tfidf, y_train)

        # Dự đoán
        y_pred = model.predict(X_test_tfidf)
        
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        print(f"Fold {fold} Accuracy: {acc:.4f}")
        if fold == 1:
             print(classification_report(y_test, y_pred))
        fold += 1

    print("\n--- KẾT QUẢ CUỐI CÙNG ---")
    print(f"Độ chính xác trung bình (Mean Accuracy): {np.mean(accuracies):.4f}")

if __name__ == "__main__":
    run_tfidf_test()