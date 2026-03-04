import pandas as pd
import numpy as np
import os
import codecs

# Định nghĩa bảng nhãn mới khớp với prepare_data.py của bạn
task1_label_dict = {
    'none': 0,
    'profanity': 1,
    'conflictual': 2,
    'sec': 3,
    'drugs': 4,
    'spam': 5,
    'self-harm': 6
}

def get_data():
    tweets = []
    # Đường dẫn tới file data_new.csv mà prepare_data.py vừa tạo ra
    file_path = './tweet_data/data_new.csv'
    
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file {file_path}. Hãy chạy prepare_data.py trước!")
        return []

    # Đọc dữ liệu bằng pandas
    df = pd.read_csv(file_path)
    
    for index, row in df.iterrows():
        # Kiểm tra nếu dòng text bị trống (NaN)
        text = str(row['text']).lower() if pd.notnull(row['text']) else ""
        label_text = row['label']
        
        # Chuyển nhãn chữ sang số dựa trên task1_label_dict
        if label_text in task1_label_dict:
            label_id = task1_label_dict[label_text]
        else:
            label_id = 0 # Mặc định là 'none' nếu nhãn lạ
            
        tweets.append({
            'text': text,
            'label': label_id
        })

    print(f"Đã tải thành công {len(tweets)} dòng dữ liệu với {len(task1_label_dict)} nhãn.")
    return tweets

if __name__=="__main__":
    tweets = get_data()
    if tweets:
        print("Dòng dữ liệu đầu tiên mẫu:", tweets[0])