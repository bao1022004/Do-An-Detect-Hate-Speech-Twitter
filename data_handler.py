import pandas as pd
import numpy as np
import os

def get_data():
    tweets = []
    # Đường dẫn file CSV thực tế của bạn
    file_path = './tweet_data/dataset.csv'
    
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file tại {file_path}")
        return []

    try:
        # Đọc file với encoding utf-8
        df = pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return []
    
    # Duyệt qua từng dòng dữ liệu dựa trên tên cột thực tế: 'tweet' và 'private'
    for index, row in df.iterrows():
        # 1. Lấy nội dung từ cột 'tweet' (thay vì 'text')
        content = str(row['tweet']).lower() if pd.notnull(row['tweet']) else ""
        
        # 2. Lấy nhãn từ cột 'private' (1 là riêng tư, 0 là không)
        label_id = int(row['private']) if pd.notnull(row['private']) else 0
        
        # 3. Vì file không có cột topic, ta để mặc định là 'unknown'
        topic_name = "unknown"
            
        tweets.append({
            'text': content,
            'label': label_id,
            'topic': topic_name
        })

    print(f"--- THÔNG TIN DATASET ---")
    print(f"Tổng số dòng tải được: {len(tweets)}")
    
    # Thống kê nhãn (0: Non-private, 1: Private)
    if tweets:
        df_stats = pd.DataFrame(tweets)
        print("Thống kê nhãn (0: Non-private, 1: Private):")
        print(df_stats['label'].value_counts())
        
    return tweets

if __name__=="__main__":
    tweets = get_data()
    if tweets:
        print("\nVí dụ dòng dữ liệu đầu tiên:")
        print(f"Nội dung: {tweets[0]['text'][:70]}...")
        print(f"Nhãn: {tweets[0]['label']} (1 là Private)")