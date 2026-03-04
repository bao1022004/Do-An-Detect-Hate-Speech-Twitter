import pandas as pd
import ast
import os

# Đọc file dữ liệu của bạn
df = pd.read_excel("tweetdataa.xlsx")

# Chuyển đổi cột labels từ chuỗi thành list thực sự
df['labels'] = df['labels'].apply(ast.literal_eval)

# Lấy nhãn đầu tiên làm nhãn chính (hoặc 'none' nếu danh sách rỗng)
def get_primary_label(label_list):
    return label_list[0] if len(label_list) > 0 else "none"

df['label'] = df['labels'].apply(get_primary_label)

# Lưu vào thư mục dự án
if not os.path.exists('tweet_data'): os.makedirs('tweet_data')
df[['text', 'label']].to_csv('tweet_data/data_new.csv', index=False)

# In ra danh sách các nhãn mới để kiểm tra
print("Các nhãn mới của bạn:", df['label'].unique())