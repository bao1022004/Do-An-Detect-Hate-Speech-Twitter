import pandas as pd
import os
import shutil

def fix_and_convert():
    excel_file = './tweet_data/tweetdataa.xlsx'
    output_path = './tweet_data/dataset.csv'

    # Nếu lỡ tay tạo thư mục tên là dataset.csv, hãy xóa nó đi
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)

    if not os.path.exists(excel_file):
        print(f"Không tìm thấy {excel_file}")
        return

    df = pd.read_excel(excel_file)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Đã tạo file chuẩn tại: {output_path}")

fix_and_convert()