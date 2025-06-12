import pandas as pd
import random

# 读取 CSV 文件
file_path = './uploads/7007_lkj_202410_B车.csv'
df = pd.read_csv(file_path)

# 随机生成 0 或 1 的标签列
labels = [random.randint(0, 1) for _ in range(len(df))]

# 在 DataFrame 中添加标签列
df['label'] = labels

# 保存修改后的 DataFrame 到新的 CSV 文件
new_file_path = '7007_lkj_202410_B车_with_label.csv'
df.to_csv(new_file_path, index=False)

print(f"已添加标签列并保存到 {new_file_path}")
