import pandas as pd

# 路径根据你的位置调整（从截图看，在 assignments/ 上层？）
df = pd.read_csv('data.csv')  # 如果报错，改成绝对路径或复制到当前目录

print("总行数:", len(df))
print("列名:", df.columns.tolist())
print("\n前5行示例:")
print(df.head())

# 如果有图像路径列，打印一些路径示例
# 常见列名可能是 'path', 'filename', 'image_path', 'id' 等
# 假设有 'filename' 列，打印前10个
if 'filename' in df.columns:
    print("\n文件名示例:")
    print(df['filename'].head(10).tolist())
elif 'path' in df.columns:
    print(df['path'].head(10).tolist())