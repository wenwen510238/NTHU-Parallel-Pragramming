import pandas as pd
import matplotlib.pyplot as plt

# 模擬讀取資料
data = {
    'Process ID': [4056662] * 8,
    'Thread ID': [4056676, 4056675, 4056677, 4056680, 4056673, 4056678, 4056674, 4056679],
    'Name': ['hw2a'] * 8,
    # 'CPU utilization': ['14.38%', '12.70%', '12.62%', '12.58%', '12.58%', '12.44%', '11.30%', '11.30%']
    'CPU utilization': ['12.49%', '12.49%', '12.48%', '12.47%', '12.47%', '12.47%', '11.44%', '11.23%']
}

df = pd.DataFrame(data)

# 去掉 % 符號並將 'CPU utilization' 轉換為浮點數
df['CPU utilization'] = df['CPU utilization'].str.replace('%', '').astype(float)

# 使用序號 ID 作為 x 軸
x_values = range(1, len(df) + 1)

# 繪製折線圖
plt.figure(figsize=(10, 6))
plt.plot(x_values, df['CPU utilization'], marker='o', linestyle='-', color='b')
plt.xlabel('ID of thread')
plt.ylabel('CPU Utilization (%)')
# plt.title('Load Balancing, pthread, #slow_01')
plt.title('Load Balancing, pthread, #strict_34')
plt.xticks(x_values)  # 設定 x 軸為 ID 1 到 8
plt.ylim(0, 20)
plt.grid(True)
# output_image = 'load_balancing_01.png'
output_image = 'load_balancing_34.png'
plt.savefig(output_image)
