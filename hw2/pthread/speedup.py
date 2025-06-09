import os
import pandas as pd
import matplotlib.pyplot as plt

# 資料夾路徑與檔案名稱
# input_folder = 'avgs'  # 替換為你的資料夾路徑
# file_names = ['average_01.csv', 'average_04.csv', 'average_08.csv', 'average_12.csv']  # 替換為你的檔名
# input_folder = 'pthread_34_csv'
# file_names = ['pthread_strict_34_1.csv', 'pthread_strict_34_2.csv', 'pthread_strict_34_4.csv', 'pthread_strict_34_8.csv']
input_folder = 'pthread_slow_01csv'
file_names = ['pthread_slow_01_1.csv', 'pthread_slow_01_2.csv', 'pthread_slow_01_4.csv', 'pthread_slow_01_8.csv']

# 初始化用於儲存資料的結構
process_counts = []
overall_times = []
computation_times = []
communication_times = []
io_times = []

# 讀取每個 CSV 檔案並提取需要的數據
for file_name in file_names:
    file_path = os.path.join(input_folder, file_name)
    df = pd.read_csv(file_path)

    # 取得 process 數量，假設檔案名格式為 'average_XX.csv'
    process_count = int(file_name.split('_')[3].split('.')[0])

    # 提取 Overall、Computation、Communication 和 IO 的時間
    # overall_time = df.loc[df['Category'] == 'Main', 'Average Total Time (s)'].values[0]
    # computation_time = df.loc[df['Category'] == 'compute', 'Average Total Time (s)'].values[0]
    # communication_time = df.loc[df['Category'] == 'Communication', 'Average Total Time (s)'].values[0]
    # io_time = df.loc[df['Category'] == 'IO', 'Average Total Time (s)'].values[0]
    df['Range'] = df['Range'].str.replace(':', '').str.strip()
    overall_time = df.loc[df['Range'] == 'Main', 'Total Time (s)'].values[0]
    io_time = df.loc[df['Range'] == 'IO', 'Total Time (s)'].values[0]
    computation_time = df.loc[df['Range'] == 'Compute', 'Total Time (s)'].values[0]
    # 儲存 process 數量和對應的運行時間
    process_counts.append(process_count)
    overall_times.append(overall_time)
    computation_times.append(computation_time)
    io_times.append(io_time)

# 計算 Speedup，將 process=1 的時間作為基準
base_overall_time = overall_times[0]
base_computation_time = computation_times[0]
base_io_time = io_times[0]

# 計算各類的 speedup
overall_speedup = [base_overall_time / time for time in overall_times]
computation_speedup = [base_computation_time / time for time in computation_times]
io_speedup = [base_io_time / time for time in io_times]

# 計算 ideal speedup (理想狀況下的線性加速)
ideal_speedup = process_counts  # 理想 speedup 隨著 process 數量線性增加
# ideal_speedup = [count / 4 for count in process_counts]
# ideal_speedup = [count / 4 for count in process_counts if isinstance(count, (int, float))]

print("Ideal speedup:", ideal_speedup)

# 繪製折線圖
plt.figure(figsize=(8, 6))

# 繪製各類 speedup 的折線
plt.plot(process_counts, overall_speedup, marker='o', linestyle='-', color='#5A80B8', label='Overall speedup')
plt.plot(process_counts, io_speedup, marker='o', linestyle='-', color='#F4CE5D', label='I/O time speedup')
plt.plot(process_counts, computation_speedup, marker='o', linestyle='-', color='#B35751', label='Computation time speedup')
plt.plot(process_counts, ideal_speedup, marker='o', linestyle='-', color='green', label='ideal time speedup')

# 設定圖表標籤與標題
plt.xlabel('number of threads', fontsize=14)
plt.ylabel('Speedup', fontsize=14)
# plt.title('Speedup - Pthread, #strict_34', fontsize=16)
plt.title('Speedup - Pthread, #slow_01', fontsize=16)

# 顯示圖例
plt.legend()

# 儲存圖檔
output_image = 'speedup_pthread_01.png'
plt.savefig(output_image)
print(f"Chart saved as {output_image}")
