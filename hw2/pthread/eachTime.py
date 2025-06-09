import os
import pandas as pd
import matplotlib.pyplot as plt

# 資料夾路徑與檔案名稱
# input_folder = 'avgs_single'  # 替換為你的資料夾路徑
# file_names = ['average_01.csv', 'average_04.csv', 'average_08.csv', 'average_12.csv']  # 替換為你的檔名
# input_folder = 'pthread_34_csv'
# file_names = ['pthread_strict_34_1.csv', 'pthread_strict_34_2.csv', 'pthread_strict_34_4.csv', 'pthread_strict_34_8.csv']
# input_folder = 'pthread_33_csv'
# file_names = ['pthread_strict_33_1.csv', 'pthread_strict_33_2.csv', 'pthread_strict_33_4.csv', 'pthread_strict_33_8.csv']
input_folder = 'pthread_slow_01csv'
file_names = ['pthread_slow_01_1.csv', 'pthread_slow_01_2.csv', 'pthread_slow_01_4.csv', 'pthread_slow_01_8.csv']

# 初始化用於儲存資料的結構
process_counts = []
computation_times = []
communication_times = []
overall_times = []
io_times = []

# 讀取每個 CSV 檔案並提取需要的數據
for file_name in file_names:
    file_path = os.path.join(input_folder, file_name)
    df = pd.read_csv(file_path)

    # 取得 process 數量，假設檔案名格式為 'average_XX.csv'
    process_count = int(file_name.split('_')[3].split('.')[0])


    df['Range'] = df['Range'].str.replace(':', '').str.strip()
    overall_time = df.loc[df['Range'] == 'Main', 'Total Time (s)'].values[0]
    io_time = df.loc[df['Range'] == 'IO', 'Total Time (s)'].values[0]
    computation_time = df.loc[df['Range'] == 'Compute', 'Total Time (s)'].values[0]

    # 儲存 process 數量和對應的運行時間
    process_counts.append(process_count)
    computation_times.append(computation_time)
    overall_times.append(overall_time)
    io_times.append(io_time)

# 繪製折線圖
plt.figure(figsize=(8, 6))

# 繪製各類時間的折線
plt.plot(process_counts, computation_times, marker='o', linestyle='-', color='#D8513F', label='Computation time')
plt.plot(process_counts, overall_times, marker='o', linestyle='-', color='#F1BF41', label='Overall time')
plt.plot(process_counts, io_times, marker='o', linestyle='-', color='#5383EC', label='I/O time')

# 設定圖表標籤與標題
plt.xlabel('number of threads', fontsize=14)
plt.ylabel('runtime (sec)', fontsize=14)
# plt.title('Time Profile - Pthread, #strict_33', fontsize=16)
plt.title('Time Profile - Pthread, #slow_01', fontsize=16)

# 顯示圖例
plt.legend()

# 儲存圖檔
output_image = 'each_time_01.png'
plt.savefig(output_image)
print(f"Chart saved as {output_image}")

