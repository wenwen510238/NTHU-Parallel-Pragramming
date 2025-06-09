import os
import pandas as pd
import matplotlib.pyplot as plt

# 資料夾路徑與檔案名稱
# input_folder = 'avgs_single'  # 替換為你的資料夾路徑
# file_names = ['average_01.csv', 'average_04.csv', 'average_08.csv', 'average_12.csv']  # 替換為實際檔名
# input_folder = 'pthread_34_csv'
# file_names = ['pthread_strict_34_1.csv', 'pthread_strict_34_2.csv', 'pthread_strict_34_4.csv', 'pthread_strict_34_8.csv']
input_folder = 'pthread_slow_01csv'
file_names = ['pthread_slow_01_1.csv', 'pthread_slow_01_2.csv', 'pthread_slow_01_4.csv', 'pthread_slow_01_8.csv']
# 初始化用於儲存資料的結構
data = {
    '# of threads': [],
    'Computation': [],
    'Overall': [],
    'I/O': []
}

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

    # 將數據加入資料結構
    data['# of threads'].append(process_count)
    data['Computation'].append(computation_time)
    data['Overall'].append(overall_time)
    data['I/O'].append(io_time)

# 將資料轉為 pandas DataFrame
df_plot = pd.DataFrame(data)

# 設定圖形大小
plt.figure(figsize=(8, 6))

# 設定正確的 x 軸位置 (去除空隙)
x_positions = range(len(df_plot['# of threads']))

# 繪製堆疊長條圖
plt.bar(x_positions, df_plot['Computation'], label='Computation', color='#5A80B8')
plt.bar(x_positions, df_plot['I/O'], bottom=df_plot['Computation'], label='I/O', color='#A1BA66')
plt.bar(x_positions, df_plot['Overall'], bottom=df_plot['Computation'] + df_plot['I/O'], label='Overall', color='#B35751')

# 設定圖表標籤與標題
plt.xlabel('number of threads', fontsize=14)
plt.ylabel('runtime (seconds)', fontsize=14)
# plt.title('Time Profile - Single Node, #35', fontsize=14)
# plt.title('Time Profile - Pthread, #strict_34', fontsize=16)
plt.title('Time Profile - Pthread, #slow_01', fontsize=16)


# 將 x 軸標籤設為 1, 2, 4, 8
plt.xticks(ticks=x_positions, labels=df_plot['# of threads'])
plt.ylim(0, df_plot[['Computation', 'I/O', 'Overall']].sum(axis=1).max() * 1.1)
# 加上圖例
plt.legend()

# 儲存圖檔
# output_image = 'runtime.png'
output_image = 'runtime_01.png'
plt.savefig(output_image)
print(f"Chart saved as {output_image}")
