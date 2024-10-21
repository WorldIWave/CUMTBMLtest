import os
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# 定义数据路径
data_dir = 'USC-HAD'
# 创建一个空的DataFrame来存储所有数据
columns = ['subject', 'age', 'height', 'weight', 'activity_number', 'trial_number', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
data_df = pd.DataFrame(columns=columns)

# 判断设备是否可用
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 遍历每个受试者目录
subject_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('Subject')]
for subject_dir in tqdm(subject_dirs, desc='Processing subjects'):
    subject_path = os.path.join(data_dir, subject_dir)
    subject_number = int(subject_dir.replace('Subject', ''))

    # 遍历每个.mat文件
    mat_files = [f for f in os.listdir(subject_path) if f.endswith('.mat')]
    for mat_file in mat_files:
        mat_path = os.path.join(subject_path, mat_file)

        # 提取活动编号和试验编号
        activity_number = int(mat_file.split('t')[0][1:])
        trial_number = int(mat_file.split('t')[1].split('.')[0])

        # 加载.mat文件
        mat_data = scipy.io.loadmat(mat_path)

        # 提取元数据信息
        age = mat_data.get('age', np.nan)
        height = mat_data.get('height', np.nan)
        weight = mat_data.get('weight', np.nan)

        # 提取传感器数据
        sensor_readings = mat_data.get('sensor_readings', None)

        if sensor_readings is not None:
            # 使用 GPU 或 CPU 进行加速处理
            sensor_readings = torch.tensor(sensor_readings, device=device)
            acc_x, acc_y, acc_z = sensor_readings[:, 0], sensor_readings[:, 1], sensor_readings[:, 2]
            gyro_x, gyro_y, gyro_z = sensor_readings[:, 3], sensor_readings[:, 4], sensor_readings[:, 5]

            # 将数据从 GPU 转回 CPU（如果使用 GPU）
            acc_x, acc_y, acc_z = acc_x.cpu().numpy(), acc_y.cpu().numpy(), acc_z.cpu().numpy()
            gyro_x, gyro_y, gyro_z = gyro_x.cpu().numpy(), gyro_y.cpu().numpy(), gyro_z.cpu().numpy()

            # 将数据组立成DataFrame格式
            temp_df = pd.DataFrame({
                'subject': [subject_number] * len(acc_x),
                'age': [age] * len(acc_x),
                'height': [height] * len(acc_x),
                'weight': [weight] * len(acc_x),
                'activity_number': [activity_number] * len(acc_x),
                'trial_number': [trial_number] * len(acc_x),
                'acc_x': acc_x,
                'acc_y': acc_y,
                'acc_z': acc_z,
                'gyro_x': gyro_x,
                'gyro_y': gyro_y,
                'gyro_z': gyro_z
            })

            # 将临时DataFrame追加到总DataFrame中
            data_df = pd.concat([data_df, temp_df], ignore_index=True)

            # 可视化传感器数据并保存图像
            plt.figure(figsize=(12, 8))
            plt.suptitle(f'Subject {subject_number} - Activity {activity_number} - Trial {trial_number}', fontsize=16)

            plt.subplot(2, 1, 1)
            plt.plot(acc_x, label='acc_x')
            plt.plot(acc_y, label='acc_y')
            plt.plot(acc_z, label='acc_z')
            plt.xlabel('Time Step')
            plt.ylabel('Acceleration (g)')
            plt.title('Accelerometer Data')
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(gyro_x, label='gyro_x')
            plt.plot(gyro_y, label='gyro_y')
            plt.plot(gyro_z, label='gyro_z')
            plt.xlabel('Time Step')
            plt.ylabel('Gyroscope (dps)')
            plt.title('Gyroscope Data')
            plt.legend()

            # 保存图像到与CSV相同的母目录下
            output_dir = os.path.join(data_dir, 'visualizations')
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir,
                                     f'subject_{subject_number}_activity_{activity_number}_trial_{trial_number}.png'))
            plt.close()

# 保存到CSV文件，便于后续分析
data_df.to_csv('imu_data.csv', index=False)

print("数据加载完成，已保存到imu_data.csv文件中。")
