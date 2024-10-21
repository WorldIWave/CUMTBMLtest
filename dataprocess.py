import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

class IMUDataProcessor:
    def __init__(self, input_file, output_dir, window_size=100, step_size=50, cutoff_frequency=5, sampling_rate=100):
        self.input_file = input_file
        self.output_dir = output_dir
        self.window_size = window_size
        self.step_size = step_size
        self.cutoff_frequency = cutoff_frequency
        self.sampling_rate = sampling_rate
        self.sensor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        self.segments = []
        self.labels = []
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """Load data from CSV file."""
        self.data_df = pd.read_csv(self.input_file)

    def clean_data(self):
        """Clean age, height, and weight fields."""
        self.data_df['age'] = self.data_df['age'].str.strip("[]'").astype(float)
        self.data_df['height'] = self.data_df['height'].str.strip("[]'cm").astype(float)
        self.data_df['weight'] = self.data_df['weight'].str.strip("[]'kg").astype(float)

    def butter_lowpass_filter(self, data, cutoff, fs, order=4):
        """Apply Butterworth low-pass filter to data."""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def process_group(self, group, subject, activity, trial):
        """Process a single group of data, applying filtering and saving visualization."""
        # Apply low-pass filter to accelerometer and gyroscope data
        filtered_group = group.copy()
        for column in self.sensor_columns:
            filtered_group[column] = self.butter_lowpass_filter(group[column], self.cutoff_frequency, self.sampling_rate)

        # Save filtered data visualization
        self.visualize_filtered_data(filtered_group, subject, activity, trial)

        return filtered_group

    def visualize_filtered_data(self, group, subject, activity, trial):
        """Visualize and save filtered accelerometer and gyroscope data."""
        plt.figure(figsize=(15, 10))
        plt.suptitle(f'Subject {subject} - Activity {activity} - Trial {trial}', fontsize=16)

        # Plot accelerometer data
        plt.subplot(2, 1, 1)
        plt.plot(group['acc_x'], label='acc_x (filtered)')
        plt.plot(group['acc_y'], label='acc_y (filtered)')
        plt.plot(group['acc_z'], label='acc_z (filtered)')
        plt.xlabel('Time Step')
        plt.ylabel('Acceleration (g)')
        plt.title('Filtered Accelerometer Data')
        plt.legend()

        # Plot gyroscope data
        plt.subplot(2, 1, 2)
        plt.plot(group['gyro_x'], label='gyro_x (filtered)')
        plt.plot(group['gyro_y'], label='gyro_y (filtered)')
        plt.plot(group['gyro_z'], label='gyro_z (filtered)')
        plt.xlabel('Time Step')
        plt.ylabel('Gyroscope (dps)')
        plt.title('Filtered Gyroscope Data')
        plt.legend()

        # Save the figure
        output_path = os.path.join(self.output_dir, f'subject_{subject}_activity_{activity}_trial_{trial}.png')
        plt.savefig(output_path)
        plt.close()
        # print(f'Visualization saved to {output_path}')

    def process_all_data(self):
        """Process all the data, group by subject, activity, and trial."""
        processed_data = []
        grouped = self.data_df.groupby(['subject', 'activity_number', 'trial_number'])

        for (subject, activity, trial), group in tqdm(grouped, desc='Processing data'):
            processed_group = self.process_group(group, subject, activity, trial)
            processed_data.append(processed_group)

        # Save processed data to CSV
        processed_df = pd.concat(processed_data, ignore_index=True)
        output_csv_path = os.path.join(self.output_dir, 'processed_imu_data.csv')
        processed_df.to_csv(output_csv_path, index=False)
        print(f"Processed data saved to {output_csv_path}")
        return output_csv_path

    def segment_data(self):
        """Segment the sensor data based on window and step size."""
        grouped = self.data_df.groupby(['subject', 'activity_number', 'trial_number'])

        for (subject, activity, trial), group in tqdm(grouped, desc='Segmenting data'):
            data = group[self.sensor_columns].values
            label = activity
            num_segments = int((len(data) - self.window_size) / self.step_size) + 1

            for i in range(num_segments):
                start = i * self.step_size
                end = start + self.window_size
                segment = data[start:end]

                if len(segment) == self.window_size:
                    self.segments.append(segment)
                    self.labels.append(label)

    def standardize_data(self):
        """Standardize the segmented data."""
        segments_array = np.array(self.segments)
        num_samples, window_size, num_features = segments_array.shape
        segments_reshaped = segments_array.reshape(-1, num_features)

        self.scaler = StandardScaler()
        segments_scaled = self.scaler.fit_transform(segments_reshaped)
        segments_scaled = segments_scaled.reshape(num_samples, window_size, num_features)

        self.segments = np.transpose(segments_scaled, (0, 2, 1))
        self.labels = np.array(self.labels)

    def save_data(self, segments_file='segments.npy', labels_file='labels.npy', scaler_file='scaler.pkl'):
        """Save the processed segments, labels, and scaler to files."""
        np.save(segments_file, self.segments)
        np.save(labels_file, self.labels)
        joblib.dump(self.scaler, scaler_file)
        # print(f"Total {self.segments.shape[0]} segments generated, each with shape {self.segments.shape[1:]}.")
        # print(f"Data has been processed and saved as '{segments_file}', '{labels_file}', and '{scaler_file}'.")

    def process_pipeline(self):
        """Run the full processing pipeline."""
        self.load_data()
        self.clean_data()
        processed_csv_path = self.process_all_data()
        self.data_df = pd.read_csv(processed_csv_path)
        self.segment_data()
        self.standardize_data()
        self.save_data()

if __name__ == "__main__":
    input_file = '../imu_data.csv'
    output_dir = '../processed_visualizations'
    processor = IMUDataProcessor(input_file, output_dir)
    processor.process_pipeline()
