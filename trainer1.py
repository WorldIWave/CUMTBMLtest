import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import joblib
from model import CNNWithSVM, CNNFeatureExtractor, MultiClassHingeLoss
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict

class IMUDataProcessor:
    def __init__(self, file_path, window_size=100, step_size=50):
        self.file_path = file_path
        self.window_size = window_size
        self.step_size = step_size
        self.sensor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        self.segments = []
        self.labels = []

    def load_data(self):
        """Load data from CSV file."""
        self.data_df = pd.read_csv(self.file_path)

    def clean_data(self):
        """Clean age, height, and weight fields."""
        self.data_df['age'] = self.data_df['age'].str.strip("[]'").astype(float)
        self.data_df['height'] = self.data_df['height'].str.strip("[]'cm").astype(float)
        self.data_df['weight'] = self.data_df['weight'].str.strip("[]'kg").astype(float)

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
        print(f"Total {self.segments.shape[0]} segments generated, each with shape {self.segments.shape[1:]}.")
        print(f"Data has been processed and saved as '{segments_file}', '{labels_file}', and '{scaler_file}'.")

    def process(self):
        """Run the full processing pipeline."""
        self.load_data()
        self.clean_data()
        self.segment_data()
        self.standardize_data()
        self.save_data()

class IMUModelTrainer:
    def __init__(self, segments_file, labels_file, scaler_file, num_classes=12, batch_size=64, num_epochs=20, learning_rate=0.001, n_splits=5, output_dir='output'):
        self.segments_file = segments_file
        self.labels_file = labels_file
        self.scaler_file = scaler_file
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.n_splits = n_splits
        self.output_dir = output_dir

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self):
        """Load processed segments and labels."""
        segments = np.load(self.segments_file)
        labels = np.load(self.labels_file)

        self.X = torch.tensor(segments, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long) - 1  # Assuming labels start from 1

    def prepare_model(self):
        """Prepare model, loss function, and optimizer."""
        feature_extractor = CNNFeatureExtractor()
        self.model = CNNWithSVM(feature_extractor, self.num_classes).to(self.device)
        self.criterion = MultiClassHingeLoss(num_classes=self.num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        """Train the model using K-Fold Cross Validation and evaluate on each fold."""
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        overall_correct = 0
        overall_total = 0
        overall_label_correct = defaultdict(int)
        overall_label_total = defaultdict(int)

        # To track loss per epoch
        epoch_losses = []
        epoch_val_losses = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(self.X)):
            print(f'Fold {fold + 1}/{self.n_splits}')

            # Prepare data for this fold
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            self.prepare_model()

            for epoch in range(self.num_epochs):
                self.model.train()
                running_loss = 0.0

                # Training loop with progress bar
                for inputs, labels in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{self.num_epochs} for Fold {fold + 1}'):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)

                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(train_dataset)
                epoch_losses.append(epoch_loss)
                print(f'Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {epoch_loss:.4f}')

                # Validation loss
                val_loss = self.evaluate_loss(test_loader)
                epoch_val_losses.append(val_loss)
                print(f'Epoch {epoch + 1}/{self.num_epochs}, Validation Loss: {val_loss:.4f}')

            # Evaluate the model on the test set for this fold
            fold_correct, fold_total, fold_label_correct, fold_label_total = self.evaluate(test_loader)
            overall_correct += fold_correct
            overall_total += fold_total
            for label in range(self.num_classes):
                overall_label_correct[label] += fold_label_correct[label]
                overall_label_total[label] += fold_label_total[label]

        # Print overall accuracy and per-label accuracy
        overall_accuracy = overall_correct / overall_total
        print(f'Overall Test Accuracy: {overall_accuracy:.4f}')

        for label in range(self.num_classes):
            if overall_label_total[label] > 0:
                label_accuracy = overall_label_correct[label] / overall_label_total[label]
                print(f'Overall Accuracy for label {label + 1}: {label_accuracy:.4f}')

        # Save the trained model
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'cnn_model.pth'))
        print("Model weights have been saved to 'cnn_model.pth'.")

        # Plot and save the training and validation loss curves
        self.plot_training_loss(epoch_losses, epoch_val_losses)

    def evaluate(self, test_loader):
        """Evaluate the model on the test set and calculate per-label accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        label_correct = defaultdict(int)
        label_total = defaultdict(int)

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Evaluating', leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for label, prediction in zip(labels, predicted):
                    label_total[label.item()] += 1
                    if label == prediction:
                        label_correct[label.item()] += 1

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')

        # Calculate per-label accuracy
        for label in range(self.num_classes):
            if label_total[label] > 0:
                label_accuracy = label_correct[label] / label_total[label]
                print(f'Accuracy for label {label + 1}: {label_accuracy:.4f}')

        return correct, total, label_correct, label_total

    def evaluate_loss(self, data_loader):
        """Evaluate the model to calculate the average loss on the given dataset."""
        self.model.eval()
        running_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        return running_loss / total_samples

    def plot_training_loss(self, train_losses, val_losses):
        """Plot the training and validation loss over epochs."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', label='Training Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', linestyle='-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'training_validation_loss_curve.png'))
        plt.close()
        print("Training and validation loss curve has been saved to 'training_validation_loss_curve.png'.")

    def run(self):
        """Run the complete training and evaluation pipeline using K-Fold Cross Validation."""
        self.load_data()
        self.train()

if __name__ == "__main__":
    processor = IMUDataProcessor('processed_imu_data.csv')
    processor.process()

    trainer = IMUModelTrainer(segments_file='segments.npy', labels_file='labels.npy', scaler_file='scaler.pkl')
    trainer.run()