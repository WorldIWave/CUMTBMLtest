import numpy as np
from tqdm import tqdm
import torch
from model import CNNWithSVM, CNNFeatureExtractor, MultiClassHingeLoss
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict

class IMUModelTrainer:
    def __init__(self, segments_file, labels_file, scaler_file, num_classes=12, batch_size=64, num_epochs=20, learning_rate=0.001, n_splits=5):
        self.segments_file = segments_file
        self.labels_file = labels_file
        self.scaler_file = scaler_file
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.n_splits = n_splits

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                print(f'Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {epoch_loss:.4f}')

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
        torch.save(self.model.state_dict(), 'cnn_model.pth')
        print("Model weights have been saved to 'cnn_model.pth'.")

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

    def run(self):
        """Run the complete training and evaluation pipeline using K-Fold Cross Validation."""
        self.load_data()
        self.train()

if __name__ == "__main__":

    trainer = IMUModelTrainer(segments_file='segments.npy', labels_file='labels.npy', scaler_file='scaler.pkl')
    trainer.run()
