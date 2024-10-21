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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay


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
        """Evaluate the model on the test set and calculate per-label accuracy, precision, recall, F1 score, ROC curve, and confusion matrix."""
        self.model.eval()
        all_labels = []
        all_predictions = []
        all_probabilities = []
        correct = 0
        total = 0
        label_correct = defaultdict(int)
        label_total = defaultdict(int)

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Evaluating', leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.softmax(outputs, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for label, prediction in zip(labels, predicted):
                    label_total[label.item()] += 1
                    if label == prediction:
                        label_correct[label.item()] += 1

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')

        # Confusion Matrix
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, cmap='Blues')  # Using 'Blues' color map to make the colors softer
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()
        print("Confusion matrix has been saved to 'confusion_matrix.png'.")


        # ROC Curve (for each class)
        plt.figure(figsize=(10, 6))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.num_classes):
            y_true_binary = [1 if label == i else 0 for label in all_labels]
            y_score = [prob[i] for prob in all_probabilities]
            fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_score)
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f'Class {i+1} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Each Class')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'))
        plt.close()
        print("ROC curve has been saved to 'roc_curve.png'.")

        # Original return values
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
    
    trainer = IMUModelTrainer(segments_file='segments.npy', labels_file='labels.npy', scaler_file='scaler.pkl')
    trainer.run()
