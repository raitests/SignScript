import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import json
from typing import Dict, Tuple, Any
import os

class GestureClassifierTrainer:
    def __init__(self, data_path: str = './data.pickle', output_dir: str = './models'):
        self.setup_logging()
        self.data_path = data_path
        self.output_dir = output_dir
        self.setup_output_directory()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_output_directory(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        try:
            with open(self.data_path, 'rb') as f:
                data_dict = pickle.load(f)
                
            # Check and standardize data
            data = [np.array(sample) for sample in data_dict['data'] if isinstance(sample, (list, np.ndarray))]
            max_length = max(len(sample) for sample in data)
            standardized_data = [np.pad(sample, (0, max_length - len(sample)), 'constant') for sample in data]
            
            data = np.asarray(standardized_data)
            labels = np.asarray(data_dict['labels'])
            
            # Ensure that labels are valid and non-empty
            unique_labels = np.unique(labels)
            if len(unique_labels) == 0:
                self.logger.error("No valid labels found in the dataset!")
                raise ValueError("No valid labels found!")
                
            self.logger.info(f"Loaded dataset with {len(data)} samples")
            self.logger.info(f"Number of features: {data.shape[1]}")
            self.logger.info(f"Number of classes: {len(unique_labels)}")
            
            return data, labels
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> None:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()

    def plot_feature_importance(self, model: RandomForestClassifier, num_features: int = 20) -> None:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:num_features]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Top {num_features} Feature Importances')
        plt.bar(range(num_features), importances[indices])
        plt.xticks(range(num_features), indices, rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
        plt.close()

    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_params_

    def train_and_evaluate(self, save_model: bool = True) -> Dict[str, Any]:
        data, labels = self.load_data()
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
        )
        
        self.logger.info("Optimizing hyperparameters...")
        best_params = self.optimize_hyperparameters(X_train, y_train)
        
        self.logger.info("Training model with optimized parameters...")
        model = RandomForestClassifier(**best_params, random_state=42)
        model.fit(X_train, y_train)
        
        # Cross-validation to estimate model's accuracy
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Evaluate on test data
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        
        test_accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Plot and save confusion matrix and feature importance
        self.plot_confusion_matrix(y_test, y_pred, np.unique(labels))
        self.plot_feature_importance(model)
        
        results = {
            'test_accuracy': test_accuracy,
            'train_accuracy': train_accuracy,
            'cross_val_mean': cv_scores.mean(),
            'cross_val_std': cv_scores.std(),
            'classification_report': report,
            'hyperparameters': best_params,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save training results to a JSON file
        with open(os.path.join(self.output_dir, 'training_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        if save_model:
            model_path = os.path.join(self.output_dir, 'model.p')
            with open(model_path, 'wb') as f:
                pickle.dump({'model': model, 'metadata': results}, f)
            self.logger.info(f"Model saved to {model_path}")
            
        self.logger.info("\nTraining Results:")
        self.logger.info(f"Test Accuracy: {test_accuracy*100:.2f}%")
        self.logger.info(f"Train Accuracy: {train_accuracy*100:.2f}%")
        self.logger.info(f"Cross-validation Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
        
        return results

if __name__ == "__main__":
    trainer = GestureClassifierTrainer()
    results = trainer.train_and_evaluate(save_model=True)
