import os
import sys
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
DATA_PICKLE_PATH = './data_20250209_181645.pickle'
MODEL_OUTPUT_PATH = 'model.p'
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_data(data_path: str):
    """Load dataset from a pickle file with validation."""
    if not os.path.exists(data_path):
        print(f"Error: Data file '{data_path}' does not exist.")
        sys.exit(1)
        
    try:
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        # Extract data and metadata if available
        data = np.asarray(data_dict.get('data', []))
        labels = np.asarray(data_dict.get('labels', []))
        metadata = data_dict.get('metadata', {})
        
        if data.size == 0 or labels.size == 0:
            print("Error: Loaded data or labels are empty.")
            sys.exit(1)
            
        print(f"\nDataset Information:")
        print(f"- Number of samples: {len(data)}")
        print(f"- Number of classes: {len(np.unique(labels))}")
        print(f"- Feature dimension: {data.shape[1]}")
        
        if metadata:
            print("\nDataset Metadata:")
            for key, value in metadata.items():
                print(f"- {key}: {value}")
                
        return data, labels
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix using seaborn."""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'confusion_matrix_{timestamp}.png')
    plt.close()

def evaluate_model(model, X_test, y_test, class_labels):
    """Perform comprehensive model evaluation."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Performance:")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    
    # Generate detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create and save confusion matrix plot
    plot_confusion_matrix(y_test, y_pred, np.unique(class_labels))
    
    # Feature importance analysis
    feature_importance = model.feature_importances_
    most_important = np.argsort(feature_importance)[-10:]  # Top 10 features
    print("\nTop 10 Most Important Features:")
    for idx in most_important[::-1]:
        print(f"Feature {idx}: {feature_importance[idx]:.4f}")

def train_classifier(data: np.ndarray, labels: np.ndarray) -> RandomForestClassifier:
    """Train a RandomForest classifier with cross-validation."""
    # Perform train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=TEST_SIZE, shuffle=True, 
            stratify=labels, random_state=RANDOM_STATE
        )
    except Exception as e:
        print(f"Error during train/test split: {e}")
        sys.exit(1)

    # Initialize and train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1  # Use all available cores
    )
    
    # Perform cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    
    # Train final model
    print("\nTraining final model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, labels)
    
    return model

def save_model(model, output_path: str):
    """Save the trained model with metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{os.path.splitext(output_path)[0]}_{timestamp}.p"
    
    model_info = {
        'model': model,
        'metadata': {
            'training_date': timestamp,
            'n_estimators': model.n_estimators,
            'n_classes': len(model.classes_),
            'feature_dimension': model.n_features_in_
        }
    }
    
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(model_info, f)
        print(f"\nModel saved successfully to '{output_path}'")
    except Exception as e:
        print(f"Error saving model: {e}")

def main():
    print("Starting classifier training...")
    data, labels = load_data(DATA_PICKLE_PATH)
    model = train_classifier(data, labels)
    save_model(model, MODEL_OUTPUT_PATH)
    print("\nTraining completed successfully!")

if __name__ == '__main__':
    main()
