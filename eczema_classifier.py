#!/usr/bin/env python3
"""
Eczema Classifier using ResNet Features (PyTorch)
=================================================

This script creates a logistic regression classifier that uses ResNet features
to detect eczema in the SCIN dataset using PyTorch.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import joblib

from scin_dataset_loader import SCINDatasetLoader


class EczemaClassifier:
    """
    A classifier that uses ResNet features to detect eczema (PyTorch version).
    """
    
    def __init__(self, 
                 resnet_model: str = 'resnet50',
                 feature_dim: int = 2048,
                 random_state: int = 42,
                 device: str = None):
        """
        Initialize the eczema classifier.
        
        Args:
            resnet_model: ResNet model to use for feature extraction
            feature_dim: Dimension of ResNet features
            random_state: Random seed for reproducibility
            device: Device to use for PyTorch (cuda/cpu)
        """
        self.resnet_model = resnet_model
        self.feature_dim = feature_dim
        self.random_state = random_state
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize ResNet model for feature extraction
        self.resnet = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet.eval()
        self.resnet.to(self.device)
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize classifier and scaler
        self.classifier = LogisticRegression(
            random_state=random_state, 
            max_iter=1000,
            class_weight='balanced'  # Handle class imbalance
        )
        self.scaler = StandardScaler()
        
        # Storage for results
        self.features = None
        self.labels = None
        self.case_ids = None
        
    def extract_resnet_features(self, image_path: str) -> np.ndarray:
        """
        Extract ResNet features from a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ResNet features as numpy array
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.resnet(img_tensor)
                features = features.squeeze().cpu().numpy()
            
            return features
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # Return zero features if image processing fails
            return np.zeros(self.feature_dim)
    
    def extract_features_for_case(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract features from multiple images of a case and average them.
        
        Args:
            image_paths: List of image paths for a case
            
        Returns:
            Averaged features for the case
        """
        if not image_paths:
            return np.zeros(self.feature_dim)
        
        case_features = []
        for img_path in image_paths:
            features = self.extract_resnet_features(img_path)
            case_features.append(features)
        
        # Average features across all images for the case
        return np.mean(case_features, axis=0)
    
    def prepare_dataset(self, loader: SCINDatasetLoader, 
                       max_cases: int = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare the dataset by extracting features and creating labels.
        
        Args:
            loader: SCIN dataset loader
            max_cases: Maximum number of cases to process (None for all)
            
        Returns:
            Tuple of (features, labels, case_ids)
        """
        print("Preparing dataset...")
        
        # Get cases with downloaded images
        cases_df = loader.get_cases_with_images()
        
        if max_cases:
            cases_df = cases_df.head(max_cases)
        
        print(f"Processing {len(cases_df)} cases with images...")
        
        # Create eczema labels
        eczema_labels = loader.create_eczema_labels()
        
        features_list = []
        labels_list = []
        case_ids_list = []
        
        for idx, row in cases_df.iterrows():
            case_id = row['case_id']
            
            # Get image paths for this case
            image_paths = loader.get_image_paths_for_case(case_id)
            
            if image_paths:
                # Extract features
                case_features = self.extract_features_for_case(image_paths)
                features_list.append(case_features)
                
                # Get label
                case_label = eczema_labels.loc[idx]
                labels_list.append(case_label)
                case_ids_list.append(case_id)
                
                if len(features_list) % 50 == 0:
                    print(f"Processed {len(features_list)} cases...")
        
        # Convert to numpy arrays
        self.features = np.array(features_list)
        self.labels = np.array(labels_list)
        self.case_ids = case_ids_list
        
        print(f"Dataset prepared: {len(self.features)} cases")
        print(f"Eczema cases: {np.sum(self.labels)} ({100*np.sum(self.labels)/len(self.labels):.1f}%)")
        
        return self.features, self.labels, self.case_ids
    
    def handle_class_imbalance(self, X_train, y_train, method='balanced_weights'):
        """
        Handle class imbalance using various methods.
        
        Args:
            X_train: Training features
            y_train: Training labels
            method: Method to use ('balanced_weights', 'smote', 'undersample', 'smoteenn')
            
        Returns:
            Balanced training data
        """
        print(f"Handling class imbalance using: {method}")
        
        # Show original class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"Original class distribution: {dict(zip(unique, counts))}")
        
        if method == 'balanced_weights':
            # Use class weights (already set in classifier initialization)
            print("Using balanced class weights")
            return X_train, y_train
            
        elif method == 'smote':
            # Oversample minority class using SMOTE
            try:
                smote = SMOTE(random_state=self.random_state)
                X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
                print(f"SMOTE balanced class distribution: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")
                return X_balanced, y_balanced
            except ImportError:
                print("SMOTE not available, falling back to balanced weights")
                return X_train, y_train
                
        elif method == 'undersample':
            # Undersample majority class
            try:
                undersampler = RandomUnderSampler(random_state=self.random_state)
                X_balanced, y_balanced = undersampler.fit_resample(X_train, y_train)
                print(f"Undersampled class distribution: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")
                return X_balanced, y_balanced
            except ImportError:
                print("RandomUnderSampler not available, falling back to balanced weights")
                return X_train, y_train
                
        elif method == 'smoteenn':
            # Combine SMOTE and ENN
            try:
                smoteenn = SMOTEENN(random_state=self.random_state)
                X_balanced, y_balanced = smoteenn.fit_resample(X_train, y_train)
                print(f"SMOTEENN balanced class distribution: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")
                return X_balanced, y_balanced
            except ImportError:
                print("SMOTEENN not available, falling back to balanced weights")
                return X_train, y_train
        else:
            print("Unknown method, using balanced weights")
            return X_train, y_train

    def train(self, test_size: float = 0.2, balance_method: str = 'balanced_weights') -> Dict:
        """
        Train the logistic regression classifier with class imbalance handling.
        
        Args:
            test_size: Fraction of data to use for testing
            balance_method: Method to handle class imbalance
            
        Returns:
            Dictionary with training results
        """
        if self.features is None:
            raise ValueError("Dataset not prepared. Call prepare_dataset() first.")
        
        print("Training classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
            self.features, self.labels, self.case_ids,
            test_size=test_size, 
            random_state=self.random_state,
            stratify=self.labels
        )
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train, balance_method)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train_scaled, y_train_balanced)
        
        # Make predictions
        y_pred = self.classifier.predict(X_test_scaled)
        y_pred_proba = self.classifier.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        results = {
            'train_size': len(X_train_balanced),
            'test_size': len(X_test),
            'train_eczema_count': np.sum(y_train_balanced),
            'test_eczema_count': np.sum(y_test),
            'accuracy': self.classifier.score(X_test_scaled, y_test),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'test_ids': test_ids,
            'balance_method': balance_method
        }
        
        print(f"Training completed!")
        print(f"Balance method: {balance_method}")
        print(f"Test accuracy: {results['accuracy']:.3f}")
        print(f"Test AUC: {results['auc']:.3f}")
        
        return results
    
    def plot_results(self, results: Dict, save_path: str = None):
        """
        Plot training results.
        
        Args:
            results: Results dictionary from training
            save_path: Path to save plots (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
        axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {results["auc"]:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Feature Importance (top 20)
        feature_importance = np.abs(self.classifier.coef_[0])
        top_indices = np.argsort(feature_importance)[-20:]
        top_importance = feature_importance[top_indices]
        
        axes[1, 0].barh(range(len(top_importance)), top_importance)
        axes[1, 0].set_yticks(range(len(top_importance)))
        axes[1, 0].set_yticklabels([f'Feature {i}' for i in top_indices])
        axes[1, 0].set_xlabel('Feature Importance (|Coefficient|)')
        axes[1, 0].set_title('Top 20 Most Important Features')
        
        # Prediction Distribution
        axes[1, 1].hist(results['y_pred_proba'][results['y_test'] == 0], 
                       alpha=0.7, label='No Eczema', bins=20)
        axes[1, 1].hist(results['y_pred_proba'][results['y_test'] == 1], 
                       alpha=0.7, label='Eczema', bins=20)
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Prediction Probability Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {save_path}")
        
        plt.show()
    
    def save_model(self, model_path: str):
        """
        Save the trained model and scaler.
        
        Args:
            model_path: Path to save the model
        """
        model_dir = Path(model_path)
        model_dir.mkdir(exist_ok=True)
        
        # Save classifier
        joblib.dump(self.classifier, model_dir / 'classifier.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, model_dir / 'scaler.pkl')
        
        # Save model info
        model_info = {
            'resnet_model': self.resnet_model,
            'feature_dim': self.feature_dim,
            'random_state': self.random_state,
            'device': str(self.device)
        }
        joblib.dump(model_info, model_dir / 'model_info.pkl')
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        model_dir = Path(model_path)
        
        # Load classifier
        self.classifier = joblib.load(model_dir / 'classifier.pkl')
        
        # Load scaler
        self.scaler = joblib.load(model_dir / 'scaler.pkl')
        
        # Load model info
        model_info = joblib.load(model_dir / 'model_info.pkl')
        self.resnet_model = model_info['resnet_model']
        self.feature_dim = model_info['feature_dim']
        self.random_state = model_info['random_state']
        
        print(f"Model loaded from {model_path}")
    
    def predict_case(self, image_paths: List[str]) -> Tuple[int, float]:
        """
        Predict eczema for a single case.
        
        Args:
            image_paths: List of image paths for the case
            
        Returns:
            Tuple of (prediction, probability)
        """
        # Extract features
        features = self.extract_features_for_case(image_paths)
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.classifier.predict(features_scaled)[0]
        probability = self.classifier.predict_proba(features_scaled)[0, 1]
        
        return prediction, probability


def main():
    """
    Main function to train and evaluate the eczema classifier.
    """
    print("Eczema Classifier using ResNet Features (PyTorch)")
    print("=================================================")
    
    # Initialize dataset loader
    loader = SCINDatasetLoader()
    
    # Download and load metadata
    print("Loading dataset...")
    loader.download_metadata()
    loader.load_metadata()
    
    # Get dataset statistics
    stats = loader.get_dataset_stats()
    print(f"\nDataset Statistics:")
    print(f"Total cases: {stats['total_cases']}")
    print(f"Total images: {stats['total_images']}")
    
    # Create eczema labels to show distribution
    eczema_labels = loader.create_eczema_labels()
    eczema_count = eczema_labels.sum()
    print(f"Cases with eczema: {eczema_count} ({100*eczema_count/len(eczema_labels):.1f}%)")
    
    # Interactive case selection
    while True:
        try:
            print(f"\nHow many cases would you like to use for training?")
            print(f"Options:")
            print(f"  1. Quick test (100 cases)")
            print(f"  2. Small dataset (500 cases)")
            print(f"  3. Medium dataset (1000 cases)")
            print(f"  4. Large dataset (2000 cases)")
            print(f"  5. All eczema cases ({eczema_count} cases)")
            print(f"  6. All cases ({stats['total_cases']} cases)")
            print(f"  7. Custom number")
            
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                num_cases = 100
                break
            elif choice == '2':
                num_cases = 500
                break
            elif choice == '3':
                num_cases = 1000
                break
            elif choice == '4':
                num_cases = 2000
                break
            elif choice == '5':
                num_cases = eczema_count
                # Use only eczema cases
                eczema_cases = loader.cases_and_labels_df[eczema_labels == 1]['case_id'].tolist()
                print(f"Using all {eczema_count} eczema cases...")
                loader.download_images(case_ids=eczema_cases)
                # Initialize classifier
                classifier = EczemaClassifier()
                # Prepare dataset with only eczema cases
                features, labels, case_ids = classifier.prepare_dataset(loader, max_cases=eczema_count)
                # Train classifier
                results = classifier.train(test_size=0.2)
                # Print results
                print("\n" + "="*50)
                print("CLASSIFICATION RESULTS")
                print("="*50)
                print(f"Test Accuracy: {results['accuracy']:.3f}")
                print(f"Test AUC: {results['auc']:.3f}")
                print(f"Test Cases: {results['test_size']}")
                print(f"Eczema Cases in Test: {results['test_eczema_count']}")
                print("\nClassification Report:")
                print(results['classification_report'])
                # Plot results
                classifier.plot_results(results, save_path='eczema_classifier_results.png')
                # Save model
                classifier.save_model('eczema_classifier_model')
                print("\nTraining completed successfully!")
                return
            elif choice == '6':
                num_cases = stats['total_cases']
                break
            elif choice == '7':
                while True:
                    try:
                        custom_num = input(f"Enter custom number (1-{stats['total_cases']}): ").strip()
                        num_cases = int(custom_num)
                        if 1 <= num_cases <= stats['total_cases']:
                            break
                        else:
                            print(f"Please enter a number between 1 and {stats['total_cases']}")
                    except ValueError:
                        print("Please enter a valid number")
                break
            else:
                print("Please enter a valid choice (1-7)")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return
    
    print(f"\nDownloading {num_cases} cases...")
    sample_cases = loader.cases_and_labels_df.sample(num_cases)['case_id'].tolist()
    loader.download_images(case_ids=sample_cases)
    
    # Initialize classifier
    classifier = EczemaClassifier()
    
    # Prepare dataset
    features, labels, case_ids = classifier.prepare_dataset(loader, max_cases=num_cases)
    
    # Train classifier
    results = classifier.train(test_size=0.2)
    
    # Print results
    print("\n" + "="*50)
    print("CLASSIFICATION RESULTS")
    print("="*50)
    print(f"Test Accuracy: {results['accuracy']:.3f}")
    print(f"Test AUC: {results['auc']:.3f}")
    print(f"Test Cases: {results['test_size']}")
    print(f"Eczema Cases in Test: {results['test_eczema_count']}")
    
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Plot results
    classifier.plot_results(results, save_path='eczema_classifier_results.png')
    
    # Save model
    classifier.save_model('eczema_classifier_model')
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main() 