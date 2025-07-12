#!/usr/bin/env python3
"""
Simple script to run the eczema classifier (PyTorch version)
===========================================================

This script provides a simple interface to train and use the eczema classifier.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from eczema_classifier import EczemaClassifier
from scin_dataset_loader import SCINDatasetLoader


def setup_environment():
    """Check and setup the environment."""
    print("Checking environment...")
    
    # Check if required packages are installed
    try:
        import torch
        import torchvision
        import sklearn
        import joblib
        print("✓ All required packages are available")
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False
    
    # Check Google Cloud authentication with timeout
    try:
        import signal
        from google.cloud import storage
        
        # Set a timeout for the authentication check
        def timeout_handler(signum, frame):
            raise TimeoutError("Google Cloud authentication timed out")
        
        # Set 10 second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)
        
        try:
            client = storage.Client()
            signal.alarm(0)  # Cancel the alarm
            print("✓ Google Cloud authentication successful")
        except TimeoutError:
            print("⚠ Google Cloud authentication timed out")
            print("This might be due to network issues or geographic restrictions")
            print("You can:")
            print("1. Try again later")
            print("2. Use the demo mode: python3 run_eczema_classifier_local.py")
            print("3. Set up a VPN if you're in a restricted region")
            return False
        except Exception as e:
            signal.alarm(0)  # Cancel the alarm
            print(f"✗ Google Cloud authentication failed: {e}")
            print("Please run: gcloud auth login && gcloud auth application-default login")
            return False
            
    except ImportError:
        print("✗ Google Cloud Storage not installed")
        print("Please install: pip install google-cloud-storage")
        return False
    
    return True


def get_interactive_choice(loader):
    """
    Get user choice for number of cases interactively.
    """
    # Get dataset statistics
    stats = loader.get_dataset_stats()
    print(f"\nDataset Statistics:")
    print(f"Total cases: {stats['total_cases']}")
    print(f"Total images: {stats['total_images']}")
    
    # Create eczema labels to show distribution
    eczema_labels = loader.create_eczema_labels()
    eczema_count = eczema_labels.sum()
    print(f"Cases with eczema: {eczema_count} ({100*eczema_count/len(eczema_labels):.1f}%)")
    
    # Check how many cases are already downloaded
    cases_with_images = loader.get_cases_with_images()
    already_downloaded = len(cases_with_images)
    print(f"Already downloaded: {already_downloaded} cases")
    
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
            print(f"  7. Use only already downloaded ({already_downloaded} cases) - NO DOWNLOAD")
            print(f"  8. Custom number")
            
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == '1':
                if already_downloaded >= 100:
                    print(f"✓ You have enough cases ({already_downloaded}). No download needed.")
                else:
                    print(f"⚠ You need {100 - already_downloaded} more cases. Will download additional cases.")
                return 100, False
            elif choice == '2':
                if already_downloaded >= 500:
                    print(f"✓ You have enough cases ({already_downloaded}). No download needed.")
                else:
                    print(f"⚠ You need {500 - already_downloaded} more cases. Will download additional cases.")
                return 500, False
            elif choice == '3':
                if already_downloaded >= 1000:
                    print(f"✓ You have enough cases ({already_downloaded}). No download needed.")
                else:
                    print(f"⚠ You need {1000 - already_downloaded} more cases. Will download additional cases.")
                return 1000, False
            elif choice == '4':
                if already_downloaded >= 2000:
                    print(f"✓ You have enough cases ({already_downloaded}). No download needed.")
                else:
                    print(f"⚠ You need {2000 - already_downloaded} more cases. Will download additional cases.")
                return 2000, False
            elif choice == '5':
                return eczema_count, True  # True means use only eczema cases
            elif choice == '6':
                return stats['total_cases'], False
            elif choice == '7':
                print(f"✓ Using {already_downloaded} already downloaded cases. No download needed.")
                return already_downloaded, False
            elif choice == '8':
                while True:
                    try:
                        custom_num = input(f"Enter custom number (1-{stats['total_cases']}): ").strip()
                        num_cases = int(custom_num)
                        if 1 <= num_cases <= stats['total_cases']:
                            if already_downloaded >= num_cases:
                                print(f"✓ You have enough cases ({already_downloaded}). No download needed.")
                            else:
                                print(f"⚠ You need {num_cases - already_downloaded} more cases. Will download additional cases.")
                            return num_cases, False
                        else:
                            print(f"Please enter a number between 1 and {stats['total_cases']}")
                    except ValueError:
                        print("Please enter a valid number")
            else:
                print("Please enter a valid choice (1-8)")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return None, False

def get_balance_method_choice():
    """
    Get user choice for class imbalance handling method.
    """
    print(f"\nClass Imbalance Handling Methods:")
    print(f"  1. Balanced Weights (default)")
    print(f"  2. SMOTE (Synthetic Minority Oversampling)")
    print(f"  3. Undersampling (reduce majority class)")
    print(f"  4. SMOTEENN (SMOTE + Edited Nearest Neighbors)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                return 'balanced_weights'
            elif choice == '2':
                return 'smote'
            elif choice == '3':
                return 'undersample'
            elif choice == '4':
                return 'smoteenn'
            else:
                print("Please enter a valid choice (1-4)")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 'balanced_weights'

def train_classifier(test_size=0.2):
    """
    Train the eczema classifier with interactive case selection.
    
    Args:
        test_size: Fraction of data for testing
    """
    print("Training eczema classifier...")
    
    # Initialize dataset loader
    loader = SCINDatasetLoader()
    
    # Download and load metadata
    print("Loading dataset metadata...")
    loader.download_metadata()
    loader.load_metadata()
    
    # Get interactive choice
    choice_result = get_interactive_choice(loader)
    if choice_result[0] is None:
        return None, None
    
    num_cases, use_only_eczema = choice_result
    
    # Check how many cases are already downloaded
    cases_with_images = loader.get_cases_with_images()
    already_downloaded = len(cases_with_images)
    
    # Download images based on choice
    if use_only_eczema:
        print(f"Using all {num_cases} eczema cases...")
        eczema_labels = loader.create_eczema_labels()
        eczema_cases = loader.cases_and_labels_df[eczema_labels == 1]['case_id'].tolist()
        loader.download_images(case_ids=eczema_cases)
    else:
        # Smart download logic
        if already_downloaded >= num_cases:
            print(f"You already have {already_downloaded} cases downloaded, which is enough for {num_cases} cases.")
            print("Using existing downloaded cases without downloading anything new.")
            # Use existing cases, no need to download
        else:
            print(f"You have {already_downloaded} cases downloaded, need {num_cases} cases.")
            print(f"Downloading {num_cases - already_downloaded} additional cases...")
            # Get cases that are not already downloaded
            existing_case_ids = set(cases_with_images['case_id'].tolist())
            all_case_ids = set(loader.cases_and_labels_df['case_id'].tolist())
            available_case_ids = list(all_case_ids - existing_case_ids)
            
            # Sample from available cases
            import random
            random.seed(42)  # For reproducibility
            additional_cases_needed = num_cases - already_downloaded
            if len(available_case_ids) >= additional_cases_needed:
                additional_cases = random.sample(available_case_ids, additional_cases_needed)
                loader.download_images(case_ids=additional_cases)
            else:
                print(f"Warning: Only {len(available_case_ids)} additional cases available, using all of them.")
                loader.download_images(case_ids=available_case_ids)
    
    # Get balance method choice
    balance_method = get_balance_method_choice()
    
    # Initialize classifier
    classifier = EczemaClassifier()
    
    # Prepare dataset
    features, labels, case_ids = classifier.prepare_dataset(loader, max_cases=num_cases)
    
    # Train classifier with balance method
    results = classifier.train(test_size=test_size, balance_method=balance_method)
    
    # Print results
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Training cases: {results['train_size']}")
    print(f"Test cases: {results['test_size']}")
    print(f"Eczema cases in training: {results['train_eczema_count']}")
    print(f"Eczema cases in test: {results['test_eczema_count']}")
    print(f"Test Accuracy: {results['accuracy']:.3f}")
    print(f"Test AUC: {results['auc']:.3f}")
    
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Save model and plots
    classifier.save_model('eczema_classifier_model')
    classifier.plot_results(results, save_path='eczema_classifier_results.png')
    
    return classifier, results


def predict_new_case(classifier, image_paths):
    """
    Predict eczema for a new case.
    
    Args:
        classifier: Trained classifier
        image_paths: List of image paths for the case
    """
    if not image_paths:
        print("No images provided for prediction")
        return
    
    prediction, probability = classifier.predict_case(image_paths)
    
    print(f"\nPrediction Results:")
    print(f"Prediction: {'Eczema' if prediction == 1 else 'No Eczema'}")
    print(f"Confidence: {probability:.3f}")
    
    if prediction == 1:
        print("This case is classified as having eczema.")
    else:
        print("This case is classified as not having eczema.")


def main():
    """Main function."""
    print("Eczema Classifier - Simple Interface (PyTorch)")
    print("==============================================")
    
    # Check environment
    if not setup_environment():
        return
    
    # Train classifier
    try:
        classifier, results = train_classifier()
        if classifier is None:
            print("\nTraining cancelled by user")
            return
        print("\n✓ Training completed successfully!")
        
        # Example of using the trained model
        print("\nExample: Predicting a case from the test set...")
        if results['test_ids']:
            # Get a test case
            test_case_id = results['test_ids'][0]
            loader = SCINDatasetLoader()
            image_paths = loader.get_image_paths_for_case(test_case_id)
            
            if image_paths:
                predict_new_case(classifier, image_paths)
            else:
                print("No images found for the test case")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Please check your internet connection and try again.")


if __name__ == "__main__":
    main() 