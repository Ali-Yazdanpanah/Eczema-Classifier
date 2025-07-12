#!/usr/bin/env python3
"""
Local Eczema Classifier (PyTorch) - No Google Cloud Required
============================================================

This script provides a local version of the eczema classifier that can work
with already downloaded data or create synthetic data for testing.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

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
    
    return True

def check_local_data():
    """Check if there's any local data available."""
    print("\nChecking local data...")
    
    data_dir = Path('./scin_dataset')
    if not data_dir.exists():
        print("✗ No local data directory found")
        return False, 0
    
    # Check for metadata
    metadata_dir = data_dir / 'metadata'
    if not metadata_dir.exists():
        print("✗ No metadata directory found")
        return False, 0
    
    # Check for images
    images_dir = data_dir / 'images'
    if not images_dir.exists():
        print("✗ No images directory found")
        return False, 0
    
    # Count downloaded cases
    case_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
    print(f"✓ Found {len(case_dirs)} downloaded cases")
    
    return True, len(case_dirs)

def create_demo_mode():
    """Create a demo mode with synthetic data for testing."""
    print("\nCreating demo mode with synthetic data...")
    
    try:
        import numpy as np
        import pandas as pd
        from eczema_classifier import EczemaClassifier
        
        # Create synthetic data
        print("Generating synthetic ResNet features...")
        num_cases = 200
        feature_dim = 2048
        
        # Generate random features
        features = np.random.randn(num_cases, feature_dim)
        
        # Generate random labels (with some eczema cases)
        np.random.seed(42)
        labels = np.random.choice([0, 1], size=num_cases, p=[0.7, 0.3])
        
        # Create case IDs
        case_ids = [f"demo_case_{i}" for i in range(num_cases)]
        
        print(f"✓ Generated {num_cases} synthetic cases")
        print(f"✓ Eczema cases: {np.sum(labels)} ({100*np.sum(labels)/len(labels):.1f}%)")
        
        # Initialize classifier
        classifier = EczemaClassifier()
        
        # Set the synthetic data
        classifier.features = features
        classifier.labels = labels
        classifier.case_ids = case_ids
        
        # Train classifier
        print("\nTraining classifier on synthetic data...")
        results = classifier.train(test_size=0.2)
        
        # Print results
        print("\n" + "="*60)
        print("DEMO TRAINING RESULTS")
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
        classifier.save_model('eczema_classifier_demo_model')
        classifier.plot_results(results, save_path='eczema_classifier_demo_results.png')
        
        print("\n✓ Demo training completed successfully!")
        print("Note: This was trained on synthetic data for demonstration purposes.")
        
        return classifier, results
        
    except Exception as e:
        print(f"Error in demo mode: {e}")
        return None, None

def main():
    """Main function."""
    print("Local Eczema Classifier - Demo Mode")
    print("===================================")
    
    # Check environment
    if not setup_environment():
        return
    
    # Check for local data
    has_data, num_cases = check_local_data()
    
    if has_data and num_cases > 0:
        print(f"\nFound {num_cases} downloaded cases!")
        print("You can run the full classifier with:")
        print("  python3 run_eczema_classifier.py")
        print("\nOr continue with demo mode to test the system.")
        
        choice = input("\nWould you like to continue with demo mode? (y/n): ").strip().lower()
        if choice != 'y':
            print("Exiting...")
            return
    else:
        print("\nNo local data found. Running in demo mode...")
    
    # Run demo mode
    try:
        classifier, results = create_demo_mode()
        if classifier is not None:
            print("\n🎉 Demo completed successfully!")
            print("\nNext steps:")
            print("1. Set up Google Cloud authentication")
            print("2. Run: python3 run_eczema_classifier.py")
            print("3. Or download data manually and run the full classifier")
        else:
            print("\n❌ Demo failed. Please check your environment.")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Please check your PyTorch installation and try again.")

if __name__ == "__main__":
    main() 