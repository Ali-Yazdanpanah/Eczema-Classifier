#!/usr/bin/env python3
"""
Test PyTorch Setup for Eczema Classifier
========================================

This script tests the PyTorch installation and GPU availability.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_pytorch_setup():
    """Test PyTorch installation and GPU availability."""
    print("Testing PyTorch Setup")
    print("=====================")
    
    try:
        import torch
        import torchvision
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ Torchvision version: {torchvision.__version__}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"✓ GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠ CUDA is not available - will use CPU")
        
        # Test ResNet model loading
        print("\nTesting ResNet model loading...")
        import torch.nn as nn
        import torchvision.models as models
        
        # Load ResNet50
        resnet = models.resnet50(pretrained=True)
        resnet = nn.Sequential(*list(resnet.children())[:-1])
        resnet.eval()
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        resnet.to(device)
        print(f"✓ ResNet50 loaded successfully on {device}")
        
        # Test image processing
        print("\nTesting image processing...")
        import torchvision.transforms as transforms
        from PIL import Image
        import numpy as np
        
        # Create a dummy image
        dummy_img = Image.new('RGB', (224, 224), color='red')
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Process image
        img_tensor = transform(dummy_img).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = resnet(img_tensor)
            features = features.squeeze().cpu().numpy()
        
        print(f"✓ Feature extraction successful")
        print(f"✓ Feature dimension: {features.shape}")
        
        print("\n✓ All PyTorch tests passed!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install PyTorch: pip install torch torchvision")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_other_dependencies():
    """Test other required dependencies."""
    print("\nTesting Other Dependencies")
    print("==========================")
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
        
        # Check for NumPy 2.x compatibility issue
        if np.__version__.startswith('2.'):
            print("⚠ Warning: NumPy 2.x detected - may cause pandas compatibility issues")
            print("   Consider running: python3 fix_dependencies.py")
        
        import pandas as pd
        print(f"✓ Pandas version: {pd.__version__}")
        
        import sklearn
        print(f"✓ Scikit-learn version: {sklearn.__version__}")
        
        import joblib
        print(f"✓ Joblib available")
        
        import matplotlib
        print(f"✓ Matplotlib version: {matplotlib.__version__}")
        
        import seaborn
        print(f"✓ Seaborn available")
        
        print("✓ All dependencies available!")
        return True
        
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False
    except ValueError as e:
        if "numpy.dtype size changed" in str(e):
            print("✗ NumPy/pandas compatibility issue detected!")
            print("   This is caused by NumPy 2.x incompatibility with pandas")
            print("   Run: python3 fix_dependencies.py")
            return False
        else:
            print(f"✗ Error: {e}")
            return False


def main():
    """Main function."""
    print("PyTorch Setup Test for Eczema Classifier")
    print("========================================")
    
    # Test PyTorch
    pytorch_ok = test_pytorch_setup()
    
    # Test other dependencies
    deps_ok = test_other_dependencies()
    
    if pytorch_ok and deps_ok:
        print("\n🎉 All tests passed! You're ready to run the eczema classifier.")
        print("\nNext steps:")
        print("1. Set up Google Cloud authentication")
        print("2. Run: python run_eczema_classifier.py")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        return False
    
    return True


if __name__ == "__main__":
    main() 