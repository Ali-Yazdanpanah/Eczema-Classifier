#!/usr/bin/env python3
"""
Fix Dependencies for Eczema Classifier
======================================

This script fixes the NumPy/pandas compatibility issue by installing
compatible versions of the packages.
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return the result."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_current_versions():
    """Check current package versions."""
    print("Checking current package versions...")
    
    packages = ['numpy', 'pandas', 'torch', 'torchvision']
    
    for package in packages:
        success, stdout, stderr = run_command(f"python3 -c \"import {package}; print('{package}:', {package}.__version__)\"")
        if success:
            print(f"✓ {stdout.strip()}")
        else:
            print(f"✗ {package}: Not installed or error")

def fix_numpy_pandas_compatibility():
    """Fix NumPy/pandas compatibility issue."""
    print("\nFixing NumPy/pandas compatibility...")
    
    # Uninstall current NumPy and pandas
    print("Uninstalling current NumPy and pandas...")
    run_command("pip3 uninstall numpy pandas -y")
    
    # Install compatible versions
    print("Installing compatible NumPy and pandas versions...")
    
    # Install NumPy 1.24.x (last stable 1.x version)
    success, stdout, stderr = run_command("pip3 install 'numpy>=1.24.0,<2.0.0'")
    if success:
        print("✓ NumPy installed successfully")
    else:
        print(f"✗ NumPy installation failed: {stderr}")
        return False
    
    # Install pandas 2.x
    success, stdout, stderr = run_command("pip3 install 'pandas>=2.0.0,<3.0.0'")
    if success:
        print("✓ Pandas installed successfully")
    else:
        print(f"✗ Pandas installation failed: {stderr}")
        return False
    
    return True

def install_other_dependencies():
    """Install other required dependencies."""
    print("\nInstalling other dependencies...")
    
    dependencies = [
        "Pillow>=8.3.0",
        "matplotlib>=3.4.0", 
        "seaborn>=0.11.0",
        "google-cloud-storage>=2.0.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.1.0"
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        success, stdout, stderr = run_command(f"pip3 install '{dep}'")
        if success:
            print(f"✓ {dep} installed")
        else:
            print(f"✗ {dep} failed: {stderr}")

def install_pytorch():
    """Install PyTorch with CUDA support if available."""
    print("\nInstalling PyTorch...")
    
    # Check if CUDA is available
    success, stdout, stderr = run_command("nvidia-smi")
    if success:
        print("CUDA detected, installing PyTorch with CUDA support...")
        # Install PyTorch with CUDA 11.8 (most common)
        success, stdout, stderr = run_command("pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    else:
        print("CUDA not detected, installing CPU-only PyTorch...")
        success, stdout, stderr = run_command("pip3 install torch torchvision")
    
    if success:
        print("✓ PyTorch installed successfully")
    else:
        print(f"✗ PyTorch installation failed: {stderr}")
        return False
    
    return True

def test_imports():
    """Test if all packages can be imported successfully."""
    print("\nTesting imports...")
    
    test_code = """
import numpy as np
import pandas as pd
import torch
import torchvision
import sklearn
import joblib
import matplotlib
import seaborn
from PIL import Image

print('All imports successful!')
print(f'NumPy: {np.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'Torchvision: {torchvision.__version__}')
"""
    
    success, stdout, stderr = run_command(f"python3 -c \"{test_code}\"")
    if success:
        print(stdout)
        return True
    else:
        print(f"✗ Import test failed: {stderr}")
        return False

def main():
    """Main function."""
    print("Fix Dependencies for Eczema Classifier")
    print("======================================")
    
    # Check current versions
    check_current_versions()
    
    # Fix NumPy/pandas compatibility
    if not fix_numpy_pandas_compatibility():
        print("Failed to fix NumPy/pandas compatibility")
        return False
    
    # Install other dependencies
    install_other_dependencies()
    
    # Install PyTorch
    if not install_pytorch():
        print("Failed to install PyTorch")
        return False
    
    # Test imports
    if not test_imports():
        print("Import test failed")
        return False
    
    print("\n🎉 All dependencies fixed successfully!")
    print("\nYou can now run:")
    print("python3 test_pytorch_setup.py")
    
    return True

if __name__ == "__main__":
    main() 