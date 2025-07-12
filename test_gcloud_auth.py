#!/usr/bin/env python3
"""
Test Google Cloud Authentication
===============================

This script tests Google Cloud authentication and access to the SCIN dataset.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_gcloud_installation():
    """Test if gcloud CLI is installed."""
    print("Testing gcloud CLI installation...")
    
    try:
        import subprocess
        result = subprocess.run(['gcloud', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ gcloud CLI is installed")
            print(f"  Version: {result.stdout.split()[0]}")
            return True
        else:
            print("✗ gcloud CLI is not installed or not in PATH")
            return False
    except FileNotFoundError:
        print("✗ gcloud CLI not found")
        print("  Install from: https://cloud.google.com/sdk/docs/install")
        return False

def test_authentication():
    """Test Google Cloud authentication."""
    print("\nTesting Google Cloud authentication...")
    
    try:
        from google.cloud import storage
        
        # Try to create a client
        client = storage.Client()
        print("✓ Google Cloud authentication successful")
        
        # Test access to the SCIN dataset bucket
        bucket_name = 'dx-scin-public-data'
        try:
            bucket = client.bucket(bucket_name)
            # Try to list a few blobs to test access
            blobs = list(bucket.list_blobs(max_results=5))
            print(f"✓ Successfully accessed bucket: {bucket_name}")
            print(f"  Found {len(blobs)} objects (showing first 5)")
            return True
        except Exception as e:
            print(f"✗ Cannot access bucket {bucket_name}: {e}")
            print("  This might be a permissions issue")
            return False
            
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        print("\nTo fix this, run:")
        print("  gcloud auth login")
        print("  gcloud auth application-default login")
        return False

def test_dataset_access():
    """Test access to specific SCIN dataset files."""
    print("\nTesting SCIN dataset access...")
    
    try:
        from google.cloud import storage
        
        client = storage.Client()
        bucket = client.bucket('dx-scin-public-data')
        
        # Test access to metadata files
        test_files = [
            'dataset/scin_cases.csv',
            'dataset/scin_labels.csv'
        ]
        
        for file_path in test_files:
            try:
                blob = bucket.blob(file_path)
                # Check if file exists
                if blob.exists():
                    print(f"✓ Can access: {file_path}")
                else:
                    print(f"✗ File not found: {file_path}")
            except Exception as e:
                print(f"✗ Cannot access {file_path}: {e}")
        
        # Test access to images directory
        try:
            blobs = list(bucket.list_blobs(prefix='dataset/images/', max_results=5))
            print(f"✓ Can access images directory (found {len(blobs)} sample images)")
        except Exception as e:
            print(f"✗ Cannot access images directory: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset access test failed: {e}")
        return False

def setup_instructions():
    """Print setup instructions."""
    print("\n" + "="*60)
    print("GOOGLE CLOUD SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n1. Install Google Cloud SDK:")
    print("   https://cloud.google.com/sdk/docs/install")
    
    print("\n2. Initialize gcloud:")
    print("   gcloud init")
    
    print("\n3. Authenticate with your Google account:")
    print("   gcloud auth login")
    
    print("\n4. Set up application default credentials:")
    print("   gcloud auth application-default login")
    
    print("\n5. Verify setup:")
    print("   python3 test_gcloud_auth.py")
    
    print("\nNote: The SCIN dataset is publicly accessible, so you should")
    print("be able to access it with basic authentication.")

def main():
    """Main function."""
    print("Google Cloud Authentication Test")
    print("================================")
    
    # Test gcloud installation
    gcloud_ok = test_gcloud_installation()
    
    # Test authentication
    auth_ok = test_authentication()
    
    # Test dataset access
    dataset_ok = test_dataset_access()
    
    if gcloud_ok and auth_ok and dataset_ok:
        print("\n🎉 All tests passed! You're ready to use the SCIN dataset.")
        print("\nYou can now run:")
        print("  python3 run_eczema_classifier.py")
    else:
        print("\n❌ Some tests failed.")
        setup_instructions()
    
    return gcloud_ok and auth_ok and dataset_ok

if __name__ == "__main__":
    main() 