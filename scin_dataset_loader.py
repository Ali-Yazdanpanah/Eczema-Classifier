#!/usr/bin/env python3
"""
SCIN Dataset Loader
===================

This script downloads and loads the SCIN (Skin Condition Image Network) dataset
from Google Cloud Storage to your local machine.
"""

import os
import io
import collections
import hashlib
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from google.cloud import storage
import tensorflow as tf


class SCINDatasetLoader:
    """
    A class to load and manage the SCIN dataset locally.
    """
    
    def __init__(self, 
                 gcp_project: str = 'dx-scin-public',
                 gcs_bucket_name: str = 'dx-scin-public-data',
                 local_data_dir: str = './scin_dataset'):
        """
        Initialize the SCIN dataset loader.
        """
        self.gcp_project = gcp_project
        self.gcs_bucket_name = gcs_bucket_name
        self.local_data_dir = Path(local_data_dir)
        
        # Create local directories
        self.local_data_dir.mkdir(exist_ok=True)
        (self.local_data_dir / 'images').mkdir(exist_ok=True)
        (self.local_data_dir / 'metadata').mkdir(exist_ok=True)
        
        # Initialize GCS client
        self.storage_client = storage.Client(self.gcp_project)
        self.bucket = self.storage_client.bucket(self.gcs_bucket_name)
        
        # Dataset file paths
        self.cases_csv = 'dataset/scin_cases.csv'
        self.labels_csv = 'dataset/scin_labels.csv'
        self.images_dir = 'dataset/images/'
        
        # Column names
        self.image_path_columns = ['image_1_path', 'image_2_path', 'image_3_path']
        self.weighted_skin_condition_label = "weighted_skin_condition_label"
        self.skin_condition_label = "dermatologist_skin_condition_on_label_name"
        
        # Data storage
        self.cases_df = None
        self.cases_and_labels_df = None
        
    def download_metadata(self) -> None:
        """
        Download the metadata CSV files from GCS.
        """
        print("Downloading metadata files...")
        
        # Download cases CSV
        cases_blob = self.bucket.blob(self.cases_csv)
        cases_local_path = self.local_data_dir / 'metadata' / 'scin_cases.csv'
        cases_blob.download_to_filename(cases_local_path)
        print(f"Downloaded cases metadata to {cases_local_path}")
        
        # Download labels CSV
        labels_blob = self.bucket.blob(self.labels_csv)
        labels_local_path = self.local_data_dir / 'metadata' / 'scin_labels.csv'
        labels_blob.download_to_filename(labels_local_path)
        print(f"Downloaded labels metadata to {labels_local_path}")
        
    def load_metadata(self) -> None:
        """
        Load metadata from local CSV files.
        """
        print("Loading metadata...")
        
        # Load cases metadata
        cases_path = self.local_data_dir / 'metadata' / 'scin_cases.csv'
        self.cases_df = pd.read_csv(cases_path, dtype={'case_id': str})
        self.cases_df['case_id'] = self.cases_df['case_id'].astype(str)
        
        # Load labels metadata
        labels_path = self.local_data_dir / 'metadata' / 'scin_labels.csv'
        labels_df = pd.read_csv(labels_path, dtype={'case_id': str})
        labels_df['case_id'] = labels_df['case_id'].astype(str)
        
        # Merge cases and labels
        self.cases_and_labels_df = pd.merge(self.cases_df, labels_df, on='case_id')
        
        print(f"Loaded {len(self.cases_and_labels_df)} cases with metadata")
        
    def download_images(self, max_images: Optional[int] = None, 
                       case_ids: Optional[List[str]] = None) -> None:
        """
        Download images from GCS to local storage.
        """
        print("Downloading images...")
        
        # Filter cases if specified
        if case_ids:
            df = self.cases_and_labels_df[self.cases_and_labels_df['case_id'].isin(case_ids)]
        else:
            df = self.cases_and_labels_df
        
        # Limit number of images if specified
        if max_images:
            df = df.head(max_images)
        
        downloaded_count = 0
        skipped_count = 0
        total_images = 0
        
        for _, row in df.iterrows():
            case_id = row['case_id']
            
            # Create case directory
            case_dir = self.local_data_dir / 'images' / case_id
            case_dir.mkdir(exist_ok=True)
            
            # Download images for this case
            for col in self.image_path_columns:
                image_path = row[col]
                if pd.notna(image_path) and isinstance(image_path, str):
                    total_images += 1
                    
                    try:
                        # Download image from GCS
                        blob = self.bucket.blob(image_path)
                        image_filename = Path(image_path).name
                        local_image_path = case_dir / image_filename
                        
                        if not local_image_path.exists():
                            blob.download_to_filename(local_image_path)
                            downloaded_count += 1
                            
                            if downloaded_count % 100 == 0:
                                print(f"Downloaded {downloaded_count} images...")
                        else:
                            skipped_count += 1
                                
                    except Exception as e:
                        print(f"Error downloading {image_path}: {e}")
        
        print(f"Downloaded {downloaded_count} new images, skipped {skipped_count} existing images")
        print(f"Total images processed: {total_images}")
        
    def get_dataset_stats(self) -> Dict:
        """
        Get statistics about the loaded dataset.
        """
        if self.cases_and_labels_df is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        
        df = self.cases_and_labels_df
        
        stats = {
            'total_cases': len(df),
            'total_images': df[self.image_path_columns].count().sum(),
            'cases_with_1_image': df['image_1_path'].count(),
            'cases_with_2_images': df['image_2_path'].count(),
            'cases_with_3_images': df['image_3_path'].count(),
        }
        
        # Sex distribution
        sex_dist = df['sex_at_birth'].value_counts()
        stats['sex_distribution'] = sex_dist.to_dict()
        
        # Fitzpatrick skin type distribution
        fst_dist = df['fitzpatrick_skin_type'].value_counts()
        stats['fitzpatrick_skin_type_distribution'] = fst_dist.to_dict()
        
        # Top skin conditions
        condition_counter = collections.Counter()
        for entry in df[self.skin_condition_label].dropna():
            if isinstance(entry, str):
                try:
                    conditions = eval(entry)
                    condition_counter.update(conditions)
                except:
                    pass
        
        stats['top_skin_conditions'] = dict(condition_counter.most_common(10))
        
        return stats
    
    def create_eczema_labels(self) -> pd.Series:
        """
        Create binary labels for eczema detection.
        1 if case has eczema, 0 otherwise.
        """
        if self.cases_and_labels_df is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        
        eczema_labels = []
        
        for _, row in self.cases_and_labels_df.iterrows():
            conditions = row[self.skin_condition_label]
            has_eczema = 0
            
            if pd.notna(conditions) and isinstance(conditions, str):
                try:
                    condition_list = eval(conditions)
                    if 'Eczema' in condition_list:
                        has_eczema = 1
                except:
                    pass
            
            eczema_labels.append(has_eczema)
        
        return pd.Series(eczema_labels, index=self.cases_and_labels_df.index)
    
    def get_image_paths_for_case(self, case_id: str) -> List[str]:
        """
        Get local image paths for a specific case.
        """
        case_dir = self.local_data_dir / 'images' / case_id
        
        if not case_dir.exists():
            return []
        
        # Get all image files in the case directory
        image_files = list(case_dir.glob('*.jpg')) + list(case_dir.glob('*.png'))
        return [str(f) for f in image_files]
    
    def get_cases_with_images(self) -> pd.DataFrame:
        """
        Get cases that have at least one image downloaded locally.
        """
        if self.cases_and_labels_df is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        
        cases_with_images = []
        
        for _, row in self.cases_and_labels_df.iterrows():
            case_id = row['case_id']
            image_paths = self.get_image_paths_for_case(case_id)
            
            if image_paths:  # If case has at least one image
                cases_with_images.append(row)
        
        return pd.DataFrame(cases_with_images)


def main():
    """
    Main function to demonstrate usage of the SCIN dataset loader.
    """
    print("SCIN Dataset Loader")
    print("==================")
    
    # Initialize loader
    loader = SCINDatasetLoader()
    
    # Download and load metadata
    loader.download_metadata()
    loader.load_metadata()
    
    # Print dataset statistics
    stats = loader.get_dataset_stats()
    print(f"\nDataset Statistics:")
    print(f"Total cases: {stats['total_cases']}")
    print(f"Total images: {stats['total_images']}")
    
    # Create eczema labels
    eczema_labels = loader.create_eczema_labels()
    eczema_count = eczema_labels.sum()
    print(f"Cases with eczema: {eczema_count} ({100*eczema_count/len(eczema_labels):.1f}%)")
    
    print("\nDataset loaded successfully!")


if __name__ == "__main__":
    main() 