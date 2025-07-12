# SCIN Dataset Loader & Eczema Classifier (PyTorch)

A Python library to download and load the SCIN (Skin Condition Image Network) dataset from Google Cloud Storage to your local machine, with an interactive eczema classifier using ResNet features. Built with PyTorch for efficient deep learning feature extraction and comprehensive class imbalance handling.

## About SCIN Dataset

The SCIN dataset contains 5,000+ volunteer contributions (10,000+ images) of common dermatology conditions. It includes:

- **Images**: Up to 3 images per case
- **Metadata**: Demographic information, symptoms, body parts affected
- **Labels**: Dermatologist-annotated skin conditions and confidence scores
- **Diversity**: Representative of various skin types, ages, and conditions

## Features

- **Smart Download System**: Intelligent caching - only downloads new images, skips existing ones
- **Interactive Interface**: User-friendly menu system for case selection and training options
- **Download metadata and images from Google Cloud Storage**
- **Load and explore dataset statistics**
- **Display case images with metadata**
- **Eczema Classifier**: Logistic regression using ResNet features (PyTorch)
- **Binary Classification**: Detect eczema (1) vs other conditions (0)
- **Advanced Class Imbalance Handling**: Multiple methods including SMOTE, undersampling, and balanced weights
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Model Persistence**: Save and load trained models
- **GPU Acceleration**: Automatic CUDA support for faster feature extraction

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The requirements include PyTorch and torchvision. If you have a CUDA-capable GPU, you may want to install PyTorch with CUDA support for faster training:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install Additional Dependencies for Class Imbalance Handling

```bash
pip install imbalanced-learn
```

### 3. Set up Google Cloud Authentication

You need to authenticate with Google Cloud to access the dataset:

#### Option A: Using gcloud CLI (Recommended)
```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud auth application-default login
```

#### Option B: Using Service Account (Advanced)
```bash
# Set environment variable for service account key
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
```

## Quick Start

### Interactive Eczema Classifier

The easiest way to get started is using the interactive classifier:

```bash
# Run the interactive eczema classifier
python run_eczema_classifier.py
```

This will:
1. Check your environment and Google Cloud authentication
2. Show you dataset statistics
3. Let you choose how many cases to use (with smart download detection)
4. Let you select class imbalance handling method
5. Train the classifier and show results

### Basic Usage

```python
from scin_dataset_loader import SCINDatasetLoader

# Initialize loader
loader = SCINDatasetLoader(local_data_dir='./scin_dataset')

# Download metadata
loader.download_metadata()
loader.load_metadata()

# Get dataset statistics
stats = loader.get_dataset_stats()
print(f"Dataset has {stats['total_cases']} cases with {stats['total_images']} images")

# Smart download - only downloads new images
sample_cases = loader.cases_and_labels_df.sample(5)['case_id'].tolist()
loader.download_images(case_ids=sample_cases)

# Display a case
loader.display_case_images(sample_cases[0])
```

## Smart Download System

The system intelligently manages downloads:

- **Skip Existing Files**: If you already have images downloaded, it won't download them again
- **Incremental Downloads**: Only downloads additional cases if needed
- **Progress Tracking**: Shows download progress and statistics
- **Error Handling**: Gracefully handles network issues and missing files

**Example Output:**
```
Already downloaded: 1500 cases
✓ You have enough cases (1500). No download needed.
Using existing downloaded cases without downloading anything new.
```

## Eczema Classifier (PyTorch)

The eczema classifier uses ResNet50 features with logistic regression to detect eczema in skin images. Built with PyTorch for efficient GPU-accelerated feature extraction.

### Interactive Training

```bash
python run_eczema_classifier.py
```

**Training Options:**
1. **Quick test (100 cases)** - Fast testing
2. **Small dataset (500 cases)** - Good for development
3. **Medium dataset (1000 cases)** - Balanced performance/speed
4. **Large dataset (2000 cases)** - Better accuracy
5. **All eczema cases** - Focus on eczema detection
6. **All cases** - Maximum dataset
7. **Use only already downloaded** - No download needed
8. **Custom number** - Specify exact number

**Class Imbalance Handling:**
1. **Balanced Weights** - Adjusts class weights in logistic regression
2. **SMOTE** - Synthetic Minority Oversampling Technique
3. **Undersampling** - Reduces majority class samples
4. **SMOTEENN** - Combines SMOTE with Edited Nearest Neighbors

### Programmatic Usage

```python
from eczema_classifier import EczemaClassifier
from scin_dataset_loader import SCINDatasetLoader

# Initialize and train
loader = SCINDatasetLoader()
loader.download_metadata()
loader.load_metadata()

# Smart download - only if needed
sample_cases = loader.cases_and_labels_df.sample(500)['case_id'].tolist()
loader.download_images(case_ids=sample_cases)

# Train classifier with class imbalance handling
classifier = EczemaClassifier()
features, labels, case_ids = classifier.prepare_dataset(loader, max_cases=500)
results = classifier.train(test_size=0.2, balance_method='smote')

# Make predictions
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
prediction, probability = classifier.predict_case(image_paths)
print(f"Eczema: {prediction}, Confidence: {probability:.3f}")
```

### Classifier Features

- **ResNet50 Feature Extraction**: Uses pre-trained ResNet50 for robust feature extraction
- **PyTorch Implementation**: Efficient GPU-accelerated processing with automatic CUDA detection
- **Multi-Image Support**: Averages features across multiple images per case
- **Feature Scaling**: StandardScaler for optimal logistic regression performance
- **Stratified Splitting**: Maintains class balance in train/test splits
- **Advanced Class Imbalance Handling**: Multiple methods for handling imbalanced datasets
- **Comprehensive Evaluation**: Accuracy, AUC, confusion matrix, ROC curves
- **Feature Importance**: Identifies most important ResNet features

## Class Imbalance Handling

Medical datasets often have class imbalance. The classifier provides multiple methods:

### 1. Balanced Weights (Default)
- Adjusts class weights in logistic regression
- No data modification
- Fast and simple

### 2. SMOTE (Synthetic Minority Oversampling)
- Creates synthetic minority class samples
- Helps with severe imbalance
- May introduce noise

### 3. Undersampling
- Reduces majority class samples
- Faster training
- May lose important information

### 4. SMOTEENN (SMOTE + Edited Nearest Neighbors)
- Combines SMOTE with noise reduction
- Best for noisy datasets
- More computationally intensive

## API Reference

### SCINDatasetLoader

Main class for loading and managing the SCIN dataset.

#### Constructor
```python
SCINDatasetLoader(
    gcp_project='dx-scin-public',
    gcs_bucket_name='dx-scin-public-data',
    local_data_dir='./scin_dataset'
)
```

#### Methods

##### `download_metadata()`
Downloads the CSV metadata files from Google Cloud Storage.

##### `load_metadata()`
Loads the downloaded metadata into memory as pandas DataFrames.

##### `download_images(max_images=None, case_ids=None)`
Downloads images from GCS to local storage with smart caching.
- `max_images`: Maximum number of images to download
- `case_ids`: List of specific case IDs to download

##### `get_dataset_stats()`
Returns a dictionary with dataset statistics including:
- Total cases and images
- Distribution by sex, skin type, and conditions
- Top skin conditions

##### `get_cases_with_images()`
Returns cases that have at least one image downloaded locally.

##### `create_eczema_labels()`
Creates binary labels for eczema detection (1 for eczema, 0 for other).

##### `get_image_paths_for_case(case_id)`
Returns local image paths for a specific case.

### EczemaClassifier

Main class for training and using the eczema classifier.

#### Constructor
```python
EczemaClassifier(
    resnet_model='resnet50',
    feature_dim=2048,
    random_state=42,
    device=None
)
```

#### Methods

##### `prepare_dataset(loader, max_cases=None)`
Prepares the dataset by extracting ResNet features and creating labels.

##### `train(test_size=0.2, balance_method='balanced_weights')`
Trains the classifier with specified class imbalance handling.

##### `predict_case(image_paths)`
Predicts eczema for a new case using multiple images.

##### `save_model(model_path)`
Saves the trained model and scaler.

##### `load_model(model_path)`
Loads a previously saved model.

## Dataset Structure

The downloaded data is organized as follows:

```
scin_dataset/
├── metadata/
│   ├── scin_cases.csv      # Case metadata
│   └── scin_labels.csv     # Dermatologist labels
└── images/
    ├── case_id_1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── image3.jpg
    └── case_id_2/
        └── ...
```

## Key Metadata Fields

- **case_id**: Unique identifier for each case
- **age_group**: Age group of the patient
- **sex_at_birth**: Self-reported sex at birth
- **fitzpatrick_skin_type**: Fitzpatrick skin type classification
- **image_1_path, image_2_path, image_3_path**: Paths to case images
- **dermatologist_skin_condition_on_label_name**: Diagnosed skin conditions
- **weighted_skin_condition_label**: Weighted confidence scores

## Troubleshooting

### Google Cloud Authentication Issues

If you encounter authentication problems:

1. **Geographic Restrictions**: Some regions may have restricted access
   - Try using a VPN
   - Contact your network administrator

2. **Browser Launch Issues**: In WSL or headless environments
   - Use manual authentication: `gcloud auth login --no-launch-browser`
   - Follow the provided URL manually

3. **Timeout Issues**: Network connectivity problems
   - Check your internet connection
   - Try again later
   - Use the demo mode if available

### Performance Tips

1. **Use GPU**: Install PyTorch with CUDA support for faster training
2. **Batch Processing**: Process images in batches for memory efficiency
3. **Smart Downloads**: Use existing downloaded cases when possible
4. **Class Imbalance**: Try different balance methods for better performance

## Examples

### Interactive Training
```bash
# Start interactive training
python run_eczema_classifier.py

# Follow the prompts to:
# 1. Choose number of cases
# 2. Select class imbalance method
# 3. View training results
```

### Smart Download Example
```python
# Download specific cases (only if not already downloaded)
case_ids = ['case_123', 'case_456']
loader.download_images(case_ids=case_ids)
# Output: "Downloaded 0 new images, skipped 6 existing images"
```

### Train with Class Imbalance Handling
```python
# Train with SMOTE for severe class imbalance
classifier = EczemaClassifier()
features, labels, case_ids = classifier.prepare_dataset(loader, max_cases=1000)
results = classifier.train(test_size=0.2, balance_method='smote')

# View results
print(f"Accuracy: {results['accuracy']:.3f}")
print(f"AUC: {results['auc']:.3f}")
print(f"Precision: {results['precision']:.3f}")
print(f"Recall: {results['recall']:.3f}")
```

## License

This project uses the SCIN dataset which is subject to its own license terms. Please refer to the dataset documentation for licensing information.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 