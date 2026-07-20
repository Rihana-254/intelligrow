"""
Disease Detection CNN Model
Uses TensorFlow/Keras to detect plant diseases from leaf images
Trained on PlantVillage dataset (https://data.mendeley.com/datasets/g6cm3v3wdp/5)
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import pickle
import warnings
warnings.filterwarnings('ignore')


class DiseaseDetectionCNN:
    """
    Convolutional Neural Network for detecting plant diseases from leaf images.
    
    Supports multiple architectures:
    - Custom CNN
    - MobileNetV2 (lightweight, fast inference)
    - ResNet50 (high accuracy, more parameters)
    - VGG16 (balanced performance)
    
    Target Classes (Plant Diseases):
    Based on PlantVillage Dataset:
    - Healthy
    - Early Blight
    - Late Blight
    - Leaf Mold
    - Septoria Leaf Spot
    - Spider Mites
    - Target Spot
    - Yellow Leaf Curl Virus
    - Mosaic Virus
    - Powdery Mildew
    - Bacterial Spot
    """
    
    def __init__(self, architecture='mobilenet', image_size=224, num_classes=38):
        """
        Initialize the disease detection CNN.
        
        Args:
            architecture (str): 'custom', 'mobilenet', 'resnet50', or 'vgg16'
            image_size (int): Input image size (224, 256, etc.)
            num_classes (int): Number of disease classes
        """
        self.architecture = architecture
        self.image_size = image_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = None
        self.disease_info = self._load_disease_info()
        
    def _load_disease_info(self):
        """
        Load disease information and treatments.
        
        Returns:
            dict: Disease names and treatment recommendations
        """
        return {
            'Healthy': {
                'description': 'Plant is healthy',
                'treatment': 'Continue regular maintenance and monitoring',
                'severity': 0
            },
            'Early Blight': {
                'description': 'Fungal disease causing concentric rings on leaves',
                'treatment': 'Remove affected leaves, apply copper fungicide, improve air circulation',
                'severity': 2,
                'symptoms': ['Brown spots with concentric rings', 'Yellow halo around spots', 'Lower leaves affected first']
            },
            'Late Blight': {
                'description': 'Severe fungal disease, spreads rapidly in cool wet conditions',
                'treatment': 'Isolate plant, apply chlorothalonil or mancozeb, reduce humidity',
                'severity': 3,
                'symptoms': ['Water-soaked spots', 'White mold on leaf undersides', 'Rapid spread']
            },
            'Leaf Mold': {
                'description': 'Fungal disease thriving in warm humid conditions',
                'treatment': 'Reduce humidity, improve ventilation, apply sulfur or copper fungicide',
                'severity': 2,
                'symptoms': ['Yellow spots on upper leaf', 'Olive-green mold on lower leaf', 'Yellowing leaves']
            },
            'Septoria Leaf Spot': {
                'description': 'Fungal disease causing small circular spots',
                'treatment': 'Remove infected leaves, apply fungicide, improve drainage',
                'severity': 2,
                'symptoms': ['Small circular spots', 'Gray centers with dark borders', 'Pycnidia in center']
            },
            'Powdery Mildew': {
                'description': 'White powdery fungal coating on leaves',
                'treatment': 'Apply sulfur, neem oil, or potassium bicarbonate fungicide',
                'severity': 1,
                'symptoms': ['White powder coating', 'Leaf curling', 'Stunted growth']
            },
            'Spider Mites': {
                'description': 'Tiny arachnid pests causing fine webbing',
                'treatment': 'Spray with insecticidal soap, neem oil, or miticide',
                'severity': 2,
                'symptoms': ['Fine webbing', 'Yellowing leaves', 'Speckling']
            },
            'Bacterial Spot': {
                'description': 'Bacterial infection causing dark lesions',
                'treatment': 'Remove infected leaves, apply copper bactericide, improve air circulation',
                'severity': 2,
                'symptoms': ['Dark water-soaked spots', 'Yellow halo', 'Oozing liquid']
            },
            'Mosaic Virus': {
                'description': 'Viral disease causing mottled appearance',
                'treatment': 'No cure, remove plant to prevent spread, control aphid vectors',
                'severity': 3,
                'symptoms': ['Mottled yellow and green', 'Distorted leaves', 'Stunted growth']
            }
        }
    
    def build_custom_cnn(self):
        """
        Build a custom CNN architecture from scratch.
        
        Returns:
            keras.Model: Compiled custom CNN model
        """
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         input_shape=(self.image_size, self.image_size, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Fully Connected Layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_mobilenet(self):
        """
        Build MobileNetV2-based transfer learning model (lightweight).
        
        Returns:
            keras.Model: Compiled MobileNetV2 model
        """
        base_model = MobileNetV2(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model weights
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            layers.BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            layers.BatchNormalization(),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_resnet50(self):
        """
        Build ResNet50-based transfer learning model (high accuracy).
        
        Returns:
            keras.Model: Compiled ResNet50 model
        """
        base_model = ResNet50(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model weights
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            layers.BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            layers.BatchNormalization(),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_vgg16(self):
        """
        Build VGG16-based transfer learning model (balanced).
        
        Returns:
            keras.Model: Compiled VGG16 model
        """
        base_model = VGG16(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model weights
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            layers.BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            layers.BatchNormalization(),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_model(self):
        """
        Build the model based on selected architecture.
        
        Returns:
            keras.Model: Compiled model
        """
        print(f"Building {self.architecture.upper()} model...")
        
        if self.architecture == 'custom':
            self.model = self.build_custom_cnn()
        elif self.architecture == 'mobilenet':
            self.model = self.build_mobilenet()
        elif self.architecture == 'resnet50':
            self.model = self.build_resnet50()
        elif self.architecture == 'vgg16':
            self.model = self.build_vgg16()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print("Model built successfully!")
        self.model.summary()
        
        return self.model
    
    def prepare_data_generators(self, train_dir, validation_split=0.2, batch_size=32):
        """
        Prepare data generators for training and validation.
        
        Args:
            train_dir (str): Path to training directory (structured as class folders)
            validation_split (float): Fraction of data to use for validation
            batch_size (int): Batch size for training
            
        Returns:
            tuple: (train_generator, validation_generator)
        """
        # Data augmentation for training
        train_augmentation = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            shear_range=0.15,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Only rescaling for validation
        validation_augmentation = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        train_generator = train_augmentation.flow_from_directory(
            train_dir,
            target_size=(self.image_size, self.image_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = validation_augmentation.flow_from_directory(
            train_dir,
            target_size=(self.image_size, self.image_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # Store class names
        self.class_names = list(train_generator.class_indices.keys())
        
        return train_generator, validation_generator
    
    def train(self, train_dir, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the disease detection model.
        
        Args:
            train_dir (str): Path to training data directory
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction for validation
        """
        if self.model is None:
            self.build_model()
        
        print("Preparing data...")
        train_gen, val_gen = self.prepare_data_generators(
            train_dir, 
            validation_split=validation_split,
            batch_size=batch_size
        )
        
        print(f"\nTraining {self.architecture} model...")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                f'ml_models/trained_models/disease_detector_{self.architecture}_best.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
    
    def predict_image(self, image_path, confidence_threshold=0.5):
        """
        Predict disease from a single image.
        
        Args:
            image_path (str): Path to image file
            confidence_threshold (float): Minimum confidence for prediction
            
        Returns:
            dict: Prediction results with disease info and recommendations
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Load and preprocess image
        img = load_img(image_path, target_size=(self.image_size, self.image_size))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_idx]
        confidence = float(predictions[0][predicted_idx])
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = {
            self.class_names[idx]: float(predictions[0][idx])
            for idx in top_3_idx
        }
        
        # Get disease information
        disease_info = self.disease_info.get(predicted_class, {})
        
        result = {
            'predicted_disease': predicted_class,
            'confidence': confidence,
            'confidence_threshold_met': confidence >= confidence_threshold,
            'top_3_predictions': top_3_predictions,
            'disease_info': disease_info,
            'recommendation': {
                'severity': disease_info.get('severity', 0),
                'treatment': disease_info.get('treatment', 'Consult with agricultural specialist'),
                'symptoms': disease_info.get('symptoms', []),
                'immediate_action': 'Monitor closely' if confidence < 0.7 else 'Apply treatment immediately'
            },
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path
        }
        
        return result
    
    def predict_batch(self, image_paths, confidence_threshold=0.5):
        """
        Predict diseases for multiple images.
        
        Args:
            image_paths (list): List of image file paths
            confidence_threshold (float): Minimum confidence for prediction
            
        Returns:
            list: List of prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path, confidence_threshold)
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        return results
    
    def evaluate(self, test_dir, batch_size=32):
        """
        Evaluate model on test data.
        
        Args:
            test_dir (str): Path to test directory
            batch_size (int): Batch size for evaluation
            
        Returns:
            dict: Evaluation metrics
        """
        test_augmentation = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_augmentation.flow_from_directory(
            test_dir,
            target_size=(self.image_size, self.image_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Evaluate
        eval_results = self.model.evaluate(test_generator, verbose=1)
        
        metrics = {
            'loss': float(eval_results[0]),
            'accuracy': float(eval_results[1]),
            'precision': float(eval_results[2]) if len(eval_results) > 2 else None,
            'recall': float(eval_results[3]) if len(eval_results) > 3 else None
        }
        
        print("\n" + "="*60)
        print(f"Model: {self.architecture.upper()}")
        print("="*60)
        print(f"Loss:      {metrics['loss']:.4f}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        if metrics['precision']:
            print(f"Precision: {metrics['precision']:.4f}")
        if metrics['recall']:
            print(f"Recall:    {metrics['recall']:.4f}")
        
        return metrics
    
    def plot_training_history(self, save_path=None):
        """
        Plot training and validation metrics.
        
        Args:
            save_path (str): Optional path to save the plot
        """
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Train')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
            axes[1, 0].set_title('Model Precision', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Train')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
            axes[1, 1].set_title('Model Recall', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath):
        """
        Save trained model to file.
        
        Args:
            filepath (str): Path to save the model
        """
        self.model.save(filepath)
        
        # Save metadata
        metadata = {
            'architecture': self.architecture,
            'image_size': self.image_size,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = filepath.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Model saved to {filepath}")
        print(f"Metadata saved to {metadata_path}")
    
    def load_model(self, filepath):
        """
        Load trained model from file.
        
        Args:
            filepath (str): Path to load the model from
        """
        self.model = keras.models.load_model(filepath)
        
        # Load metadata
        metadata_path = filepath.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.class_names = metadata.get('class_names')
        
        print(f"Model loaded from {filepath}")


def main():
    """
    Main function demonstrating the disease detection CNN.
    Note: Requires PlantVillage dataset to be downloaded and extracted.
    """
    print("="*70)
    print("DISEASE DETECTION CNN - TRAINING DEMONSTRATION")
    print("="*70)
    
    print("\nDataset Information:")
    print("Dataset: PlantVillage")
    print("URL: https://data.mendeley.com/datasets/g6cm3v3wdp/5")
    print("Classes: 38 different plant disease categories")
    print("\nDownload Instructions:")
    print("1. Visit: https://data.mendeley.com/datasets/g6cm3v3wdp/5")
    print("2. Download the dataset")
    print("3. Extract to: data/raw/plantvillage/")
    print("4. Structure should be:")
    print("   data/raw/plantvillage/")
    print("   ├── class1/")
    print("   │   ├── image1.jpg")
    print("   │   └── image2.jpg")
    print("   ├── class2/")
    print("   └── ...")
    
    # Initialize model
    detector = DiseaseDetectionCNN(
        architecture='mobilenet',
        image_size=224,
        num_classes=38
    )
    
    # Build model
    detector.build_model()
    
    print("\n" + "="*70)
    print("Ready to train on PlantVillage dataset!")
    print("="*70)
    print("\nUsage:")
    print("1. Prepare data in data/raw/plantvillage/")
    print("2. Call: detector.train('data/raw/plantvillage/', epochs=50)")
    print("3. Predict: detector.predict_image('path/to/leaf/image.jpg')")
    print("4. Save: detector.save_model('ml_models/trained_models/disease_detector.h5')")


if __name__ == '__main__':
    main()
