"""
Nutrient Deficiency Predictor
Uses Random Forest and Gradient Boosting models to predict nutrient deficiencies
based on sensor data (pH, EC, Temperature, Humidity, etc.)
"""

import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class NutrientDeficiencyPredictor:
    """
    Predicts nutrient deficiencies in hydroponic systems based on sensor readings.
    
    Features:
    - pH level (0-14)
    - EC (Electrical Conductivity) in μS/cm (0-2000)
    - Water Temperature in °C (-10 to 50)
    - Humidity in % (0-100)
    - Light Intensity in lux (0-10000)
    - Days since planting (0-365)
    
    Target Classes (Nutrient Deficiencies):
    - N: Nitrogen deficiency
    - P: Phosphorus deficiency
    - K: Potassium deficiency
    - Mg: Magnesium deficiency
    - Ca: Calcium deficiency
    - S: Sulfur deficiency
    - Fe: Iron deficiency
    - Mn: Manganese deficiency
    - Zn: Zinc deficiency
    - B: Boron deficiency
    - Normal: No deficiency detected
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the nutrient predictor.
        
        Args:
            model_type (str): 'random_forest' or 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = ['pH', 'EC', 'Temperature', 'Humidity', 'Light_Intensity', 'Days_Since_Planting']
        self.nutrient_classes = ['N', 'P', 'K', 'Mg', 'Ca', 'S', 'Fe', 'Mn', 'Zn', 'B', 'Normal']
        self.nutrient_descriptions = {
            'N': 'Nitrogen Deficiency - Yellowing of lower leaves, stunted growth',
            'P': 'Phosphorus Deficiency - Purple/dark red coloration, delayed flowering',
            'K': 'Potassium Deficiency - Brown leaf edges, weak stems',
            'Mg': 'Magnesium Deficiency - Interveinal chlorosis on older leaves',
            'Ca': 'Calcium Deficiency - Tip burn on new leaves, distorted growth',
            'S': 'Sulfur Deficiency - Uniform yellowing of young leaves',
            'Fe': 'Iron Deficiency - Interveinal chlorosis on young leaves',
            'Mn': 'Manganese Deficiency - Interveinal chlorosis, dark spots',
            'Zn': 'Zinc Deficiency - Stunted growth, small leaves',
            'B': 'Boron Deficiency - Thick, brittle leaves, distorted growth',
            'Normal': 'No nutrient deficiency detected - Plant is healthy'
        }
        
    def generate_synthetic_data(self, n_samples=1000, random_state=42):
        """
        Generate synthetic training data for nutrient deficiencies.
        In production, this would be replaced with real sensor data.
        
        Args:
            n_samples (int): Number of samples to generate
            random_state (int): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Feature data
            np.ndarray: Labels
        """
        np.random.seed(random_state)
        
        data = []
        labels = []
        
        for _ in range(n_samples):
            # Generate random sensor readings
            ph = np.random.uniform(5.5, 7.5)
            ec = np.random.uniform(800, 1600)
            temp = np.random.uniform(18, 28)
            humidity = np.random.uniform(50, 90)
            light = np.random.uniform(3000, 8000)
            days = np.random.uniform(1, 180)
            
            # Assign nutrient deficiency based on sensor ranges
            if ph < 5.8:
                label = np.random.choice(['Fe', 'Mn', 'Zn', 'Normal'], p=[0.4, 0.3, 0.2, 0.1])
            elif ph > 7.2:
                label = np.random.choice(['Fe', 'Mn', 'B', 'Normal'], p=[0.3, 0.3, 0.2, 0.2])
            elif ec < 900:
                label = np.random.choice(['N', 'P', 'K', 'Normal'], p=[0.3, 0.2, 0.3, 0.2])
            elif ec > 1500:
                label = np.random.choice(['K', 'Ca', 'Mg', 'Normal'], p=[0.2, 0.3, 0.3, 0.2])
            elif temp < 20:
                label = np.random.choice(['P', 'K', 'Normal'], p=[0.3, 0.3, 0.4])
            elif temp > 26:
                label = np.random.choice(['Ca', 'B', 'Normal'], p=[0.2, 0.3, 0.5])
            elif light < 4000:
                label = np.random.choice(['N', 'S', 'Normal'], p=[0.3, 0.2, 0.5])
            else:
                label = 'Normal'
            
            data.append([ph, ec, temp, humidity, light, days])
            labels.append(label)
        
        df = pd.DataFrame(data, columns=self.feature_names)
        return df, np.array(labels)
    
    def train(self, X_train, y_train):
        """
        Train the nutrient deficiency predictor model.
        
        Args:
            X_train (pd.DataFrame or np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        print(f"Training {self.model_type} model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize and train model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_train_scaled, y_train)
        print(f"Model training completed!")
    
    def predict(self, sensor_data):
        """
        Predict nutrient deficiency from sensor readings.
        
        Args:
            sensor_data (dict or np.ndarray): Sensor readings
            Format: {'pH': float, 'EC': float, 'Temperature': float, 
                     'Humidity': float, 'Light_Intensity': float, 'Days_Since_Planting': float}
        
        Returns:
            dict: Prediction results with confidence scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert dict to array if needed
        if isinstance(sensor_data, dict):
            X = np.array([[sensor_data[feat] for feat in self.feature_names]])
        else:
            X = np.array(sensor_data).reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probabilities
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Create result dictionary
        result = {
            'predicted_deficiency': prediction,
            'description': self.nutrient_descriptions.get(prediction, 'Unknown deficiency'),
            'confidence': float(probabilities[list(self.model.classes_).index(prediction)]),
            'all_probabilities': {
                nutrient: float(prob) 
                for nutrient, prob in zip(self.model.classes_, probabilities)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def predict_batch(self, sensor_data_list):
        """
        Predict nutrient deficiencies for multiple samples.
        
        Args:
            sensor_data_list (list): List of sensor data dictionaries or arrays
        
        Returns:
            list: List of prediction results
        """
        results = []
        for sensor_data in sensor_data_list:
            results.append(self.predict(sensor_data))
        return results
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test (pd.DataFrame or np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        print("\n" + "="*60)
        print(f"Model: {self.model_type.upper()}")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        return metrics
    
    def get_feature_importance(self):
        """
        Get feature importance scores from the trained model.
        
        Returns:
            dict: Feature importance rankings
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importances = self.model.feature_importances_
        feature_importance = {
            feat: float(imp) 
            for feat, imp in zip(self.feature_names, importances)
        }
        
        # Sort by importance
        sorted_importance = dict(sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return sorted_importance
    
    def save_model(self, filepath):
        """
        Save trained model to file.
        
        Args:
            filepath (str): Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'nutrient_classes': self.nutrient_classes
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model from file.
        
        Args:
            filepath (str): Path to load the model from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.model_type = data['model_type']
            self.feature_names = data['feature_names']
            self.nutrient_classes = data['nutrient_classes']
        print(f"Model loaded from {filepath}")
    
    def plot_feature_importance(self, save_path=None):
        """
        Plot feature importance visualization.
        
        Args:
            save_path (str): Optional path to save the plot
        """
        importance = self.get_feature_importance()
        
        plt.figure(figsize=(10, 6))
        features = list(importance.keys())
        scores = list(importance.values())
        
        plt.barh(features, scores, color='steelblue')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Feature Importance - {self.model_type.upper()}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


def main():
    """
    Main function to demonstrate the nutrient deficiency predictor.
    """
    print("="*70)
    print("NUTRIENT DEFICIENCY PREDICTOR - TRAINING DEMONSTRATION")
    print("="*70)
    
    # Initialize predictor
    predictor = NutrientDeficiencyPredictor(model_type='random_forest')
    
    # Generate synthetic training data
    print("\nGenerating synthetic training data...")
    X, y = predictor.generate_synthetic_data(n_samples=1000)
    print(f"Generated {len(X)} samples with {len(X.columns)} features")
    print(f"Classes: {np.unique(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    predictor.train(X_train, y_train)
    
    # Evaluate model
    metrics = predictor.evaluate(X_test, y_test)
    
    # Display feature importance
    print("\nFeature Importance:")
    importance = predictor.get_feature_importance()
    for feat, score in importance.items():
        print(f"  {feat}: {score:.4f}")
    
    # Make sample predictions
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)
    
    sample_data = [
        {
            'pH': 5.5,
            'EC': 850,
            'Temperature': 22,
            'Humidity': 70,
            'Light_Intensity': 5000,
            'Days_Since_Planting': 45
        },
        {
            'pH': 6.8,
            'EC': 1400,
            'Temperature': 24,
            'Humidity': 65,
            'Light_Intensity': 6000,
            'Days_Since_Planting': 60
        },
        {
            'pH': 7.5,
            'EC': 1100,
            'Temperature': 26,
            'Humidity': 75,
            'Light_Intensity': 4500,
            'Days_Since_Planting': 80
        }
    ]
    
    for i, data in enumerate(sample_data, 1):
        print(f"\nSample {i}:")
        print(f"  Input: pH={data['pH']}, EC={data['EC']}, Temp={data['Temperature']}°C")
        
        prediction = predictor.predict(data)
        print(f"  Prediction: {prediction['predicted_deficiency']}")
        print(f"  Description: {prediction['description']}")
        print(f"  Confidence: {prediction['confidence']:.2%}")
    
    # Save model
    model_path = 'ml_models/trained_models/nutrient_predictor_rf.pkl'
    predictor.save_model(model_path)
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
