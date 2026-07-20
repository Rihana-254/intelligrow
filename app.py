"""
IntelliGrow Flask API

RESTful API for plant health monitoring with:
- Nutrient Deficiency Prediction (from sensor data)
- Disease Detection (from leaf images)

Endpoints:
    POST /api/v1/predict-nutrient - Predict nutrient deficiency from sensor data
    POST /api/v1/predict-disease - Predict disease from leaf image
    POST /api/v1/predict-batch-disease - Batch disease prediction
    GET /api/v1/health - Health check
    GET /api/v1/models/status - Model status

Usage:
    python app.py

Configuration:
    - PORT: 5000 (default)
    - DEBUG: False (production)
    - Models loaded on startup
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_models.nutrient_deficiency_predictor import NutrientDeficiencyPredictor
from ml_models.disease_detection_cnn import DiseaseDetectionCNN

# Load environment variables
load_dotenv()

# Flask app configuration
app = Flask(__name__)
CORS(app)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global model instances
nutrient_model = None
disease_model = None
models_loaded = False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    """Load ML models at startup"""
    global nutrient_model, disease_model, models_loaded
    
    try:
        logger.info("Loading ML models...")
        
        # Load nutrient predictor
        logger.info("Loading nutrient deficiency predictor...")
        nutrient_model_path = 'ml_models/trained_models/nutrient_predictor_rf.pkl'
        
        if os.path.exists(nutrient_model_path):
            nutrient_model = NutrientDeficiencyPredictor(model_type='random_forest')
            nutrient_model.load_model(nutrient_model_path)
            logger.info("✅ Nutrient model loaded successfully")
        else:
            logger.warning(f"⚠️  Nutrient model not found at {nutrient_model_path}")
            logger.info("Nutrient predictions will use untrained model")
            nutrient_model = NutrientDeficiencyPredictor(model_type='random_forest')
        
        # Load disease detector
        logger.info("Loading disease detection model...")
        disease_model_path = 'ml_models/trained_models/disease_detector_mobilenet.h5'
        
        # Check for any .h5 file in trained_models
        trained_models_dir = 'ml_models/trained_models'
        if os.path.exists(trained_models_dir):
            h5_files = [f for f in os.listdir(trained_models_dir) if f.endswith('.h5')]
            if h5_files:
                disease_model_path = os.path.join(trained_models_dir, h5_files[0])
        
        if os.path.exists(disease_model_path):
            disease_model = DiseaseDetectionCNN(architecture='mobilenet')
            disease_model.load_model(disease_model_path)
            logger.info("✅ Disease detection model loaded successfully")
        else:
            logger.warning(f"⚠️  Disease model not found at {disease_model_path}")
            logger.info("Disease detection will build model on first use")
            disease_model = DiseaseDetectionCNN(architecture='mobilenet')
            disease_model.build_model()
        
        models_loaded = True
        logger.info("✅ All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        models_loaded = False
        raise


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': models_loaded,
        'api_version': 'v1'
    }), 200


@app.route('/api/v1/models/status', methods=['GET'])
def models_status():
    """Check status of loaded models"""
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'nutrient_model': {
            'loaded': nutrient_model is not None,
            'type': 'random_forest',
            'features': ['pH', 'EC', 'Temperature', 'Humidity', 'Light_Intensity', 'Days_Since_Planting'],
            'classes': 11  # Number of nutrient classes
        },
        'disease_model': {
            'loaded': disease_model is not None,
            'type': 'mobilenet_v2',
            'architecture': 'transfer_learning',
            'classes': 38  # Number of disease classes
        }
    }), 200


# ============================================================================
# NUTRIENT PREDICTION ENDPOINTS
# ============================================================================

@app.route('/api/v1/predict-nutrient', methods=['POST'])
def predict_nutrient():
    """
    Predict nutrient deficiency from sensor data
    
    Request JSON:
    {
        "pH": 6.5,
        "EC": 1200,
        "Temperature": 24,
        "Humidity": 70,
        "Light_Intensity": 5000,
        "Days_Since_Planting": 45
    }
    
    Returns:
    {
        "predicted_deficiency": "Normal",
        "confidence": 0.95,
        "description": "No nutrient deficiency detected - Plant is healthy",
        "probabilities": {...},
        "timestamp": "2024-07-20T12:00:00"
    }
    """
    
    try:
        if not nutrient_model:
            return jsonify({
                'error': 'Nutrient model not loaded',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        # Validate request
        if not request.json:
            return jsonify({
                'error': 'No JSON data provided',
                'required_fields': ['pH', 'EC', 'Temperature', 'Humidity', 'Light_Intensity', 'Days_Since_Planting']
            }), 400
        
        data = request.json
        required_fields = ['pH', 'EC', 'Temperature', 'Humidity', 'Light_Intensity', 'Days_Since_Planting']
        
        # Check required fields
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields,
                'required_fields': required_fields
            }), 400
        
        # Validate data types
        try:
            sensor_data = {f: float(data[f]) for f in required_fields}
        except (ValueError, TypeError):
            return jsonify({
                'error': 'Invalid data types. All values must be numeric.',
                'provided': data
            }), 400
        
        # Validate ranges
        if not (4 <= sensor_data['pH'] <= 9):
            return jsonify({'error': 'pH must be between 4 and 9'}), 400
        if not (0 <= sensor_data['EC'] <= 3000):
            return jsonify({'error': 'EC must be between 0 and 3000'}), 400
        if not (-10 <= sensor_data['Temperature'] <= 50):
            return jsonify({'error': 'Temperature must be between -10 and 50°C'}), 400
        if not (0 <= sensor_data['Humidity'] <= 100):
            return jsonify({'error': 'Humidity must be between 0 and 100%'}), 400
        if not (0 <= sensor_data['Light_Intensity'] <= 100000):
            return jsonify({'error': 'Light Intensity must be between 0 and 100000 lux'}), 400
        if not (0 <= sensor_data['Days_Since_Planting'] <= 365):
            return jsonify({'error': 'Days Since Planting must be between 0 and 365'}), 400
        
        # Make prediction
        logger.info(f"Predicting nutrient deficiency for sensor data: {sensor_data}")
        prediction = nutrient_model.predict(sensor_data)
        
        logger.info(f"Prediction: {prediction['predicted_deficiency']} ({prediction['confidence']:.2%})")
        
        return jsonify(prediction), 200
        
    except Exception as e:
        logger.error(f"Error in nutrient prediction: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


# ============================================================================
# DISEASE PREDICTION ENDPOINTS
# ============================================================================

@app.route('/api/v1/predict-disease', methods=['POST'])
def predict_disease():
    """
    Predict plant disease from leaf image
    
    Request: multipart/form-data with 'image' file
    
    Returns:
    {
        "predicted_disease": "Late Blight",
        "confidence": 0.92,
        "description": "Severe fungal disease...",
        "recommendation": {...},
        "timestamp": "2024-07-20T12:00:00"
    }
    """
    
    try:
        if not disease_model:
            return jsonify({
                'error': 'Disease model not loaded',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'required': 'multipart/form-data with "image" field'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'allowed': list(ALLOWED_EXTENSIONS)
            }), 400
        
        # Save uploaded file
        filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Image saved to: {filepath}")
        
        # Get confidence threshold from query params
        confidence_threshold = request.args.get('threshold', 0.5, type=float)
        
        # Make prediction
        logger.info(f"Predicting disease from image: {filename}")
        prediction = disease_model.predict_image(filepath, confidence_threshold=confidence_threshold)
        
        logger.info(f"Prediction: {prediction['predicted_disease']} ({prediction['confidence']:.2%})")
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(prediction), 200
        
    except Exception as e:
        logger.error(f"Error in disease prediction: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/v1/predict-batch-disease', methods=['POST'])
def predict_batch_disease():
    """
    Batch predict diseases from multiple images
    
    Request: multipart/form-data with multiple 'images' files
    
    Returns:
    {
        "total_images": 3,
        "successful": 3,
        "failed": 0,
        "predictions": [{...}, {...}, {...}],
        "timestamp": "2024-07-20T12:00:00"
    }
    """
    
    try:
        if not disease_model:
            return jsonify({
                'error': 'Disease model not loaded',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        # Check if image files are present
        if 'images' not in request.files:
            return jsonify({
                'error': 'No image files provided',
                'required': 'multipart/form-data with "images" field'
            }), 400
        
        files = request.files.getlist('images')
        
        if not files or len(files) == 0:
            return jsonify({'error': 'No image files selected'}), 400
        
        # Get confidence threshold
        confidence_threshold = request.args.get('threshold', 0.5, type=float)
        
        # Process images
        predictions = []
        successful = 0
        failed = 0
        
        for file in files:
            try:
                if not allowed_file(file.filename):
                    failed += 1
                    predictions.append({
                        'filename': file.filename,
                        'error': 'Invalid file type'
                    })
                    continue
                
                # Save file
                filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Predict
                prediction = disease_model.predict_image(filepath, confidence_threshold=confidence_threshold)
                predictions.append(prediction)
                successful += 1
                
                # Clean up
                os.remove(filepath)
                
            except Exception as e:
                failed += 1
                logger.error(f"Error processing {file.filename}: {str(e)}")
                predictions.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        result = {
            'total_images': len(files),
            'successful': successful,
            'failed': failed,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch prediction completed: {successful}/{len(files)} successful")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in batch disease prediction: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'path': request.path,
        'method': request.method
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'error': 'Method not allowed',
        'method': request.method,
        'path': request.path
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500


# ============================================================================
# INITIALIZATION
# ============================================================================

@app.before_request
def before_request():
    """Before request processing"""
    logger.info(f"{request.method} {request.path}")


if __name__ == '__main__':
    # Load models on startup
    try:
        load_models()
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        logger.warning("Starting API without models...")
    
    # Get configuration from environment
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('FLASK_ENV', 'production') == 'development'
    
    # Print startup info
    print("\n" + "="*80)
    print("IntelliGrow API - Starting Server")
    print("="*80)
    print(f"Server: http://localhost:{PORT}")
    print(f"Debug Mode: {DEBUG}")
    print(f"Models Loaded: {models_loaded}")
    print("\nAvailable Endpoints:")
    print("  GET  /api/v1/health")
    print("  GET  /api/v1/models/status")
    print("  POST /api/v1/predict-nutrient")
    print("  POST /api/v1/predict-disease")
    print("  POST /api/v1/predict-batch-disease")
    print("="*80 + "\n")
    
    # Start server
    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=DEBUG,
        threaded=True
    )
