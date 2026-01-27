"""
CNN KEYPOINT DETECTION MODEL - TRAINING SCRIPT
===============================================
This script trains a Convolutional Neural Network (CNN) to detect
21 body keypoints from 2D character images.

Author: Angelo Tristan Sinohin
Thesis: Animatics: 2D to 3D Character Rigging using CNN

Algorithm: Convolutional Neural Network (CNN)
Dataset: 2,191 labeled images (human + anime)
Output: 21 keypoints (x, y coordinates) = 42 values
"""

import os
import json
import numpy as np
import cv2
import time
from datetime import datetime
from pathlib import Path

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks # type: ignore
from sklearn.model_selection import train_test_split

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    # Data settings
    "human_labels": "./labeled_data/all_labels.json",
    "anime_labels": "./labeled_data_anime/all_labels.json",
    "image_size": (256, 256),  # Resize all images to this size
    "num_keypoints": 21,
    
    # Training settings
    "batch_size": 16,
    "epochs": 100,
    "learning_rate": 0.001,
    "validation_split": 0.15,  # 15% for validation
    "test_split": 0.10,        # 10% for testing
    
    # Output settings
    "model_save_path": "./trained_model",
    "training_log_path": "./training_logs",
}

KEYPOINT_NAMES = [
    'head_top', 'head_center', 'neck', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'chest_center',
    'waist_center', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_hand', 'right_hand', 'left_foot', 'right_foot'
]


class KeypointDataLoader:
    """Handles loading and preprocessing of training data."""
    
    def __init__(self, config):
        self.config = config
        self.image_size = config["image_size"]
        self.num_keypoints = config["num_keypoints"]
        
    def load_labels(self, labels_path):
        """Load labels from JSON file."""
        with open(labels_path, 'r') as f:
            data = json.load(f)
        return data['labels']
    
    def preprocess_image(self, image_path):
        """Load and preprocess a single image."""
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        original_h, original_w = image.shape[:2]
        
        # Resize image
        image = cv2.resize(image, self.image_size)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image, (original_w, original_h)
    
    def normalize_keypoints(self, keypoints, original_size, target_size):
        """Normalize keypoints to [0, 1] range based on target image size."""
        original_w, original_h = original_size
        target_w, target_h = target_size
        
        normalized = []
        for kp in keypoints:
            # Scale keypoint to new image size, then normalize to [0, 1]
            x = (kp['x'] / original_w)  # Normalize to [0, 1]
            y = (kp['y'] / original_h)  # Normalize to [0, 1]
            
            # Clip to valid range
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            
            normalized.extend([x, y])
        
        return normalized
    
    def load_dataset(self):
        """Load complete dataset from both human and anime labels."""
        print("\n" + "="*50)
        print("LOADING DATASET")
        print("="*50)
        
        all_images = []
        all_keypoints = []
        failed_loads = []
        
        # Load both datasets
        datasets = [
            ("Human", self.config["human_labels"]),
            ("Anime", self.config["anime_labels"])
        ]
        
        for dataset_name, labels_path in datasets:
            if not os.path.exists(labels_path):
                print(f"WARNING: {labels_path} not found, skipping...")
                continue
                
            print(f"\nLoading {dataset_name} dataset...")
            labels = self.load_labels(labels_path)
            
            loaded = 0
            for label_data in labels:
                image_path = label_data.get('filepath')
                
                if not image_path or not os.path.exists(image_path):
                    failed_loads.append(image_path)
                    continue
                
                # Load and preprocess image
                image, original_size = self.preprocess_image(image_path)
                
                if image is None:
                    failed_loads.append(image_path)
                    continue
                
                # Normalize keypoints
                keypoints = self.normalize_keypoints(
                    label_data['keypoints'],
                    original_size,
                    self.image_size
                )
                
                all_images.append(image)
                all_keypoints.append(keypoints)
                loaded += 1
            
            print(f"  Loaded: {loaded} images")
        
        # Convert to numpy arrays
        X = np.array(all_images)
        y = np.array(all_keypoints)
        
        print(f"\n{'='*50}")
        print(f"DATASET SUMMARY")
        print(f"{'='*50}")
        print(f"Total images loaded: {len(X)}")
        print(f"Failed to load: {len(failed_loads)}")
        print(f"Image shape: {X.shape}")
        print(f"Keypoints shape: {y.shape}")
        print(f"{'='*50}\n")
        
        return X, y


class KeypointCNN:
    """CNN Model for keypoint detection."""
    
    def __init__(self, config):
        self.config = config
        self.input_shape = (*config["image_size"], 3)
        self.output_size = config["num_keypoints"] * 2  # x, y for each keypoint
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build the CNN architecture.
        
        Architecture:
        - Input: 256x256x3 image
        - 4 Convolutional blocks with MaxPooling
        - Global Average Pooling
        - Dense layers
        - Output: 42 values (21 keypoints × 2 coordinates)
        """
        print("\n" + "="*50)
        print("BUILDING CNN MODEL")
        print("="*50)
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Block 1: 256x256 -> 128x128
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2: 128x128 -> 64x64
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3: 64x64 -> 32x32
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4: 32x32 -> 16x16
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 5: 16x16 -> 8x8
            layers.Conv2D(512, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(512, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense layers
            layers.Flatten(),
            
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            # Output layer: 42 values (21 keypoints × 2 coordinates)
            # Using sigmoid to output values in [0, 1] range
            layers.Dense(self.output_size, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config["learning_rate"]),
            loss='mse',  # Mean Squared Error for regression
            metrics=['mae']  # Mean Absolute Error
        )
        
        self.model = model
        
        # Print model summary
        print("\nModel Architecture:")
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model."""
        print("\n" + "="*50)
        print("TRAINING CNN MODEL")
        print("="*50)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Epochs: {self.config['epochs']}")
        print("="*50 + "\n")
        
        # Create callbacks
        log_dir = Path(self.config["training_log_path"])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        model_dir = Path(self.config["model_save_path"])
        model_dir.mkdir(parents=True, exist_ok=True)
        
        callback_list = [
            # Save best model
            callbacks.ModelCheckpoint(
                filepath=str(model_dir / "best_model.keras"),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            # CSV logger for documentation
            callbacks.CSVLogger(
                str(log_dir / "training_log.csv"),
                separator=',',
                append=False
            )
        ]
        
        # Train
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.config["batch_size"],
            epochs=self.config["epochs"],
            callbacks=callback_list,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        print(f"\n{'='*50}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*50}")
        print(f"Total training time: {training_time/60:.2f} minutes")
        print(f"Best validation loss: {min(self.history.history['val_loss']):.6f}")
        print(f"{'='*50}\n")
        
        return self.history, training_time
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set."""
        print("\n" + "="*50)
        print("EVALUATING MODEL")
        print("="*50)
        
        # Get predictions
        start_time = time.time()
        predictions = self.model.predict(X_test, verbose=0)
        inference_time = (time.time() - start_time) / len(X_test)
        
        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        
        # Calculate PCK (Percentage of Correct Keypoints)
        # A keypoint is correct if within threshold distance
        threshold = 0.1  # 10% of image size
        
        # Reshape for per-keypoint analysis
        pred_kp = predictions.reshape(-1, 21, 2)
        true_kp = y_test.reshape(-1, 21, 2)
        
        distances = np.sqrt(np.sum((pred_kp - true_kp) ** 2, axis=2))
        correct = distances < threshold
        pck = np.mean(correct) * 100
        
        # Per-keypoint accuracy
        per_keypoint_pck = np.mean(correct, axis=0) * 100
        
        results = {
            "mse": float(mse),
            "mae": float(mae),
            "pck_at_0.1": float(pck),
            "avg_inference_time_ms": float(inference_time * 1000),
            "throughput_fps": float(1 / inference_time),
            "per_keypoint_pck": {
                name: float(acc) 
                for name, acc in zip(KEYPOINT_NAMES, per_keypoint_pck)
            }
        }
        
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"PCK @ 0.1: {pck:.2f}%")
        print(f"Average inference time: {inference_time*1000:.2f} ms")
        print(f"Throughput: {1/inference_time:.1f} FPS")
        print(f"{'='*50}\n")
        
        return results
    
    def save_model(self):
        """Save the trained model."""
        model_dir = Path(self.config["model_save_path"])
        
        # Save in Keras format
        self.model.save(str(model_dir / "keypoint_model.keras"))
        
        # Save in SavedModel format (for serving)
        self.model.save(str(model_dir / "keypoint_model_saved"))
        
        print(f"Model saved to: {model_dir}")
    
    def predict(self, image):
        """Predict keypoints for a single image."""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        predictions = self.model.predict(image, verbose=0)
        
        # Reshape to (21, 2) for keypoints
        keypoints = predictions[0].reshape(21, 2)
        
        return keypoints


def save_training_report(config, data_stats, results, training_time, history):
    """Save comprehensive training report for thesis documentation."""
    log_dir = Path(config["training_log_path"])
    
    report = {
        "experiment_info": {
            "date": datetime.now().isoformat(),
            "model_type": "Convolutional Neural Network (CNN)",
            "task": "2D Keypoint Detection for Character Rigging"
        },
        "dataset": {
            "total_images": data_stats["total"],
            "training_images": data_stats["train"],
            "validation_images": data_stats["val"],
            "test_images": data_stats["test"],
            "image_size": config["image_size"],
            "num_keypoints": config["num_keypoints"]
        },
        "model_architecture": {
            "input_shape": [*config["image_size"], 3],
            "output_size": config["num_keypoints"] * 2,
            "conv_blocks": 5,
            "conv_filters": [32, 64, 128, 256, 512],
            "dense_layers": [1024, 512, 256],
            "dropout_rate": 0.25,
            "activation": "ReLU",
            "output_activation": "Sigmoid"
        },
        "training_config": {
            "batch_size": config["batch_size"],
            "max_epochs": config["epochs"],
            "actual_epochs": len(history.history['loss']),
            "learning_rate": config["learning_rate"],
            "optimizer": "Adam",
            "loss_function": "Mean Squared Error (MSE)"
        },
        "training_results": {
            "total_training_time_minutes": round(training_time / 60, 2),
            "final_training_loss": float(history.history['loss'][-1]),
            "final_validation_loss": float(history.history['val_loss'][-1]),
            "best_validation_loss": float(min(history.history['val_loss']))
        },
        "evaluation_metrics": results,
        "keypoint_names": KEYPOINT_NAMES
    }
    
    # Save report
    report_path = log_dir / "training_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save human-readable summary
    summary_path = log_dir / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CNN KEYPOINT DETECTION - TRAINING REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("DATASET INFORMATION\n")
        f.write("-"*40 + "\n")
        f.write(f"Total images: {data_stats['total']}\n")
        f.write(f"Training set: {data_stats['train']} ({data_stats['train']/data_stats['total']*100:.1f}%)\n")
        f.write(f"Validation set: {data_stats['val']} ({data_stats['val']/data_stats['total']*100:.1f}%)\n")
        f.write(f"Test set: {data_stats['test']} ({data_stats['test']/data_stats['total']*100:.1f}%)\n")
        f.write(f"Image size: {config['image_size']}\n")
        f.write(f"Number of keypoints: {config['num_keypoints']}\n\n")
        
        f.write("MODEL ARCHITECTURE\n")
        f.write("-"*40 + "\n")
        f.write("Type: Convolutional Neural Network (CNN)\n")
        f.write("Convolutional blocks: 5\n")
        f.write("Filters per block: 32 -> 64 -> 128 -> 256 -> 512\n")
        f.write("Dense layers: 1024 -> 512 -> 256 -> 42\n")
        f.write("Regularization: Dropout (0.25 conv, 0.5 dense) + BatchNorm\n\n")
        
        f.write("TRAINING CONFIGURATION\n")
        f.write("-"*40 + "\n")
        f.write(f"Batch size: {config['batch_size']}\n")
        f.write(f"Learning rate: {config['learning_rate']}\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Loss function: Mean Squared Error (MSE)\n")
        f.write(f"Early stopping patience: 15 epochs\n\n")
        
        f.write("TRAINING RESULTS\n")
        f.write("-"*40 + "\n")
        f.write(f"Total training time: {training_time/60:.2f} minutes\n")
        f.write(f"Epochs completed: {len(history.history['loss'])}\n")
        f.write(f"Best validation loss: {min(history.history['val_loss']):.6f}\n\n")
        
        f.write("EVALUATION METRICS (Test Set)\n")
        f.write("-"*40 + "\n")
        f.write(f"Mean Squared Error (MSE): {results['mse']:.6f}\n")
        f.write(f"Mean Absolute Error (MAE): {results['mae']:.6f}\n")
        f.write(f"PCK @ 0.1 threshold: {results['pck_at_0.1']:.2f}%\n")
        f.write(f"Average inference time: {results['avg_inference_time_ms']:.2f} ms\n")
        f.write(f"Throughput: {results['throughput_fps']:.1f} FPS\n\n")
        
        f.write("PER-KEYPOINT ACCURACY (PCK @ 0.1)\n")
        f.write("-"*40 + "\n")
        for name, acc in results['per_keypoint_pck'].items():
            f.write(f"{name}: {acc:.2f}%\n")
    
    print(f"\nTraining report saved to: {log_dir}")
    print(f"  - training_report.json (detailed)")
    print(f"  - training_summary.txt (human-readable)")
    print(f"  - training_log.csv (epoch-by-epoch)")


def main():
    print("\n" + "="*60)
    print("  CNN KEYPOINT DETECTION - TRAINING SCRIPT")
    print("  For 2D to 3D Character Rigging")
    print("="*60 + "\n")
    
    # Check TensorFlow
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Load data
    data_loader = KeypointDataLoader(CONFIG)
    X, y = data_loader.load_dataset()
    
    if len(X) == 0:
        print("ERROR: No data loaded. Check your label files.")
        return
    
    # Split data: 75% train, 15% validation, 10% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=CONFIG["test_split"], random_state=42
    )
    
    val_size = CONFIG["validation_split"] / (1 - CONFIG["test_split"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42
    )
    
    data_stats = {
        "total": len(X),
        "train": len(X_train),
        "val": len(X_val),
        "test": len(X_test)
    }
    
    print(f"Data split:")
    print(f"  Training: {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    
    # Build model
    cnn = KeypointCNN(CONFIG)
    cnn.build_model()
    
    # Train
    history, training_time = cnn.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    results = cnn.evaluate(X_test, y_test)
    
    # Save model
    cnn.save_model()
    
    # Save training report
    save_training_report(CONFIG, data_stats, results, training_time, history)
    
    print("\n" + "="*60)
    print("  TRAINING COMPLETE!")
    print("="*60)
    print(f"\nYour trained model is saved in: {CONFIG['model_save_path']}")
    print(f"Training logs are in: {CONFIG['training_log_path']}")
    print("\nNext step: Integrate this model into your Animatic app!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()