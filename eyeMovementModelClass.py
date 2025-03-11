import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



class EyeMovementModel:
    def __init__(self, data_dir="eye_movement_data", sequence_length=10, feature_dim=15):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = None
        self.scaler = StandardScaler()
        self.eye_positions = ["neutral", "looking_up", "looking_down", "scroll"]


    def load_data(self):
        """Load collected data and prepare for model training"""
        x, y = [], []

        for position_idx, position_name in enumerate(self.eye_positions):
            position_dir = os.path.join(self.data_dir, position_name)
            if not os.path.exists(position_dir):
                continue

            files = os.listdir(position_dir)
            for file in files:
                if file.endswith('.pkl'):
                    filepath = os.path.join(position_dir, file)

                    with open(filepath, 'rb') as f:
                        sequence = pickle.load(f)
                    # Pad or truncate to fixed length
                    if len(sequence) < self.sequence_length:
                        # Pad with zeros if sequence is too short
                        padding = [np.zeros(self.feature_dim) for _ in range(self.sequence_length - len(sequence))]
                        sequence = sequence + padding
                    elif len(sequence) > self.sequence_length:
                        # Truncate if sequence is too long
                        sequence = sequence[:self.sequence_length]
                    
                    x.append(sequence)
                    y.append(position_idx)

        x=np.array(x)
        y=np.array(y)

        return x,y
    
    def preprocess_data(self, x, y):
        """Preprocess data for model training"""
        # Reshape for scaling
        original_shape = x.shape

        x_reshaped = x.reshape(-1, x.shape[-1])
        
        # Fit and transform
        x_scaled = self.scaler.fit_transform(x_reshaped)
        
        # Reshape back
        x_processed = x_scaled.reshape(original_shape)
        
        # One-hot encode labels
        y_one_hot = tf.keras.utils.to_categorical(y, num_classes=len(self.eye_positions))

        
        return x_processed, y_one_hot
    
    def build_model(self):
        """Build the LSTM model for sequence classification"""
        model = Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(self.sequence_length, self.feature_dim)),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(len(self.eye_positions), activation='softmax')
        ])
    
        model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
        self.model = model
        return model
    
    def train(self, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model on collected data"""
        # Load and preprocess data
        x, y = self.load_data()
        x_processed, y_one_hot = self.preprocess_data(x, y)
        
        # Split into train and validation
        x_train, x_val, y_train, y_val = train_test_split(
            x_processed, y_one_hot, test_size=validation_split, random_state=42
        )
        
        # Build model if not already built
        if self.model is None:
            self.build_model()

        # Train
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            verbose=1
        )

        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

    def save_model(self, model_path = "eye_movement_model"):
        """Save the trained model and scaler"""
        if self.model is None:
            print("There is no model")
            return
        
        # Create directory
        os.makedirs(model_path, exist_ok=True)

        # Save model
        self.model.save(os.path.join(model_path, "my_model.h5"))

        # Save scaler
        with open(os.path.join(model_path, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        print(f"Model and scaler saved to {model_path}")

    def load_model(self, model_path = "eye_movement_model") -> bool:
        model_file = os.path.join(model_path, "my_model.h5")
        scaler_file = os.path.join(model_path, "scaler.pkl")

        if not os.path.exists(model_file) or not os.path.exists(scaler_file):
            print("Model or scaler file not found")
            return False

        #Load model
        self.model = tf.keras.models.load_model(model_file)

         # Load scaler
        with open(scaler_file, "rb") as f:
            self.scaler = pickle.load(f)

        print(f"Model and scaler loaded from {model_path}")
        return True
    
    def predict_sequence(self, sequence):
        if self.model is None:
            print("No model loaded")
            return None
        
        # Ensure sequence has correct length
        if len(sequence) < self.sequence_length:
            padding = [np.zeros(self.feature_dim) for _ in range(self.sequence_length - len(sequence))]
            sequence = sequence + padding
        elif len(sequence) > self.sequence_length:
            sequence = sequence[:self.sequence_length]

        # Reshape and normalize
        x = np.array([sequence])
        x_reshaped = x.reshape(-1, x.shape[-1])
        x_scaled = self.scaler.transform(x_reshaped)
        x_processed = x_scaled.reshape(x.shape)

         # Predict
        prediction = self.model.predict(x_processed)[0]
        position_idx = np.argmax(prediction)
        confidence = prediction[position_idx]

        return {
            'position': self.eye_positions[position_idx],
            'confidence': float(confidence),
            'raw_predictions': {position: float(pred) for position, pred in zip(self.eye_positions, prediction)}
        }
    
    def get_highest_prediction(predictions):
        """
        Given a dictionary of predictions, return the class with the highest probability 
        and its corresponding value.
        """
        highest_class = max(predictions, key=predictions.get)
        highest_value = predictions[highest_class]
        return highest_class, highest_value


