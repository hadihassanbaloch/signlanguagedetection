import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load pre-trained CNN-LSTM model
model = tf.keras.models.load_model(r'D:\PythonProject\signlanguage\sign_model2.h5')

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Label encoder for sign language (A-Z and 0-9)
label_encoder = LabelEncoder()
label_encoder.fit(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

# Function to normalize hand landmarks (without flattening)
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(21, 3)  # Shape: (21, 3)
    
    # Find min/max for normalization
    x_max, x_min = np.max(landmarks[:, 0]), np.min(landmarks[:, 0])
    y_max, y_min = np.max(landmarks[:, 1]), np.min(landmarks[:, 1])
    
    # Normalize x and y coordinates
    if x_max != x_min:
        landmarks[:, 0] = (landmarks[:, 0] - x_min) / (x_max - x_min)
    if y_max != y_min:
        landmarks[:, 1] = (landmarks[:, 1] - y_min) / (y_max - y_min)
    
    return landmarks  # Return the (21, 3) array

# Open webcam for live capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    # Convert frame from BGR to RGB (required by MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    result = hands.process(frame_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmark_row = []
            for lm in hand_landmarks.landmark:
                landmark_row.extend([lm.x, lm.y, lm.z])
            
            # Normalize the landmarks
            landmark_row = normalize_landmarks(landmark_row)
            
            # Convert to numpy array and reshape for CNN-LSTM input (batch_size, timesteps, landmarks, features, channels)
            landmark_row = np.array(landmark_row).reshape(1, 1, 21, 3, 1)  # Shape: (1, 1, 21, 3, 1)
            
            # Make prediction using the model
            prediction = model.predict(landmark_row)
            predicted_class = np.argmax(prediction)
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
            
            # Display the predicted label on the frame
            cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the frame with predictions
    cv2.imshow('Sign Language Detection', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
