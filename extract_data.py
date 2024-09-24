import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Define paths
dataset_dir = r'D:\PythonProject\signlanguage\usa_dataset\asl_dataset\asl_dataset' # download and give path to american sign language dataset
output_csv = 'landmarks.csv'

# Initialize a list to store landmarks and labels
data = []

# Loop over all classes and images
for label in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, label)
    if not os.path.isdir(class_dir):
        continue

    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:
            continue
        
        # Convert image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and extract hand landmarks
        result = hands.process(image_rgb)
        
        # Check if hand landmarks are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Flatten the landmarks (21 points x 3 coordinates)
                landmark_row = []
                for lm in hand_landmarks.landmark:
                    landmark_row.extend([lm.x, lm.y, lm.z])
                
                # Append landmarks with label to the data list
                landmark_row.append(label)
                data.append(landmark_row)

        else:
            # Skip images where no landmarks are detected
            print(f"No landmarks detected in {image_name}, skipping...")

# Save the data into a CSV file (or use another format like numpy)
column_names = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
df = pd.DataFrame(data, columns=column_names)
df.to_csv(output_csv, index=False)

print(f"Landmarks extracted and saved to {output_csv}")
