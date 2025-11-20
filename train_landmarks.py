import cv2
import numpy as np
import os
import pickle
from glob import glob

print("=" * 60)
print("Landmark Training (Step 2 of 2)")
print("=" * 60)

# Check if images exist
if not os.path.exists('training_images'):
    print("Error: training_images/ folder not found!")
    print("Please run 'python capture_images.py' first to capture training images.")
    exit()

# Get all captured images
image_files = sorted(glob('training_images/face_*.jpg'))

if len(image_files) == 0:
    print("Error: No images found in training_images/!")
    print("Please run 'python capture_images.py' first.")
    exit()

print(f"Found {len(image_files)} training images")
print("Extracting facial landmarks...\n")

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def extract_landmarks(image):
    """Extract facial landmarks from an image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None
    
    # Use the largest face
    (fx, fy, fw, fh) = faces[0]
    face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
    
    landmarks = {
        'face_width': fw,
        'face_height': fh,
        'face_aspect_ratio': fw / fh if fh > 0 else 0,
        'eyes': [],
        'mouth': None
    }
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 5)
    for (ex, ey, ew, eh) in eyes[:2]:
        eye_data = {
            'x': (ex + ew/2) / fw,
            'y': (ey + eh/2) / fh,
            'width': ew / fw,
            'height': eh / fh
        }
        landmarks['eyes'].append(eye_data)
    
    # Detect mouth
    mouths = mouth_cascade.detectMultiScale(face_roi_gray, 1.7, 22)
    if len(mouths) > 0:
        (mx, my, mw, mh) = mouths[0]
        landmarks['mouth'] = {
            'x': (mx + mw/2) / fw,
            'y': (my + mh/2) / fh,
            'width': mw / fw,
            'height': mh / fh
        }
    
    # Calculate measurements
    if len(landmarks['eyes']) >= 2:
        eyes_sorted = sorted(landmarks['eyes'], key=lambda e: e['x'])
        
        # Eye spacing (Euclidean distance)
        eye_distance = np.sqrt(
            (eyes_sorted[1]['x'] - eyes_sorted[0]['x'])**2 +
            (eyes_sorted[1]['y'] - eyes_sorted[0]['y'])**2
        )
        landmarks['eye_distance'] = eye_distance
        
        # Eye center position
        eye_center_y = (eyes_sorted[0]['y'] + eyes_sorted[1]['y']) / 2
        landmarks['eye_center_y'] = eye_center_y
        
        # Eye symmetry
        eye_level_diff = abs(eyes_sorted[0]['y'] - eyes_sorted[1]['y'])
        landmarks['eye_level_diff'] = eye_level_diff
        
        # Eye-to-mouth distance
        if landmarks['mouth']:
            eye_to_mouth = landmarks['mouth']['y'] - eye_center_y
            landmarks['eye_to_mouth'] = eye_to_mouth
    
    return landmarks

# Process all images
collected_landmarks = []
successful = 0
failed = 0

for i, image_path in enumerate(image_files):
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"✗ Could not load: {image_path}")
        failed += 1
        continue
    
    # Extract landmarks
    landmarks = extract_landmarks(image)
    
    if landmarks and len(landmarks.get('eyes', [])) >= 2:
        collected_landmarks.append(landmarks)
        successful += 1
        
        # Show progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images... "
                  f"(✓ {successful} successful, ✗ {failed} failed)")
    else:
        failed += 1

print(f"\nProcessing complete: {successful} successful, {failed} failed")

# Save landmarks
if len(collected_landmarks) > 0:
    with open('face_landmarks.pkl', 'wb') as f:
        pickle.dump(collected_landmarks, f)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Successfully extracted landmarks from {len(collected_landmarks)} images")
    print(f"Landmarks saved to: face_landmarks.pkl")
    
    # Show statistics
    print("\nLandmark Statistics:")
    print(f"  - Face aspect ratio: {np.mean([l['face_aspect_ratio'] for l in collected_landmarks]):.3f}")
    print(f"  - Eye distance: {np.mean([l.get('eye_distance', 0) for l in collected_landmarks if 'eye_distance' in l]):.3f}")
    print(f"  - Images with mouth detected: {sum(1 for l in collected_landmarks if l.get('mouth'))}/{len(collected_landmarks)}")
    
    print("\nNext step: Run 'python recognize_face.py' to test recognition")
    print("=" * 60)
else:
    print("\n" + "=" * 60)
    print("Training Failed!")
    print("=" * 60)
    print("No valid landmarks were extracted from the images.")
    print("Please try recapturing with better lighting and clearer face visibility.")
    print("=" * 60)
