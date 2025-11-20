import cv2
import numpy as np
import pickle
import os

print("=" * 60)
print("Face Recognition (Step 3)")
print("=" * 60)

# Load training landmarks
if not os.path.exists('face_landmarks.pkl'):
    print("Error: face_landmarks.pkl not found!")
    print("Please run these scripts first:")
    print("  1. python capture_images.py")
    print("  2. python train_landmarks.py")
    exit()

print("Loading trained landmarks...")

with open('face_landmarks.pkl', 'rb') as f:
    known_landmarks = pickle.load(f)

print(f"Loaded {len(known_landmarks)} training landmark sets")

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def extract_landmarks(frame):
    """Extract facial landmarks from a frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None
    
    (fx, fy, fw, fh) = faces[0]
    face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
    
    landmarks = {
        'face_width': fw,
        'face_height': fh,
        'face_aspect_ratio': fw / fh if fh > 0 else 0,
        'eyes': [],
        'mouth': None
    }
    
    eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 5)
    for (ex, ey, ew, eh) in eyes[:2]:
        eye_data = {
            'x': (ex + ew/2) / fw,
            'y': (ey + eh/2) / fh,
            'width': ew / fw,
            'height': eh / fh
        }
        landmarks['eyes'].append(eye_data)
    
    mouths = mouth_cascade.detectMultiScale(face_roi_gray, 1.7, 22)
    if len(mouths) > 0:
        (mx, my, mw, mh) = mouths[0]
        landmarks['mouth'] = {
            'x': (mx + mw/2) / fw,
            'y': (my + mh/2) / fh,
            'width': mw / fw,
            'height': mh / fh
        }
    
    if len(landmarks['eyes']) >= 2:
        eyes_sorted = sorted(landmarks['eyes'], key=lambda e: e['x'])
        
        eye_distance = np.sqrt(
            (eyes_sorted[1]['x'] - eyes_sorted[0]['x'])**2 +
            (eyes_sorted[1]['y'] - eyes_sorted[0]['y'])**2
        )
        landmarks['eye_distance'] = eye_distance
        
        eye_center_y = (eyes_sorted[0]['y'] + eyes_sorted[1]['y']) / 2
        landmarks['eye_center_y'] = eye_center_y
        
        eye_level_diff = abs(eyes_sorted[0]['y'] - eyes_sorted[1]['y'])
        landmarks['eye_level_diff'] = eye_level_diff
        
        if landmarks['mouth']:
            eye_to_mouth = landmarks['mouth']['y'] - eye_center_y
            landmarks['eye_to_mouth'] = eye_to_mouth
    
    return landmarks

def compare_landmarks(landmarks1, landmarks2):
    """
    Compare two sets of landmarks using Euclidean distance
    Returns similarity score (lower = more similar)
    """
    if not landmarks1 or not landmarks2:
        return float('inf')
    
    if len(landmarks1.get('eyes', [])) < 2 or len(landmarks2.get('eyes', [])) < 2:
        return float('inf')
    
    # Features to compare (feature_name, weight)
    features = [
        ('face_aspect_ratio', 2.0),
        ('eye_distance', 4.0),
        ('eye_center_y', 2.0),
        ('eye_level_diff', 1.5),
        ('eye_to_mouth', 2.5),
    ]
    
    total_distance = 0
    total_weight = 0
    
    for feature_name, weight in features:
        if feature_name in landmarks1 and feature_name in landmarks2:
            distance = abs(landmarks1[feature_name] - landmarks2[feature_name])
            total_distance += distance * weight
            total_weight += weight
    
    # Compare eye sizes
    if len(landmarks1['eyes']) >= 2 and len(landmarks2['eyes']) >= 2:
        eyes1_sorted = sorted(landmarks1['eyes'], key=lambda e: e['x'])
        eyes2_sorted = sorted(landmarks2['eyes'], key=lambda e: e['x'])
        
        for i in range(2):
            width_diff = abs(eyes1_sorted[i]['width'] - eyes2_sorted[i]['width'])
            height_diff = abs(eyes1_sorted[i]['height'] - eyes2_sorted[i]['height'])
            
            total_distance += (width_diff + height_diff) * 1.5
            total_weight += 3.0
    
    if total_weight > 0:
        return total_distance / total_weight
    else:
        return float('inf')

# Calculate average template
print("Computing average face template...")

avg_landmarks = {
    'face_aspect_ratio': np.mean([l['face_aspect_ratio'] for l in known_landmarks]),
    'eye_distance': np.mean([l.get('eye_distance', 0) for l in known_landmarks if 'eye_distance' in l]),
    'eye_center_y': np.mean([l.get('eye_center_y', 0) for l in known_landmarks if 'eye_center_y' in l]),
    'eye_level_diff': np.mean([l.get('eye_level_diff', 0) for l in known_landmarks if 'eye_level_diff' in l]),
    'eye_to_mouth': np.mean([l.get('eye_to_mouth', 0) for l in known_landmarks if 'eye_to_mouth' in l]),
}

avg_landmarks['eyes'] = [
    {
        'x': np.mean([e['x'] for l in known_landmarks for e in l['eyes'] if len(l['eyes']) > 0]),
        'y': np.mean([e['y'] for l in known_landmarks for e in l['eyes'] if len(l['eyes']) > 0]),
        'width': np.mean([e['width'] for l in known_landmarks for e in l['eyes'] if len(l['eyes']) > 0]),
        'height': np.mean([e['height'] for l in known_landmarks for e in l['eyes'] if len(l['eyes']) > 0])
    }
] * 2

# Initialize webcam
print("Starting webcam...")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Recognition parameters
MATCH_THRESHOLD = 0.15
process_every_n_frames = 3
frame_count = 0

last_face_match = False
last_similarity_score = float('inf')

print("\n" + "=" * 60)
print("Face Recognition Active")
print("=" * 60)
print(f"Match threshold: {MATCH_THRESHOLD}")
print("Features: Face + Eyes + Mouth (Euclidean distance)")
print("Press 'q' to quit")
print("=" * 60 + "\n")

while True:
    ret, frame = cap.read()
    
    if ret:
        frame_count += 1
        
        # Process every nth frame
        if frame_count % process_every_n_frames == 0:
            current_landmarks = extract_landmarks(frame)
            
            if current_landmarks and len(current_landmarks.get('eyes', [])) >= 2:
                similarity = compare_landmarks(current_landmarks, avg_landmarks)
                last_similarity_score = similarity
                last_face_match = similarity < MATCH_THRESHOLD
            else:
                last_face_match = False
                last_similarity_score = float('inf')
        
        # Visualize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (fx, fy, fw, fh) in faces:
            # Color based on match
            if last_face_match:
                color = (0, 255, 0)  # Green
                label = "MATCH"
            else:
                color = (0, 0, 255)  # Red
                label = "NO MATCH"
            
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), color, 2)
            
            # Draw landmarks
            face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
            face_roi_color = frame[fy:fy+fh, fx:fx+fw]
            
            # Eyes (green)
            eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                cv2.circle(face_roi_color, (ex + ew//2, ey + eh//2), 3, (0, 255, 0), -1)
            
            # Mouth (yellow)
            mouths = mouth_cascade.detectMultiScale(face_roi_gray, 1.7, 22)
            for (mx, my, mw, mh) in mouths:
                cv2.rectangle(face_roi_color, (mx, my), (mx+mw, my+mh), (255, 255, 0), 2)
            
            cv2.putText(frame, label, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Status display
        if last_face_match:
            status = "AUTHORIZED"
            status_color = (0, 255, 0)
        elif len(faces) > 0:
            status = "UNAUTHORIZED"
            status_color = (0, 0, 255)
        else:
            status = "NO FACE"
            status_color = (255, 255, 0)
        
        cv2.putText(frame, status, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 2)
        
        # Similarity score
        if last_similarity_score != float('inf'):
            score_text = f"Score: {last_similarity_score:.4f}"
            cv2.putText(frame, score_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Face Recognition", frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nRecognition stopped")
