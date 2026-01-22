"""
AUTO-LABELING SCRIPT FOR CNN TRAINING DATA
==========================================
This script automatically generates keypoint labels for your images
using MediaPipe, then saves them in a format ready for CNN training.

Author: Angelo Tristan Sinohin
Thesis: 2D to 3D Character Rigging using CNN
"""

import os
import cv2
import json
import numpy as np
import mediapipe as mp
from datetime import datetime
from pathlib import Path

# ============================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================

# Folder containing your images (change this to your actual path)
INPUT_FOLDER = "./human_images"  # Change to your human images folder
# INPUT_FOLDER = "./anime_images"  # Use this later for anime images

# Output folder for labels
OUTPUT_FOLDER = "./labeled_data"

# ============================================
# KEYPOINT DEFINITIONS (21 keypoints)
# ============================================

KEYPOINT_NAMES = [
    'head_top',        # 0
    'head_center',     # 1
    'neck',            # 2
    'left_shoulder',   # 3
    'right_shoulder',  # 4
    'left_elbow',      # 5
    'right_elbow',     # 6
    'left_wrist',      # 7
    'right_wrist',     # 8
    'chest_center',    # 9
    'waist_center',    # 10
    'left_hip',        # 11
    'right_hip',       # 12
    'left_knee',       # 13
    'right_knee',      # 14
    'left_ankle',      # 15
    'right_ankle',     # 16
    'left_hand',       # 17
    'right_hand',      # 18
    'left_foot',       # 19
    'right_foot'       # 20
]

# MediaPipe landmark indices mapping to our keypoints
MP_TO_CUSTOM = {
    0: 'nose',  # We'll use this to estimate head_center
    11: 'left_shoulder',
    12: 'right_shoulder',
    13: 'left_elbow',
    14: 'right_elbow',
    15: 'left_wrist',
    16: 'right_wrist',
    23: 'left_hip',
    24: 'right_hip',
    25: 'left_knee',
    26: 'right_knee',
    27: 'left_ankle',
    28: 'right_ankle',
    19: 'left_index',   # For hand position
    20: 'right_index',  # For hand position
    31: 'left_foot_index',
    32: 'right_foot_index'
}


class AutoLabeler:
    def __init__(self, input_folder, output_folder):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        
        # Create output directories
        self.output_folder.mkdir(parents=True, exist_ok=True)
        (self.output_folder / "labels").mkdir(exist_ok=True)
        (self.output_folder / "visualizations").mkdir(exist_ok=True)
        (self.output_folder / "failed").mkdir(exist_ok=True)
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # Statistics
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'low_confidence': 0
        }
        
    def process_image(self, image_path):
        """Process a single image and extract keypoints."""
        image = cv2.imread(str(image_path))
        if image is None:
            return None, "Could not read image"
        
        height, width = image.shape[:2]
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run pose detection
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None, "No pose detected"
        
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Convert to our keypoint format
        keypoints = self._convert_landmarks(landmarks, width, height)
        
        # Calculate average confidence
        confidences = [landmarks[i].visibility for i in MP_TO_CUSTOM.keys() if i < len(landmarks)]
        avg_confidence = np.mean(confidences)
        
        return {
            'keypoints': keypoints,
            'confidence': float(avg_confidence),
            'image_width': width,
            'image_height': height,
            'keypoint_names': KEYPOINT_NAMES
        }, None
    
    def _convert_landmarks(self, landmarks, width, height):
        """Convert MediaPipe landmarks to our 21-keypoint format."""
        keypoints = {}
        
        # Direct mappings from MediaPipe
        keypoints['left_shoulder'] = self._get_point(landmarks[11], width, height)
        keypoints['right_shoulder'] = self._get_point(landmarks[12], width, height)
        keypoints['left_elbow'] = self._get_point(landmarks[13], width, height)
        keypoints['right_elbow'] = self._get_point(landmarks[14], width, height)
        keypoints['left_wrist'] = self._get_point(landmarks[15], width, height)
        keypoints['right_wrist'] = self._get_point(landmarks[16], width, height)
        keypoints['left_hip'] = self._get_point(landmarks[23], width, height)
        keypoints['right_hip'] = self._get_point(landmarks[24], width, height)
        keypoints['left_knee'] = self._get_point(landmarks[25], width, height)
        keypoints['right_knee'] = self._get_point(landmarks[26], width, height)
        keypoints['left_ankle'] = self._get_point(landmarks[27], width, height)
        keypoints['right_ankle'] = self._get_point(landmarks[28], width, height)
        keypoints['left_hand'] = self._get_point(landmarks[19], width, height)
        keypoints['right_hand'] = self._get_point(landmarks[20], width, height)
        keypoints['left_foot'] = self._get_point(landmarks[31], width, height)
        keypoints['right_foot'] = self._get_point(landmarks[32], width, height)
        
        # Calculated keypoints
        nose = self._get_point(landmarks[0], width, height)
        
        # Neck: midpoint between shoulders
        keypoints['neck'] = {
            'x': (keypoints['left_shoulder']['x'] + keypoints['right_shoulder']['x']) / 2,
            'y': (keypoints['left_shoulder']['y'] + keypoints['right_shoulder']['y']) / 2,
            'confidence': (landmarks[11].visibility + landmarks[12].visibility) / 2
        }
        
        # Head center: use nose position
        keypoints['head_center'] = nose
        
        # Head top: estimate above head_center
        head_height = abs(keypoints['neck']['y'] - nose['y'])
        keypoints['head_top'] = {
            'x': nose['x'],
            'y': nose['y'] - head_height * 0.8,
            'confidence': nose['confidence']
        }
        
        # Chest center: midpoint between neck and waist
        hip_center_y = (keypoints['left_hip']['y'] + keypoints['right_hip']['y']) / 2
        keypoints['chest_center'] = {
            'x': keypoints['neck']['x'],
            'y': (keypoints['neck']['y'] + hip_center_y) / 2,
            'confidence': (landmarks[11].visibility + landmarks[23].visibility) / 2
        }
        
        # Waist center: midpoint between hips
        keypoints['waist_center'] = {
            'x': (keypoints['left_hip']['x'] + keypoints['right_hip']['x']) / 2,
            'y': (keypoints['left_hip']['y'] + keypoints['right_hip']['y']) / 2,
            'confidence': (landmarks[23].visibility + landmarks[24].visibility) / 2
        }
        
        # Convert to list format for CNN training
        keypoints_list = []
        for name in KEYPOINT_NAMES:
            kp = keypoints[name]
            keypoints_list.append({
                'name': name,
                'x': kp['x'],
                'y': kp['y'],
                'confidence': kp['confidence']
            })
        
        return keypoints_list
    
    def _get_point(self, landmark, width, height):
        """Convert a MediaPipe landmark to pixel coordinates."""
        return {
            'x': landmark.x * width,
            'y': landmark.y * height,
            'confidence': landmark.visibility
        }
    
    def visualize_keypoints(self, image_path, keypoints_data, output_path):
        """Draw keypoints on image for verification."""
        image = cv2.imread(str(image_path))
        
        # Define skeleton connections for visualization
        skeleton = [
            ('head_top', 'head_center'),
            ('head_center', 'neck'),
            ('neck', 'left_shoulder'),
            ('neck', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('left_wrist', 'left_hand'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('right_wrist', 'right_hand'),
            ('neck', 'chest_center'),
            ('chest_center', 'waist_center'),
            ('waist_center', 'left_hip'),
            ('waist_center', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('left_ankle', 'left_foot'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            ('right_ankle', 'right_foot'),
        ]
        
        # Create keypoint lookup
        kp_dict = {kp['name']: kp for kp in keypoints_data['keypoints']}
        
        # Draw skeleton lines
        for start_name, end_name in skeleton:
            start = kp_dict[start_name]
            end = kp_dict[end_name]
            pt1 = (int(start['x']), int(start['y']))
            pt2 = (int(end['x']), int(end['y']))
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        
        # Draw keypoints
        for kp in keypoints_data['keypoints']:
            x, y = int(kp['x']), int(kp['y'])
            confidence = kp['confidence']
            
            # Color based on confidence (red=low, green=high)
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))
            
            cv2.circle(image, (x, y), 5, color, -1)
            cv2.circle(image, (x, y), 7, (255, 255, 255), 1)
        
        # Add confidence text
        cv2.putText(
            image, 
            f"Confidence: {keypoints_data['confidence']:.2f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        cv2.imwrite(str(output_path), image)
    
    def process_all_images(self, visualize=True, save_every=50):
        """Process all images in the input folder."""
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [
            f for f in self.input_folder.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        self.stats['total'] = len(image_files)
        print(f"\n{'='*60}")
        print(f"AUTO-LABELING STARTED")
        print(f"{'='*60}")
        print(f"Input folder: {self.input_folder}")
        print(f"Output folder: {self.output_folder}")
        print(f"Total images found: {self.stats['total']}")
        print(f"{'='*60}\n")
        
        # Master label file
        all_labels = []
        failed_images = []
        low_confidence_images = []
        
        for i, image_path in enumerate(image_files):
            # Process image
            result, error = self.process_image(image_path)
            
            if error:
                self.stats['failed'] += 1
                failed_images.append({
                    'filename': image_path.name,
                    'error': error
                })
                # Copy failed image to failed folder
                failed_path = self.output_folder / "failed" / image_path.name
                if image_path.exists():
                    import shutil
                    shutil.copy(str(image_path), str(failed_path))
            else:
                self.stats['success'] += 1
                
                # Check confidence
                if result['confidence'] < 0.7:
                    self.stats['low_confidence'] += 1
                    low_confidence_images.append(image_path.name)
                
                # Save individual label file
                label_data = {
                    'filename': image_path.name,
                    'filepath': str(image_path),
                    **result,
                    'labeled_at': datetime.now().isoformat()
                }
                
                label_path = self.output_folder / "labels" / f"{image_path.stem}.json"
                with open(label_path, 'w') as f:
                    json.dump(label_data, f, indent=2)
                
                all_labels.append(label_data)
                
                # Visualize
                if visualize:
                    vis_path = self.output_folder / "visualizations" / f"{image_path.stem}_labeled.jpg"
                    self.visualize_keypoints(image_path, result, vis_path)
            
            # Progress update
            if (i + 1) % save_every == 0 or (i + 1) == self.stats['total']:
                progress = (i + 1) / self.stats['total'] * 100
                print(f"Progress: {i+1}/{self.stats['total']} ({progress:.1f}%) - "
                      f"Success: {self.stats['success']}, Failed: {self.stats['failed']}")
        
        # Save master labels file (for CNN training)
        master_file = self.output_folder / "all_labels.json"
        with open(master_file, 'w') as f:
            json.dump({
                'total_images': self.stats['total'],
                'successful': self.stats['success'],
                'failed': self.stats['failed'],
                'low_confidence_count': self.stats['low_confidence'],
                'keypoint_names': KEYPOINT_NAMES,
                'labels': all_labels
            }, f, indent=2)
        
        # Save failed images list
        failed_file = self.output_folder / "failed_images.json"
        with open(failed_file, 'w') as f:
            json.dump(failed_images, f, indent=2)
        
        # Save low confidence images list (for manual review)
        review_file = self.output_folder / "needs_review.json"
        with open(review_file, 'w') as f:
            json.dump({
                'count': self.stats['low_confidence'],
                'images': low_confidence_images
            }, f, indent=2)
        
        # Print summary
        self._print_summary()
        
        return self.stats
    
    def _print_summary(self):
        """Print labeling summary."""
        print(f"\n{'='*60}")
        print("LABELING COMPLETE - SUMMARY")
        print(f"{'='*60}")
        print(f"Total images processed: {self.stats['total']}")
        print(f"Successfully labeled:   {self.stats['success']} ({self.stats['success']/max(1,self.stats['total'])*100:.1f}%)")
        print(f"Failed:                 {self.stats['failed']} ({self.stats['failed']/max(1,self.stats['total'])*100:.1f}%)")
        print(f"Low confidence:         {self.stats['low_confidence']} (needs manual review)")
        print(f"{'='*60}")
        print(f"\nOutput files:")
        print(f"  - Labels: {self.output_folder}/labels/")
        print(f"  - Visualizations: {self.output_folder}/visualizations/")
        print(f"  - Master file: {self.output_folder}/all_labels.json")
        print(f"  - Failed images: {self.output_folder}/failed_images.json")
        print(f"  - Needs review: {self.output_folder}/needs_review.json")
        print(f"{'='*60}\n")


def main():
    """Main function to run auto-labeling."""
    print("\n" + "="*60)
    print("  ANIMATIC - AUTO LABELING TOOL")
    print("  For CNN Training Data Preparation")
    print("="*60 + "\n")
    
    # Check if input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"ERROR: Input folder '{INPUT_FOLDER}' not found!")
        print(f"\nPlease do the following:")
        print(f"1. Create a folder called '{INPUT_FOLDER}'")
        print(f"2. Put your human images inside it")
        print(f"3. Run this script again")
        print(f"\nOr edit INPUT_FOLDER at the top of this script")
        return
    
    # Count images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_count = len([
        f for f in os.listdir(INPUT_FOLDER) 
        if os.path.splitext(f)[1].lower() in image_extensions
    ])
    
    if image_count == 0:
        print(f"ERROR: No images found in '{INPUT_FOLDER}'!")
        print(f"Supported formats: {image_extensions}")
        return
    
    print(f"Found {image_count} images in '{INPUT_FOLDER}'")
    print(f"Output will be saved to '{OUTPUT_FOLDER}'")
    
    # Ask for confirmation
    response = input("\nStart auto-labeling? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Run labeler
    labeler = AutoLabeler(INPUT_FOLDER, OUTPUT_FOLDER)
    stats = labeler.process_all_images(visualize=True)
    
    print("\nNext steps:")
    print("1. Check the 'visualizations' folder to verify labels are correct")
    print("2. Review images listed in 'needs_review.json'")
    print("3. Move any badly labeled images to 'failed' folder")
    print("4. Run the CNN training script (coming next!)")


if __name__ == "__main__":
    main()