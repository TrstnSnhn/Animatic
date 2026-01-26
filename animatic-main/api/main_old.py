from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import tempfile
import json
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import struct
import base64
import mediapipe as mp
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

class AnimeCharacterProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        self.mp_landmarks = {
            0: 'nose', 1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
            4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
            7: 'left_ear', 8: 'right_ear', 9: 'mouth_left', 10: 'mouth_right',
            11: 'left_shoulder', 12: 'right_shoulder',13: 'left_elbow', 14: 'right_elbow',
            15: 'left_wrist', 16: 'right_wrist', 17: 'left_pinky', 18: 'right_pinky',
            19: 'left_index', 20: 'right_index', 21: 'left_thumb', 22: 'right_thumb',
            23: 'left_hip', 24: 'right_hip', 25: 'left_knee', 26: 'right_knee',
            27: 'left_ankle', 28: 'right_ankle', 29: 'left_heel', 30: 'right_heel',
            31: 'left_foot_index', 32: 'right_foot_index'
        }
        
        self.anime_keypoints = {
            0: 'head_top', 1: 'head_center', 2: 'neck', 3: 'left_shoulder', 4: 'right_shoulder',
            5: 'left_elbow', 6: 'right_elbow', 7: 'left_wrist', 8: 'right_wrist',
            9: 'chest_center', 10: 'waist_center', 11: 'left_hip', 12: 'right_hip',
            13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle',
            17: 'left_hand', 18: 'right_hand', 19: 'left_foot', 20: 'right_foot'
        }
        
        self.anime_enhancement = {
            'contrast_boost': 1.4,
            'brightness_boost': 1.1,
            'saturation_boost': 1.3,
            'sharpness_boost': 1.5,
            'edge_enhance': True
        }
        
        print("Anime MediaPipe Processor initialized successfully")

    def preprocess_anime_image(self, image):
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(rgb_image, table)
        
        pil_image = Image.fromarray(gamma_corrected)
        
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.6) 
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(2.0)
        
        pil_image = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        enhanced_array = np.array(pil_image)
        
        enhanced_array = cv2.bilateralFilter(enhanced_array, 9, 100, 100)
        
        return enhanced_array

    def detect_anime_character_region(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        edges = cv2.Canny(gray, 30, 100, apertureSize=3, L2gradient=True)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        main_contour = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(main_contour)
        if area < image.shape[0] * image.shape[1] * 0.005:  
            return None
        
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = h / w
        
        if aspect_ratio < 0.8:  # Too wide, probably not a standing character
            contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in contours_sorted[1:4]: 
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w
                if aspect_ratio >= 0.8 and cv2.contourArea(contour) > area * 0.3:
                    main_contour = contour
                    break
        
        return main_contour

    def enhance_pose_for_anime(self, pose_landmarks, image_shape):
        if not pose_landmarks or not pose_landmarks.landmark:
            return None
        
        height, width = image_shape[:2]
        landmarks_array = np.array(
            [[lm.x * width, lm.y * height, lm.visibility] for lm in pose_landmarks.landmark]
        )

        mp_to_anime = {
            11: 3, 12: 4, 13: 5, 14: 6, 15: 7, 16: 8, 23: 11, 24: 12, 
            25: 13, 26: 14, 27: 15, 28: 16,
            19: 17, # left_index -> left_hand
            20: 18, # right_index -> right_hand
            31: 19, # left_foot_index -> left_foot
            32: 20  # right_foot_index -> right_foot
        }
        
        anime_to_mp = {v: k for k, v in mp_to_anime.items()}

        limb_chains = {
            5: (3, 7),  
            6: (4, 8),  
            13: (11, 15), 
            14: (12, 16)  
        }
        
        num_keypoints = len(self.anime_keypoints)
        temp_keypoints = np.zeros((num_keypoints, 3))
        for mp_idx, anime_idx in mp_to_anime.items():
            if mp_idx < len(landmarks_array):
                temp_keypoints[anime_idx] = landmarks_array[mp_idx]

        for mid_idx, (start_idx, end_idx) in limb_chains.items():
            start_kp, mid_kp, end_kp = temp_keypoints[start_idx], temp_keypoints[mid_idx], temp_keypoints[end_idx]
            
            if start_kp[2] > 0.5 and end_kp[2] > 0.5 and mid_kp[2] < 0.4:
                expected_pos = (start_kp[:2] + end_kp[:2]) / 2
                detected_pos = mid_kp[:2]
                distance = np.linalg.norm(expected_pos - detected_pos)
                tolerance = np.linalg.norm(start_kp[:2] - end_kp[:2]) * 0.25
                
                if distance < tolerance:
                    if mid_idx in anime_to_mp:
                        mp_index_to_update = anime_to_mp[mid_idx]
                        landmarks_array[mp_index_to_update][2] = min(mid_kp[2] * 2.5, 0.9)

        enhanced_keypoints = np.zeros((num_keypoints, 3))
        for mp_idx, anime_idx in mp_to_anime.items():
            if mp_idx < len(landmarks_array):
                enhanced_keypoints[anime_idx] = landmarks_array[mp_idx]

        nose = landmarks_array[0]
        left_shoulder, right_shoulder = landmarks_array[11], landmarks_array[12]
        left_hip, right_hip = landmarks_array[23], landmarks_array[24]

        if nose[2] > 0.3:
            enhanced_keypoints[1] = nose
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            head_height_est = abs(shoulder_y - nose[1])
            enhanced_keypoints[0] = [nose[0], nose[1] - head_height_est * 0.9, nose[2]]

        neck_pt = np.mean([left_shoulder, right_shoulder], axis=0)
        waist_pt = np.mean([left_hip, right_hip], axis=0)
        chest_pt = np.mean([neck_pt, waist_pt], axis=0)
            
        enhanced_keypoints[2] = neck_pt
        enhanced_keypoints[10] = waist_pt
        enhanced_keypoints[9] = chest_pt
        
        return enhanced_keypoints

    def refine_anime_pose(self, keypoints, image_shape):
        height, width = image_shape[:2]
        refined_keypoints = keypoints.copy()
        valid_points = keypoints[keypoints[:, 2] > 0.3]
        if len(valid_points) == 0:
            return refined_keypoints
        
        min_y = np.min(valid_points[:, 1])
        max_y = np.max(valid_points[:, 1])
        char_height = max_y - min_y
        
        if refined_keypoints[1, 2] > 0.3:
            head_center = refined_keypoints[1]
            expected_head_y = min_y + char_height * 0.08
            if abs(head_center[1] - expected_head_y) > char_height * 0.05:
                refined_keypoints[1, 1] = expected_head_y
                refined_keypoints[0, 1] = expected_head_y - (char_height * 0.07)
        
        anatomy_chains = [
            [11, 13, 15],
            [12, 14, 16]
        ]
        
        for chain in anatomy_chains:
            valid_chain = [i for i in chain if refined_keypoints[i, 2] > 0.3]
            if len(valid_chain) >= 2:
                for i in range(len(valid_chain) - 1):
                    curr_idx = valid_chain[i]
                    next_idx = valid_chain[i + 1]
                    if refined_keypoints[next_idx, 1] <= refined_keypoints[curr_idx, 1]:
                        refined_keypoints[next_idx, 1] = refined_keypoints[curr_idx, 1] + height * 0.05
        
        for i in range(len(refined_keypoints)):
            if refined_keypoints[i, 2] > 0.2:
                refined_keypoints[i, 2] = min(refined_keypoints[i, 2] * 1.3, 0.95)
        
        return refined_keypoints

    def process_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image file")
            
            enhanced_image = self.preprocess_anime_image(image)
            
            char_contour = self.detect_anime_character_region(enhanced_image)
            
            if char_contour is not None:
                x, y, w, h = cv2.boundingRect(char_contour)
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(enhanced_image.shape[1] - x, w + 2*padding)
                h = min(enhanced_image.shape[0] - y, h + 2*padding)
                
                cropped_image = enhanced_image[y:y+h, x:x+w]
                process_image = cropped_image
                offset = (x, y)
            else:
                process_image = enhanced_image
                offset = (0, 0)
            
            results = self.pose_detector.process(process_image)
            
            if not results.pose_landmarks:
                raise ValueError("No pose detected. Please ensure the anime character is clearly visible and in a recognizable pose.")
            
            keypoints = self.enhance_pose_for_anime(results.pose_landmarks, process_image.shape)
            
            if keypoints is None:
                raise ValueError("Could not extract valid pose keypoints from anime character.")
            
            if char_contour is not None:
                keypoints[:, 0] += offset[0]
                keypoints[:, 1] += offset[1]
            
            valid_keypoints = keypoints[keypoints[:, 2] > 0.3]
            if len(valid_keypoints) < 8:
                raise ValueError("Could not detect enough anime character features. Please try with a clearer, full-body character image.")
            
            return keypoints, keypoints[:, 2], image
            
        except Exception as e:
            raise ValueError(f"Anime character processing failed: {str(e)}")

    def create_mesh_vertices_with_uv(self, keypoints, image_shape):
        height, width = image_shape[:2]
        vertices = []
        uvs = []
        
        for i, (x, y, confidence) in enumerate(keypoints):
            if confidence > 0.3:  
                norm_x = (x / width) * 2 - 1
                norm_y = -((y / height) * 2 - 1)
                vertices.append([float(norm_x), float(norm_y), 0.0])
                
                uv_x = x / width
                uv_y = y / height 
                uvs.append([float(uv_x), float(uv_y)])
            else:
                vertices.append([0.0, 0.0, 0.0])
                uvs.append([0.0, 0.0])

        if len(vertices) < 3:
            raise ValueError("Too few valid keypoints detected for mesh creation")
        
        additional_vertices, additional_uvs = self.generate_anime_character_mesh_with_uv(
            keypoints, 
            image_shape, 
            arm_width=25.0,
            leg_width=30.0
        )
        vertices.extend(additional_vertices)
        uvs.extend(additional_uvs)
        
        return vertices, uvs

    def generate_anime_character_mesh_with_uv(self, keypoints, image_shape, arm_width=8.0, leg_width=12.0):
        body_vertices = []
        body_uvs = []
        height, width = image_shape[:2]
        np_keypoints = np.array([kp[:2] for kp in keypoints])

        ls, rs = np_keypoints[3], np_keypoints[4]
        lh, rh = np_keypoints[11], np_keypoints[12]
        for i in range(5):
            for j in range(5):
                t_horiz, t_vert = i / 4.0, j / 4.0
                top_pt = ls + (rs - ls) * t_horiz
                bottom_pt = lh + (rh - lh) * t_horiz
                final_pt = top_pt + (bottom_pt - top_pt) * t_vert
                body_vertices.append(final_pt)
                
                uv_x = final_pt[0] / width
                uv_y = final_pt[1] / height
                body_uvs.append([float(uv_x), float(uv_y)])

        head_center, neck = np_keypoints[1], np_keypoints[2]
        head_radius_pixels = np.linalg.norm(head_center - neck)
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            x = head_center[0] + head_radius_pixels * np.cos(angle)
            y = head_center[1] + (head_radius_pixels * np.sin(angle)) * (width / height)
            body_vertices.append(np.array([x, y]))
            
            uv_x = x / width
            uv_y = y / height
            body_uvs.append([float(uv_x), float(uv_y)])

        neck_keypoint = np_keypoints[2]
        left_shoulder = np_keypoints[3]
        right_shoulder = np_keypoints[4]
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        
        neck_center = (neck_keypoint + shoulder_midpoint) / 2
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        neck_radius_pixels = shoulder_width * 0.2
        
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            x = neck_center[0] + neck_radius_pixels * np.cos(angle)
            y = neck_center[1] + (neck_radius_pixels * np.sin(angle)) * (width / height)
            body_vertices.append(np.array([x, y]))
            
            uv_x = x / width
            uv_y = y / height
            body_uvs.append([float(uv_x), float(uv_y)])

        limbs = {
            'left_arm': {'segments': [(3, 5), (5, 7), (7, 17)], 'width': arm_width},
            'right_arm': {'segments': [(4, 6), (6, 8), (8, 18)], 'width': arm_width},
            'left_leg': {'segments': [(11, 13), (13, 15), (15, 19)], 'width': leg_width},
            'right_leg': {'segments': [(12, 14), (14, 16), (16, 20)], 'width': leg_width}
        }
        
        for limb_info in limbs.values():
            perp_direction = None
            for start_idx, end_idx in limb_info['segments']:
                p1, p2 = np_keypoints[start_idx], np_keypoints[end_idx]
                direction = p2 - p1
                if np.linalg.norm(direction) > 1e-6:
                    if perp_direction is None:
                        perp_direction = np.array([-direction[1], direction[0]])
                        perp_direction /= np.linalg.norm(perp_direction)
                else:
                    perp_direction = np.array([0, 1])

                perp_vec = perp_direction * limb_info['width']
                
                limb_verts = [p1 - perp_vec, p1 + perp_vec, p2 - perp_vec, p2 + perp_vec]
                body_vertices.extend(limb_verts)
                
                for vert in limb_verts:
                    uv_x = vert[0] / width
                    uv_y = vert[1] / height
                    body_uvs.append([float(uv_x), float(uv_y)])
        
        normalized_vertices = []
        for x, y in body_vertices:
            norm_x = (x / width) * 2 - 1
            norm_y = -((y / height) * 2 - 1)
            normalized_vertices.append([float(norm_x), float(norm_y), 0.0])
            
        return normalized_vertices, body_uvs

    def _calculate_world_matrices(self, bone_order, bones_dict):
        world_matrices = {}
        for bone_name in bone_order:
            if bone_name not in bones_dict: continue
            
            bone = bones_dict[bone_name]
            parent_name = bone.get('parent')
            
            if parent_name and parent_name in bones_dict:
                parent_head = bones_dict[parent_name]['head']
                local_translation = bone['head'] - parent_head
            else:
                local_translation = bone['head']

            local_matrix = np.identity(4)
            local_matrix[0:3, 3] = local_translation
            
            if parent_name and parent_name in world_matrices:
                world_matrices[bone_name] = np.dot(world_matrices[parent_name], local_matrix)
            else:
                world_matrices[bone_name] = local_matrix
                
        return world_matrices

    def _invert_matrix(self, matrix):
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return np.identity(4)

    def create_tpose_armature(self, keypoints, image_shape):
        height, width = image_shape[:2]
        
        def get_keypoint_pos(keypoint_name):
            for i, name in self.anime_keypoints.items():
                if name == keypoint_name and i < len(keypoints) and keypoints[i][2] > 0.3:
                    x, y, _ = keypoints[i]
                    norm_x = (x / width) * 2 - 1
                    norm_y = -((y / height) * 2 - 1)
                    return np.array([float(norm_x), float(norm_y), 0.0])
            return None

        def get_center_point(name1, name2):
            p1 = get_keypoint_pos(name1)
            p2 = get_keypoint_pos(name2)
            if p1 is not None and p2 is not None:
                return (p1 + p2) / 2
            return p1 if p1 is not None else p2

        pos = {name: get_keypoint_pos(name) for idx, name in self.anime_keypoints.items()}
        
        pos['hip_center'] = get_center_point('left_hip', 'right_hip')
        pos['shoulder_center'] = get_center_point('left_shoulder', 'right_shoulder')
        pos['chest'] = pos['shoulder_center']

        if pos['hip_center'] is None or pos['chest'] is None:
             print("ERROR: Core points (hips or shoulders) not detected.")
             return [], {}
        
        if pos['neck'] is None: pos['neck'] = pos['chest']
        if pos['chest_center'] is None: pos['chest_center'] = (pos['chest'] + pos['hip_center']) / 2.0
        
        if pos.get('left_elbow') is None and pos.get('left_shoulder') is not None and pos.get('left_wrist') is not None:
            pos['left_elbow'] = (pos['left_shoulder'] + pos['left_wrist']) / 2.0
        if pos.get('right_elbow') is None and pos.get('right_shoulder') is not None and pos.get('right_wrist') is not None:
            pos['right_elbow'] = (pos['right_shoulder'] + pos['right_wrist']) / 2.0
        if pos.get('left_knee') is None and pos.get('left_hip') is not None and pos.get('left_ankle') is not None:
            pos['left_knee'] = (pos['left_hip'] + pos['left_ankle']) / 2.0
        if pos.get('right_knee') is None and pos.get('right_hip') is not None and pos.get('right_ankle') is not None:
            pos['right_knee'] = (pos['right_hip'] + pos['right_ankle']) / 2.0
            
        bone_points = ['hip_center', 'chest_center', 'chest', 'neck', 'head_center', 
                       'left_shoulder', 'left_elbow', 'left_wrist', 'left_hand',
                       'right_shoulder', 'right_elbow', 'right_wrist', 'right_hand',
                       'left_hip', 'left_knee', 'left_ankle', 'left_foot',
                       'right_hip', 'right_knee', 'right_ankle', 'right_foot']
        
        for p_name in bone_points:
            if pos.get(p_name) is None:
                print(f"FATAL: Could not determine position for '{p_name}'. Armature creation failed.")
                return [], {}

        bone_order = [
            'hips', 'spine', 'chest', 'neck', 'head',
            'shoulder_L', 'upper_arm_L', 'forearm_L', 'hand_L',
            'shoulder_R', 'upper_arm_R', 'forearm_R', 'hand_R',
            'thigh_L', 'shin_L', 'foot_L',
            'thigh_R', 'shin_R', 'foot_R'
        ]

        bones_def = {
            'hips': {'head': pos['hip_center'], 'tail': pos['chest_center'], 'parent': None},
            'spine': {'head': pos['chest_center'], 'tail': pos['chest'], 'parent': 'hips'},
            'chest': {'head': pos['chest'], 'tail': pos['neck'], 'parent': 'spine'},
            'neck': {'head': pos['neck'], 'tail': pos['head_center'], 'parent': 'chest'},
            'head': {'head': pos['head_center'], 'tail': pos.get('head_top', pos['head_center'] + np.array([0, 0.1, 0])), 'parent': 'neck'},

            'shoulder_L': {'head': pos['chest'], 'tail': pos['left_shoulder'], 'parent': 'chest'},
            'upper_arm_L': {'head': pos['left_shoulder'], 'tail': pos['left_elbow'], 'parent': 'shoulder_L'},
            'forearm_L': {'head': pos['left_elbow'], 'tail': pos['left_wrist'], 'parent': 'upper_arm_L'},
            'hand_L': {'head': pos['left_wrist'], 'tail': pos['left_hand'], 'parent': 'forearm_L'},

            'shoulder_R': {'head': pos['chest'], 'tail': pos['right_shoulder'], 'parent': 'chest'},
            'upper_arm_R': {'head': pos['right_shoulder'], 'tail': pos['right_elbow'], 'parent': 'shoulder_R'},
            'forearm_R': {'head': pos['right_elbow'], 'tail': pos['right_wrist'], 'parent': 'upper_arm_R'},
            'hand_R': {'head': pos['right_wrist'], 'tail': pos['right_hand'], 'parent': 'forearm_R'},
            
            'thigh_L': {'head': pos['hip_center'], 'tail': pos['left_knee'], 'parent': 'hips'},
            'shin_L': {'head': pos['left_knee'], 'tail': pos['left_ankle'], 'parent': 'thigh_L'},
            'foot_L': {'head': pos['left_ankle'], 'tail': pos['left_foot'], 'parent': 'shin_L'},

            'thigh_R': {'head': pos['hip_center'], 'tail': pos['right_knee'], 'parent': 'hips'},
            'shin_R': {'head': pos['right_knee'], 'tail': pos['right_ankle'], 'parent': 'thigh_R'},
            'foot_R': {'head': pos['right_ankle'], 'tail': pos['right_foot'], 'parent': 'shin_R'},
        }
        
        final_bones = {name: bones_def[name] for name in bone_order}
        
        print(f"Created armature from pose with {len(final_bones)} bones.")
        return bone_order, final_bones

    def _automatic_skin_weights(self, vertices, bone_order, bones_dict, world_matrices):
        num_vertices = len(vertices)
        joints_data = np.zeros((num_vertices, 4), dtype=np.uint8)
        weights_data = np.zeros((num_vertices, 4), dtype=np.float32)
        bone_map = {name: i for i, name in enumerate(bone_order)}

        for i, vertex in enumerate(vertices):
            v = np.array(vertex)
            
            min_dist = float('inf')
            closest_bone_idx = 0
            
            for bone_name in bone_order:
                if bone_name not in bones_dict: continue
                
                p1 = bones_dict[bone_name]['head']
                p2 = bones_dict[bone_name]['tail']
                
                line_vec = p2 - p1
                p_vec = v - p1
                line_len_sq = np.dot(line_vec, line_vec)
                
                dist = 0
                if line_len_sq < 1e-9:
                    dist = np.linalg.norm(v - p1)
                else:
                    t = max(0, min(1, np.dot(p_vec, line_vec) / line_len_sq))
                    closest_point = p1 + t * line_vec
                    dist = np.linalg.norm(v - closest_point)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_bone_idx = bone_map[bone_name]

            joints_data[i, 0] = closest_bone_idx
            weights_data[i, 0] = 1.0

        return joints_data.flatten().tolist(), weights_data.flatten().tolist()

    def create_glb_with_texture(self, vertices, uvs, faces, armature_data, image_path, output_path="character.glb"):
        bone_order, bones_dict = armature_data
        if not bone_order:
            raise ValueError("Armature data is empty, cannot create GLB.")

        texture_data = None
        try:
            with open(image_path, 'rb') as f:
                texture_data = f.read()
            print(f"Loaded texture image: {len(texture_data)} bytes")
        except Exception as e:
            print(f"Warning: Could not load texture image: {e}")

        world_matrices = self._calculate_world_matrices(bone_order, bones_dict)
        inverse_bind_matrices_list = []
        for bone_name in bone_order:
            inv_matrix = self._invert_matrix(world_matrices.get(bone_name, np.identity(4)))
            inverse_bind_matrices_list.extend(inv_matrix.T.flatten())

        nodes = [{"mesh": 0, "skin": 0, "name": "CharacterMesh"}]
        bone_name_to_node_idx = {}
        root_bone_indices = []

        for i, bone_name in enumerate(bone_order):
            bone = bones_dict.get(bone_name)
            if not bone: continue
            node_idx = len(nodes)
            bone_name_to_node_idx[bone_name] = node_idx
            
            parent = bone.get('parent')
            parent_head = bones_dict.get(parent, {}).get('head', np.array([0,0,0]))
            translation = (bone['head'] - parent_head).tolist()

            nodes.append({"name": bone_name, "translation": translation, "children": []})
            if parent is None:
                root_bone_indices.append(node_idx)

        for bone_name, bone in bones_dict.items():
            parent_name = bone.get('parent')
            if parent_name and parent_name in bone_name_to_node_idx:
                parent_idx = bone_name_to_node_idx[parent_name]
                child_idx = bone_name_to_node_idx[bone_name]
                nodes[parent_idx]['children'].append(child_idx)
        
        for node in nodes:
            if "children" in node and not node["children"]:
                del node["children"]
        
        joint_node_indices = [bone_name_to_node_idx[name] for name in bone_order if name in bone_name_to_node_idx]

        joints_data, weights_data = self._automatic_skin_weights(vertices, bone_order, bones_dict, world_matrices)
        
        vert_bytes = struct.pack(f'<{len(vertices) * 3}f', *[c for v in vertices for c in v])
        uv_bytes = struct.pack(f'<{len(uvs) * 2}f', *[c for uv in uvs for c in uv])
        indices_flat = [i for f in faces for i in f]
        indices_bytes = struct.pack(f'<{len(indices_flat)}H', *indices_flat)
        ibm_bytes = struct.pack(f'<{len(inverse_bind_matrices_list)}f', *inverse_bind_matrices_list)
        joints_bytes = struct.pack(f'<{len(joints_data)}B', *joints_data)
        weights_bytes = struct.pack(f'<{len(weights_data)}f', *weights_data)

        buffer_chunks = [vert_bytes, uv_bytes, indices_bytes, ibm_bytes, joints_bytes, weights_bytes]
        
        texture_buffer_idx = None
        if texture_data:
            buffer_chunks.append(texture_data)
            texture_buffer_idx = len(buffer_chunks) - 1
            
        binary_blob = b''.join(buffer_chunks)
        
        while len(binary_blob) % 4 != 0:
            binary_blob += b'\x00'

        byte_offset = 0
        buffer_views = []
        for i, chunk in enumerate(buffer_chunks):
            buffer_view = {"buffer": 0, "byteOffset": byte_offset, "byteLength": len(chunk)}
            
            if i == 0: buffer_view['target'] = 34962
            elif i == 1: buffer_view['target'] = 34962
            elif i == 2: buffer_view['target'] = 34963
            elif i == 4: buffer_view['target'] = 34962
            elif i == 5: buffer_view['target'] = 34962
            
            buffer_views.append(buffer_view)
            byte_offset += len(chunk)

        accessors = [
            {"bufferView": 0, "componentType": 5126, "count": len(vertices), "type": "VEC3",
            "max": [max(v[0] for v in vertices), max(v[1] for v in vertices), max(v[2] for v in vertices)],
            "min": [min(v[0] for v in vertices), min(v[1] for v in vertices), min(v[2] for v in vertices)]},
            {"bufferView": 1, "componentType": 5126, "count": len(uvs), "type": "VEC2"},
            {"bufferView": 2, "componentType": 5123, "count": len(indices_flat), "type": "SCALAR"},
            {"bufferView": 3, "componentType": 5126, "count": len(joint_node_indices), "type": "MAT4"},
            {"bufferView": 4, "componentType": 5121, "count": len(vertices), "type": "VEC4"},
            {"bufferView": 5, "componentType": 5126, "count": len(vertices), "type": "VEC4"},
        ]

        materials, textures, images = [], [], []
        
        if texture_data:
            images.append({"bufferView": texture_buffer_idx, "mimeType": "image/png"})
            textures.append({"sampler": 0, "source": 0})
            materials.append({
                "name": "CharacterMaterial",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8
                },
                "doubleSided": True
            })

        mesh_primitive = {
            "attributes": {"POSITION": 0, "TEXCOORD_0": 1, "JOINTS_0": 4, "WEIGHTS_0": 5},
            "indices": 2, 
            "mode": 4
        }
        
        if materials:
            mesh_primitive["material"] = 0

        gltf_json = {
            "asset": {"version": "2.0", "generator": "Anime Character Processor"},
            "scene": 0,
            "scenes": [{"nodes": [0] + root_bone_indices}],
            "nodes": nodes,
            "meshes": [{"primitives": [mesh_primitive]}],
            "skins": [{"inverseBindMatrices": 3, "joints": joint_node_indices}],
            "buffers": [{"byteLength": len(binary_blob)}],
            "bufferViews": buffer_views,
            "accessors": accessors,
        }
        
        if materials:
            gltf_json["materials"] = materials
            gltf_json["textures"] = textures
            gltf_json["images"] = images
            gltf_json["samplers"] = [{"magFilter": 9729, "minFilter": 9987}]
        
        json_str = json.dumps(gltf_json, separators=(',', ':'))
        json_bytes = json_str.encode('utf-8')
        
        while len(json_bytes) % 4 != 0:
            json_bytes += b' '

        file_length = 12 + 8 + len(json_bytes) + 8 + len(binary_blob)
        with open(output_path, 'wb') as f:
            f.write(b'glTF')
            f.write(struct.pack('<I', 2))
            f.write(struct.pack('<I', file_length))
            f.write(struct.pack('<I', len(json_bytes)))
            f.write(b'JSON')
            f.write(json_bytes)
            f.write(struct.pack('<I', len(binary_blob)))
            f.write(b'BIN\x00')
            f.write(binary_blob)

def create_anime_character_faces(vertex_count):
    faces = []
    num_keypoints = 21 
    if vertex_count < num_keypoints: return faces

    # --- Vertex Index Definitions ---
    torso_start_idx = num_keypoints
    head_start_idx = torso_start_idx + 25
    neck_base_start_idx = head_start_idx + 12 
    limbs_start_idx = neck_base_start_idx + 12

    # 1. Create faces for the detailed torso grid
    if vertex_count >= head_start_idx:
        for i in range(4): 
            for j in range(4): 
                idx = torso_start_idx + (i * 5) + j
                v0, v1, v2, v3 = idx, idx + 1, idx + 5, idx + 6
                if (i + j) % 2 == 0:
                    faces.extend([[v0, v2, v1], [v1, v2, v3]])
                else:
                    faces.extend([[v0, v2, v3], [v0, v3, v1]])

    # 2. Create faces for the head fan
    if vertex_count >= neck_base_start_idx:
        for i in range(12):
            p1 = head_start_idx + i
            p2 = head_start_idx + ((i + 1) % 12)
            faces.append([1, p1, p2]) 

    # 3. Create faces for the Neck Cylinder
    if vertex_count >= limbs_start_idx:
        for i in range(12):
            h1 = head_start_idx + i
            h2 = head_start_idx + ((i + 1) % 12)
            n1 = neck_base_start_idx + i
            n2 = neck_base_start_idx + ((i + 1) % 12)
            faces.extend([[h1, n1, h2], [h2, n1, n2]])

    # 4. Create faces for the volumetric limbs
    num_limb_segments = 12 
    if vertex_count >= limbs_start_idx + (num_limb_segments * 4):
        for i in range(num_limb_segments):
            start_v_idx = limbs_start_idx + (i * 4)
            v0, v1, v2, v3 = start_v_idx, start_v_idx + 1, start_v_idx + 2, start_v_idx + 3
            faces.extend([[v0, v2, v1], [v1, v2, v3]])

    # 5. Stitch everything together
    # Connect neck base to torso
    for i in range(12):
        faces.append([2, neck_base_start_idx + i, neck_base_start_idx + ((i+1)%12)])
    
    # Connect Limbs and Torso to main joints
    l_shoulder, r_shoulder, l_hip, r_hip = 3, 4, 11, 12
    torso_tl, torso_tr = torso_start_idx, torso_start_idx + 4
    torso_bl, torso_br = torso_start_idx + 20, torso_start_idx + 24
    
    # Each limb now has 3 segments * 4 vertices = 12 vertices.
    l_arm = limbs_start_idx 
    r_arm = limbs_start_idx + 12 
    l_leg = limbs_start_idx + 24
    r_leg = limbs_start_idx + 36
    
    faces.extend([[l_shoulder, torso_tl, l_arm + 1], [l_shoulder, l_arm, torso_tl]])
    faces.extend([[r_shoulder, r_arm + 1, torso_tr], [r_shoulder, torso_tr, r_arm]])
    faces.extend([[l_hip, l_leg + 1, torso_bl], [l_hip, torso_bl, l_leg]])
    faces.extend([[r_hip, torso_br, r_leg + 1], [r_hip, r_leg, torso_br]])

    return faces
##Visualization 
def visualize_keypoints(image, keypoints, keypoint_names=None):
    vis_image = image.copy()
    
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.3:  # Only draw confident points
            cv2.circle(vis_image, (int(x), int(y)), 4, (0, 255, 0), -1)  # green dot
            if keypoint_names and i in keypoint_names:
                cv2.putText(vis_image, keypoint_names[i], (int(x)+5, int(y)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    return vis_image

skeleton = [
    (0, 1), (1, 2),        # head
    (2, 3), (2, 4),        # neck -> shoulders
    (3, 5), (5, 7), (7, 17), # left arm -> hand
    (4, 6), (6, 8), (8, 18), # right arm -> hand
    (2, 9), (9, 10),       # torso
    (10, 11), (10, 12),    # waist -> hips
    (11, 13), (13, 15), (15, 19), # left leg -> foot
    (12, 14), (14, 16), (16, 20)  # right leg -> foot
]

def draw_skeleton(image, keypoints, skeleton, threshold=0.3):
    vis_image = image.copy()
    for i, j in skeleton:
        if i < len(keypoints) and j < len(keypoints) and \
           keypoints[i][2] > threshold and keypoints[j][2] > threshold:
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(vis_image, pt1, pt2, (0, 0, 255), 2)
    return vis_image
def show_resized(window_name, image, max_width=800, max_height=800):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)  # don’t upscale if smaller
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    cv2.imshow(window_name, resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_vertices_on_image(image, vertices):
    vis_image = image.copy()
    height, width, _ = vis_image.shape

    for i, vertex in enumerate(vertices):
        norm_x, norm_y, _ = vertex

        pixel_x = int(((norm_x + 1) / 2) * width)
        pixel_y = int(((-norm_y + 1) / 2) * height)

        cv2.circle(vis_image, (pixel_x, pixel_y), radius=3, color=(0, 255, 0), thickness=-1) # Green dot
        
        cv2.putText(vis_image, str(i), (pixel_x + 5, pixel_y + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1) # Blue text

    cv2.imshow("Vertex Visualization", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_mesh_on_image(image, vertices, faces):
    vis_image = image.copy()
    height, width, _ = vis_image.shape

    for face in faces:
        if len(face) != 3:
            continue

        try:
            points = [vertices[i] for i in face]
        except IndexError:
            print(f"Warning: Face {face} contains an out-of-bounds vertex index. Skipping.")
            continue

        pixel_points = []
        for norm_x, norm_y, _ in points:
            px = int(((norm_x + 1) / 2) * width)
            py = int(((-norm_y + 1) / 2) * height)
            pixel_points.append([px, py])
        
        triangle_pts = np.array(pixel_points, np.int32)
        triangle_pts = triangle_pts.reshape((-1, 1, 2))

        cv2.polylines(vis_image, [triangle_pts], isClosed=True, color=(255, 255, 0), thickness=1)

    return vis_image
@app.route('/api/rig-character', methods=['POST'])
def process_character():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            file.save(temp_file.name)
            temp_image_path = temp_file.name
        
        try:
            processor = AnimeCharacterProcessor()
            keypoints, keypoint_scores, image = processor.process_image(temp_image_path)

            vertices, uvs = processor.create_mesh_vertices_with_uv(keypoints, image.shape)
            faces = create_anime_character_faces(len(vertices))

            # Optional Previews for debugging
            # vis = visualize_keypoints(image, keypoints, processor.anime_keypoints)
            # vis = draw_skeleton(vis, keypoints, skeleton)
            # show_resized("Pose Estimation", vis)
            # mesh_image = draw_mesh_on_image(image, vertices, faces)
            # show_resized("Mesh Preview", mesh_image)
            
            bones = processor.create_tpose_armature(keypoints, image.shape)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.glb') as glb_file:
                glb_file.close()  # Close the file so create_glb_with_texture can open it
                processor.create_glb_with_texture(vertices, uvs, faces, bones, temp_image_path, glb_file.name)
                glb_path = glb_file.name
            
            os.unlink(temp_image_path)
            
            return send_file(
                glb_path,
                as_attachment=True,
                download_name='mmpose_character.glb',
                mimetype='model/gltf-binary'
            )
            
        except Exception as e:
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            raise e
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'type': 'character_processor'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)