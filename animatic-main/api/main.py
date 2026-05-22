"""
ANIMATIC API - 2D to 3D Character Rigging
==========================================
This API converts 2D character images to rigged 3D GLB models.

Now powered by a custom-trained CNN for keypoint detection!

Author: Angelo Tristan Sinohin
Thesis: 2D to 3D Character Rigging using CNN
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
import cv2
import numpy as np
import os
import tempfile
import json
import io
from PIL import Image
import struct
import warnings
warnings.filterwarnings('ignore')

# Import CNN predictor
from cnn_predictor import CNNKeypointPredictor, HybridKeypointPredictor, load_predictor
from validation import MAX_REQUEST_SIZE_BYTES, UploadValidationError, validate_uploaded_image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_REQUEST_SIZE_BYTES
CORS(app)

# Global predictor instance
PREDICTOR = None


def api_error(code, message, status_code=400):
    return jsonify({'error': {'code': code, 'message': message}}), status_code


@app.errorhandler(RequestEntityTooLarge)
def handle_request_too_large(_error):
    return api_error('file_too_large', 'Please upload an image smaller than 10 MB.', 413)

def get_predictor():
    """Get or initialize the keypoint predictor."""
    global PREDICTOR
    if PREDICTOR is None:
        model_path = os.environ.get('CNN_MODEL_PATH', 'trained_model/best_model.keras')
        use_fallback = os.environ.get('USE_MEDIAPIPE_FALLBACK', 'true').lower() == 'true'
        
        print(f"Initializing predictor with model: {model_path}")
        print(f"MediaPipe fallback: {use_fallback}")
        
        PREDICTOR = load_predictor(model_path, use_fallback)
    
    return PREDICTOR


class CNNCharacterProcessor:
    """
    Character processor using CNN for keypoint detection.
    
    This replaces the MediaPipe-based AnimeCharacterProcessor with
    a custom-trained CNN model.
    """
    
    def __init__(self):
        self.predictor = get_predictor()
        
        # Keypoint names (same as training)
        self.anime_keypoints = {
            0: 'head_top', 1: 'head_center', 2: 'neck', 
            3: 'left_shoulder', 4: 'right_shoulder',
            5: 'left_elbow', 6: 'right_elbow', 
            7: 'left_wrist', 8: 'right_wrist',
            9: 'chest_center', 10: 'waist_center', 
            11: 'left_hip', 12: 'right_hip',
            13: 'left_knee', 14: 'right_knee', 
            15: 'left_ankle', 16: 'right_ankle',
            17: 'left_hand', 18: 'right_hand', 
            19: 'left_foot', 20: 'right_foot'
        }
        
        print("CNN Character Processor initialized")

    def process_image(self, image_path):
        """
        Process an image and extract keypoints using CNN.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            tuple: (keypoints, confidence_scores, original_image)
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Could not read image file")
            
            # Get predictions from CNN (or fallback)
            keypoints, method = self.predictor.predict(image)
            print(f"Keypoints detected using: {method}")
            
            # Extract confidence scores
            confidence_scores = keypoints[:, 2]
            
            # Validate keypoints
            valid_keypoints = keypoints[keypoints[:, 2] > 0.3]
            if len(valid_keypoints) < 8:
                raise ValueError(
                    "Could not detect enough character features. "
                    "Please try with a clearer, full-body character image."
                )
            
            return keypoints, confidence_scores, image
            
        except Exception as e:
            raise ValueError(f"Character processing failed: {str(e)}")

    def create_mesh_vertices_with_uv(self, keypoints, image_shape):
        """Create mesh vertices and UV coordinates from keypoints."""
        height, width = image_shape[:2]
        vertices = []
        uvs = []
        
        # Create vertices from keypoints
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
        
        # Generate character mesh
        additional_vertices, additional_uvs = self.generate_character_mesh_with_uv(
            keypoints, image_shape, arm_width=25.0, leg_width=30.0
        )
        vertices.extend(additional_vertices)
        uvs.extend(additional_uvs)
        
        return vertices, uvs

    def generate_character_mesh_with_uv(self, keypoints, image_shape, arm_width=8.0, leg_width=12.0):
        """Generate detailed character mesh from keypoints."""
        body_vertices = []
        body_uvs = []
        height, width = image_shape[:2]
        np_keypoints = np.array([kp[:2] for kp in keypoints])

        # Torso grid
        ls, rs = np_keypoints[3], np_keypoints[4]  # shoulders
        lh, rh = np_keypoints[11], np_keypoints[12]  # hips
        
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

        # Head circle
        head_center, neck = np_keypoints[1], np_keypoints[2]
        head_radius_pixels = np.linalg.norm(head_center - neck)
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            x = head_center[0] + head_radius_pixels * np.cos(angle)
            y = head_center[1] + (head_radius_pixels * np.sin(angle)) * (width / height)
            body_vertices.append(np.array([x, y]))
            
            uv_x = x / width
            uv_y = y / height
            body_uvs.append([float(uv_x), float(uv_y)])

        # Neck cylinder
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

        # Limbs
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
        
        # Normalize vertices
        normalized_vertices = []
        for x, y in body_vertices:
            norm_x = (x / width) * 2 - 1
            norm_y = -((y / height) * 2 - 1)
            normalized_vertices.append([float(norm_x), float(norm_y), 0.0])
            
        return normalized_vertices, body_uvs

    def _calculate_world_matrices(self, bone_order, bones_dict):
        """Calculate world transformation matrices for bones."""
        world_matrices = {}
        for bone_name in bone_order:
            if bone_name not in bones_dict:
                continue
            
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
        """Invert a matrix with error handling."""
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return np.identity(4)

    def create_tpose_armature(self, keypoints, image_shape):
        """Create T-pose armature from keypoints."""
        height, width = image_shape[:2]
        
        def get_keypoint_pos(idx):
            if idx < len(keypoints) and keypoints[idx][2] > 0.3:
                x, y, _ = keypoints[idx]
                norm_x = (x / width) * 2 - 1
                norm_y = -((y / height) * 2 - 1)
                return np.array([float(norm_x), float(norm_y), 0.0])
            return None

        # Get all keypoint positions
        pos = {name: get_keypoint_pos(idx) for idx, name in self.anime_keypoints.items()}
        
        # Calculate derived positions
        if pos['left_hip'] is not None and pos['right_hip'] is not None:
            pos['hip_center'] = (pos['left_hip'] + pos['right_hip']) / 2
        else:
            pos['hip_center'] = None
            
        if pos['left_shoulder'] is not None and pos['right_shoulder'] is not None:
            pos['shoulder_center'] = (pos['left_shoulder'] + pos['right_shoulder']) / 2
        else:
            pos['shoulder_center'] = None

        if pos['hip_center'] is None or pos['shoulder_center'] is None:
            print("ERROR: Core points (hips or shoulders) not detected.")
            return [], {}
        
        pos['chest'] = pos['shoulder_center']
        
        if pos['neck'] is None:
            pos['neck'] = pos['chest']
        if pos['chest_center'] is None:
            pos['chest_center'] = (pos['chest'] + pos['hip_center']) / 2.0
        
        # Interpolate missing limb joints
        if pos['left_elbow'] is None and pos['left_shoulder'] is not None and pos['left_wrist'] is not None:
            pos['left_elbow'] = (pos['left_shoulder'] + pos['left_wrist']) / 2.0
        if pos['right_elbow'] is None and pos['right_shoulder'] is not None and pos['right_wrist'] is not None:
            pos['right_elbow'] = (pos['right_shoulder'] + pos['right_wrist']) / 2.0
        if pos['left_knee'] is None and pos['left_hip'] is not None and pos['left_ankle'] is not None:
            pos['left_knee'] = (pos['left_hip'] + pos['left_ankle']) / 2.0
        if pos['right_knee'] is None and pos['right_hip'] is not None and pos['right_ankle'] is not None:
            pos['right_knee'] = (pos['right_hip'] + pos['right_ankle']) / 2.0
            
        # Check required points
        bone_points = [
            'hip_center', 'chest_center', 'chest', 'neck', 'head_center', 
            'left_shoulder', 'left_elbow', 'left_wrist', 'left_hand',
            'right_shoulder', 'right_elbow', 'right_wrist', 'right_hand',
            'left_hip', 'left_knee', 'left_ankle', 'left_foot',
            'right_hip', 'right_knee', 'right_ankle', 'right_foot'
        ]
        
        for p_name in bone_points:
            if pos.get(p_name) is None:
                print(f"FATAL: Could not determine position for '{p_name}'.")
                return [], {}

        # Define bone hierarchy
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
        
        print(f"Created armature with {len(final_bones)} bones.")
        return bone_order, final_bones

    def _automatic_skin_weights(self, vertices, bone_order, bones_dict, world_matrices):
        """Calculate automatic skin weights for vertices."""
        num_vertices = len(vertices)
        joints_data = np.zeros((num_vertices, 4), dtype=np.uint8)
        weights_data = np.zeros((num_vertices, 4), dtype=np.float32)
        bone_map = {name: i for i, name in enumerate(bone_order)}

        for i, vertex in enumerate(vertices):
            v = np.array(vertex)
            
            min_dist = float('inf')
            closest_bone_idx = 0
            
            for bone_name in bone_order:
                if bone_name not in bones_dict:
                    continue
                
                p1 = bones_dict[bone_name]['head']
                p2 = bones_dict[bone_name]['tail']
                
                line_vec = p2 - p1
                p_vec = v - p1
                line_len_sq = np.dot(line_vec, line_vec)
                
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
        """Create GLB file with texture and armature."""
        bone_order, bones_dict = armature_data
        if not bone_order:
            raise ValueError("Armature data is empty, cannot create GLB.")

        # Load texture
        texture_data = None
        try:
            with open(image_path, 'rb') as f:
                texture_data = f.read()
            print(f"Loaded texture image: {len(texture_data)} bytes")
        except Exception as e:
            print(f"Warning: Could not load texture image: {e}")

        # Calculate matrices
        world_matrices = self._calculate_world_matrices(bone_order, bones_dict)
        inverse_bind_matrices_list = []
        for bone_name in bone_order:
            inv_matrix = self._invert_matrix(world_matrices.get(bone_name, np.identity(4)))
            inverse_bind_matrices_list.extend(inv_matrix.T.flatten())

        # Build nodes
        nodes = [{"mesh": 0, "skin": 0, "name": "CharacterMesh"}]
        bone_name_to_node_idx = {}
        root_bone_indices = []

        for i, bone_name in enumerate(bone_order):
            bone = bones_dict.get(bone_name)
            if not bone:
                continue
            node_idx = len(nodes)
            bone_name_to_node_idx[bone_name] = node_idx
            
            parent = bone.get('parent')
            parent_head = bones_dict.get(parent, {}).get('head', np.array([0, 0, 0]))
            translation = (bone['head'] - parent_head).tolist()

            nodes.append({"name": bone_name, "translation": translation, "children": []})
            if parent is None:
                root_bone_indices.append(node_idx)

        # Set up parent-child relationships
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

        # Calculate skin weights
        joints_data, weights_data = self._automatic_skin_weights(vertices, bone_order, bones_dict, world_matrices)
        
        # Pack binary data
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

        # Build buffer views
        byte_offset = 0
        buffer_views = []
        for i, chunk in enumerate(buffer_chunks):
            buffer_view = {"buffer": 0, "byteOffset": byte_offset, "byteLength": len(chunk)}
            
            if i == 0:
                buffer_view['target'] = 34962
            elif i == 1:
                buffer_view['target'] = 34962
            elif i == 2:
                buffer_view['target'] = 34963
            elif i == 4:
                buffer_view['target'] = 34962
            elif i == 5:
                buffer_view['target'] = 34962
            
            buffer_views.append(buffer_view)
            byte_offset += len(chunk)

        # Build accessors
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

        # Build materials and textures
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

        # Build glTF JSON
        gltf_json = {
            "asset": {"version": "2.0", "generator": "Animatic CNN Character Processor"},
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

        # Write GLB file
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


def create_character_faces(vertex_count):
    """Create faces for the character mesh."""
    faces = []
    num_keypoints = 21
    if vertex_count < num_keypoints:
        return faces

    # Vertex index definitions
    torso_start_idx = num_keypoints
    head_start_idx = torso_start_idx + 25
    neck_base_start_idx = head_start_idx + 12
    limbs_start_idx = neck_base_start_idx + 12

    # Torso grid faces
    if vertex_count >= head_start_idx:
        for i in range(4):
            for j in range(4):
                idx = torso_start_idx + (i * 5) + j
                v0, v1, v2, v3 = idx, idx + 1, idx + 5, idx + 6
                if (i + j) % 2 == 0:
                    faces.extend([[v0, v2, v1], [v1, v2, v3]])
                else:
                    faces.extend([[v0, v2, v3], [v0, v3, v1]])

    # Head fan faces
    if vertex_count >= neck_base_start_idx:
        for i in range(12):
            p1 = head_start_idx + i
            p2 = head_start_idx + ((i + 1) % 12)
            faces.append([1, p1, p2])

    # Neck cylinder faces
    if vertex_count >= limbs_start_idx:
        for i in range(12):
            h1 = head_start_idx + i
            h2 = head_start_idx + ((i + 1) % 12)
            n1 = neck_base_start_idx + i
            n2 = neck_base_start_idx + ((i + 1) % 12)
            faces.extend([[h1, n1, h2], [h2, n1, n2]])

    # Limb faces
    num_limb_segments = 12
    if vertex_count >= limbs_start_idx + (num_limb_segments * 4):
        for i in range(num_limb_segments):
            start_v_idx = limbs_start_idx + (i * 4)
            v0, v1, v2, v3 = start_v_idx, start_v_idx + 1, start_v_idx + 2, start_v_idx + 3
            faces.extend([[v0, v2, v1], [v1, v2, v3]])

    # Stitching
    for i in range(12):
        faces.append([2, neck_base_start_idx + i, neck_base_start_idx + ((i + 1) % 12)])
    
    # Connect limbs to torso
    l_shoulder, r_shoulder, l_hip, r_hip = 3, 4, 11, 12
    torso_tl, torso_tr = torso_start_idx, torso_start_idx + 4
    torso_bl, torso_br = torso_start_idx + 20, torso_start_idx + 24
    
    l_arm = limbs_start_idx
    r_arm = limbs_start_idx + 12
    l_leg = limbs_start_idx + 24
    r_leg = limbs_start_idx + 36
    
    faces.extend([[l_shoulder, torso_tl, l_arm + 1], [l_shoulder, l_arm, torso_tl]])
    faces.extend([[r_shoulder, r_arm + 1, torso_tr], [r_shoulder, torso_tr, r_arm]])
    faces.extend([[l_hip, l_leg + 1, torso_bl], [l_hip, torso_bl, l_leg]])
    faces.extend([[r_hip, torso_br, r_leg + 1], [r_hip, r_leg, torso_br]])

    return faces


# ============================================
# API ENDPOINTS
# ============================================

@app.route('/api/rig-character', methods=['POST'])
def process_character():
    """Main endpoint to process a character image and return rigged GLB."""
    temp_image_path = None
    glb_path = None

    if 'image' not in request.files:
        return api_error('missing_file', 'No image file was provided.')

    file = request.files['image']
    try:
        validate_uploaded_image(file, request.content_length)
    except UploadValidationError as e:
        return api_error(e.code, e.message, e.status_code)

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            file.save(temp_file.name)
            temp_image_path = temp_file.name

        # Process with CNN
        processor = CNNCharacterProcessor()
        keypoints, keypoint_scores, image = processor.process_image(temp_image_path)

        # Create mesh
        vertices, uvs = processor.create_mesh_vertices_with_uv(keypoints, image.shape)
        faces = create_character_faces(len(vertices))

        # Create armature
        bones = processor.create_tpose_armature(keypoints, image.shape)

        # Generate GLB
        with tempfile.NamedTemporaryFile(delete=False, suffix='.glb') as glb_file:
            glb_file.close()
            processor.create_glb_with_texture(vertices, uvs, faces, bones, temp_image_path, glb_file.name)
            glb_path = glb_file.name

        with open(glb_path, 'rb') as f:
            glb_bytes = f.read()

        return send_file(
            io.BytesIO(glb_bytes),
            as_attachment=True,
            download_name='cnn_rigged_character.glb',
            mimetype='model/gltf-binary'
        )

    except Exception as e:
        app.logger.exception("Character generation failed")
        return api_error(
            'generation_failed',
            'Unable to generate a GLB from this image. Please try a clearer full-body character image.',
            500
        )
    finally:
        for path in (temp_image_path, glb_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    app.logger.warning("Could not remove temporary file: %s", path)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    predictor = get_predictor()
    
    status = {
        'status': 'healthy',
        'type': 'cnn_character_processor',
        'cnn_loaded': predictor.cnn_predictor is not None,
        'mediapipe_fallback': predictor.mediapipe_available
    }
    
    return jsonify(status)


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return information about the loaded model."""
    return jsonify({
        'model_type': 'CNN Keypoint Detection',
        'architecture': '5 Conv Blocks + 3 Dense Layers',
        'input_size': [256, 256, 3],
        'output': '21 keypoints (42 values)',
        'training_accuracy': '86.26% PCK @ 0.1',
        'training_samples': 2191
    })


if __name__ == '__main__':
    # Initialize predictor on startup
    print("Initializing CNN Character Processor...")
    get_predictor()
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
