
import numpy as np
import time
from sklearn.metrics import mean_squared_error

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_pck(pred_kp, true_kp, threshold_ratio=0.2):
    torso = distance(true_kp['left_shoulder'], true_kp['right_hip'])
    threshold = torso * threshold_ratio

    correct = 0
    total = len(true_kp)

    for key in true_kp:
        if distance(pred_kp[key], true_kp[key]) <= threshold:
            correct += 1

    return correct / total

def bone_length_score(pred_kp, true_kp, bones):
    scores = []
    for (a, b) in bones:
        pred_len = distance(pred_kp[a], pred_kp[b])
        true_len = distance(true_kp[a], true_kp[b])
        ratio = pred_len / true_len if true_len > 0 else 0
        scores.append(ratio)
    return np.mean(scores)

def evaluate_rig_model(model, test_images, true_keypoints, bones):
    all_mse = []
    all_pck = []
    all_struct_scores = []
    inference_times = []

    for img, true_kp in zip(test_images, true_keypoints):
        start = time.time()
        pred_kp = model.predict(img)
        end = time.time()

        mse = mean_squared_error(
            np.array(list(true_kp.values())),
            np.array(list(pred_kp.values()))
        )

        pck = compute_pck(pred_kp, true_kp)
        struct_score = bone_length_score(pred_kp, true_kp, bones)
        inference = end - start

        all_mse.append(mse)
        all_pck.append(pck)
        all_struct_scores.append(struct_score)
        inference_times.append(inference)

    return {
        "mse": float(np.mean(all_mse)),
        "pck": float(np.mean(all_pck)),
        "structural_consistency": float(np.mean(all_struct_scores)),
        "avg_inference_time": float(np.mean(inference_times)),
        "throughput_fps": float(1 / np.mean(inference_times))
    }
