"""
CLEANUP SCRIPT - Sync Labels with Visualizations
=================================================
This script:
1. Checks which visualizations still exist
2. Removes JSON labels that don't have matching visualizations
3. Regenerates the all_labels.json file

Run this after manually removing bad images from visualizations folder.
"""

import os
import json
from pathlib import Path
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================

DATASETS = [
    {
        "name": "Human",
        "labels_folder": "./labeled_data/labels",
        "visualizations_folder": "./labeled_data/visualizations",
        "output_file": "./labeled_data/all_labels.json"
    },
    {
        "name": "Anime", 
        "labels_folder": "./labeled_data_anime/labels",
        "visualizations_folder": "./labeled_data_anime/visualizations",
        "output_file": "./labeled_data_anime/all_labels.json"
    }
]

KEYPOINT_NAMES = [
    'head_top', 'head_center', 'neck', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'chest_center',
    'waist_center', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_hand', 'right_hand', 'left_foot', 'right_foot'
]


def cleanup_dataset(dataset_config):
    """Clean up a single dataset."""
    name = dataset_config["name"]
    labels_folder = Path(dataset_config["labels_folder"])
    vis_folder = Path(dataset_config["visualizations_folder"])
    output_file = Path(dataset_config["output_file"])
    
    print(f"\n{'='*50}")
    print(f"Processing: {name} Dataset")
    print(f"{'='*50}")
    
    if not labels_folder.exists():
        print(f"ERROR: Labels folder not found: {labels_folder}")
        return None
    
    if not vis_folder.exists():
        print(f"ERROR: Visualizations folder not found: {vis_folder}")
        return None
    
    # Get list of existing visualizations (without _labeled.jpg suffix)
    existing_vis = set()
    for f in vis_folder.iterdir():
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Remove "_labeled" suffix to get original name
            stem = f.stem
            if stem.endswith('_labeled'):
                stem = stem[:-8]  # Remove "_labeled"
            existing_vis.add(stem)
    
    print(f"Found {len(existing_vis)} visualizations")
    
    # Check each JSON label file
    labels_to_keep = []
    labels_to_remove = []
    
    for json_file in labels_folder.iterdir():
        if json_file.suffix == '.json':
            stem = json_file.stem
            
            if stem in existing_vis:
                # Visualization exists, keep this label
                labels_to_keep.append(json_file)
            else:
                # Visualization was deleted, remove this label
                labels_to_remove.append(json_file)
    
    print(f"Labels to keep: {len(labels_to_keep)}")
    print(f"Labels to remove: {len(labels_to_remove)}")
    
    # Remove orphaned JSON files
    if labels_to_remove:
        print(f"\nRemoving {len(labels_to_remove)} orphaned label files...")
        for json_file in labels_to_remove:
            print(f"  Deleting: {json_file.name}")
            json_file.unlink()
    
    # Regenerate all_labels.json
    print(f"\nRegenerating {output_file.name}...")
    
    all_labels = []
    success_count = 0
    error_count = 0
    
    for json_file in labels_to_keep:
        try:
            with open(json_file, 'r') as f:
                label_data = json.load(f)
                all_labels.append(label_data)
                success_count += 1
        except Exception as e:
            print(f"  Error reading {json_file.name}: {e}")
            error_count += 1
    
    # Save new all_labels.json
    output_data = {
        'total_images': len(all_labels),
        'successful': len(all_labels),
        'failed': 0,
        'low_confidence_count': 0,  # Will need manual update if needed
        'keypoint_names': KEYPOINT_NAMES,
        'cleaned_at': datetime.now().isoformat(),
        'labels': all_labels
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved: {output_file}")
    print(f"Total labels: {len(all_labels)}")
    
    return {
        'name': name,
        'kept': len(labels_to_keep),
        'removed': len(labels_to_remove),
        'final_count': len(all_labels)
    }


def main():
    print("\n" + "="*50)
    print("  CLEANUP SCRIPT")
    print("  Sync Labels with Visualizations")
    print("="*50)
    
    results = []
    
    for dataset in DATASETS:
        result = cleanup_dataset(dataset)
        if result:
            results.append(result)
    
    # Print summary
    print("\n" + "="*50)
    print("  CLEANUP COMPLETE - SUMMARY")
    print("="*50)
    
    total_kept = 0
    total_removed = 0
    
    for r in results:
        print(f"\n{r['name']} Dataset:")
        print(f"  - Labels kept: {r['kept']}")
        print(f"  - Labels removed: {r['removed']}")
        print(f"  - Final count: {r['final_count']}")
        total_kept += r['kept']
        total_removed += r['removed']
    
    print(f"\n{'='*50}")
    print(f"TOTAL: {total_kept} labels kept, {total_removed} removed")
    print(f"READY FOR CNN TRAINING: {total_kept} images")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()