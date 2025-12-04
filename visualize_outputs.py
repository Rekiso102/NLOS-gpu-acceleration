"""
Script to create a table and visualization of NLOS reconstruction outputs
Shows the relationship between input parameters and output characteristics
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Define parameters for each FILE_ID
parameters = {
    1: {'scene': 'letter4', 'tag_maxdepth': 1.5, 'v_apt_sz': 2, 'depth_max': 1.5, 'd_offset': 0.1, 'exposure_ms': None},
    2: {'scene': 'resolutionbar', 'tag_maxdepth': 1.5, 'v_apt_sz': 2, 'depth_max': 1.5, 'd_offset': 0.1, 'exposure_ms': None},
    3: {'scene': 'NLOSletter', 'tag_maxdepth': 1.5, 'v_apt_sz': 2, 'depth_max': 1.5, 'd_offset': 0.1, 'exposure_ms': None},
    4: {'scene': 'shelf_targets_lighton', 'tag_maxdepth': 1.5, 'v_apt_sz': 2, 'depth_max': 1.5, 'd_offset': 0.1, 'exposure_ms': None},
    5: {'scene': 'letter44i', 'tag_maxdepth': 1.5, 'v_apt_sz': 2, 'depth_max': 1.5, 'd_offset': 0.1, 'exposure_ms': None},
    6: {'scene': 'officescene', 'tag_maxdepth': 3, 'v_apt_sz': 3, 'depth_max': 2.5, 'd_offset': 0.1, 'exposure_ms': None},
    7: {'scene': 'officescene_corrected', 'tag_maxdepth': 3, 'v_apt_sz': 3, 'depth_max': 3, 'd_offset': 0, 'exposure_ms': 1},
    8: {'scene': 'officescene_corrected', 'tag_maxdepth': 3, 'v_apt_sz': 3, 'depth_max': 3, 'd_offset': 0, 'exposure_ms': 5},
    9: {'scene': 'officescene_corrected', 'tag_maxdepth': 3, 'v_apt_sz': 3, 'depth_max': 3, 'd_offset': 0, 'exposure_ms': 10},
    10: {'scene': 'officescene_corrected', 'tag_maxdepth': 3, 'v_apt_sz': 3, 'depth_max': 3, 'd_offset': 0, 'exposure_ms': 20},
}

# Collect data for table
data = []
for file_id in range(1, 11):
    params = parameters[file_id]
    png_path = f'outputs/{file_id}/nlos_reconstruction_id{file_id}_output.png'

    # Check if output exists
    if os.path.exists(png_path):
        # Load image to get basic stats
        img = Image.open(png_path)
        img_array = np.array(img)

        # Calculate image statistics
        mean_intensity = np.mean(img_array)
        max_intensity = np.max(img_array)
        file_size_kb = os.path.getsize(png_path) / 1024

        status = 'Success'
    else:
        mean_intensity = None
        max_intensity = None
        file_size_kb = None
        status = 'Failed'

    data.append({
        'FILE_ID': file_id,
        'Scene': params['scene'],
        'Exposure (ms)': params['exposure_ms'] if params['exposure_ms'] else 'N/A',
        'Max Depth': params['tag_maxdepth'],
        'Aperture Size': params['v_apt_sz'],
        'Depth Max': params['depth_max'],
        'Offset': params['d_offset'],
        'Status': status,
        'Mean Intensity': f'{mean_intensity:.2f}' if mean_intensity else 'N/A',
        'Max Intensity': f'{max_intensity:.2f}' if max_intensity else 'N/A',
        'File Size (KB)': f'{file_size_kb:.2f}' if file_size_kb else 'N/A',
    })

# Create DataFrame
df = pd.DataFrame(data)

# Print table
print("\n" + "="*120)
print("NLOS RECONSTRUCTION OUTPUT SUMMARY")
print("="*120)
print(df.to_string(index=False))
print("="*120 + "\n")

# Save to CSV
df.to_csv('outputs/summary_table.csv', index=False)
print("Summary saved to outputs/summary_table.csv\n")

# Create visualization grid for successful outputs
successful_ids = [i for i in range(1, 11) if os.path.exists(f'outputs/{i}/nlos_reconstruction_id{i}_output.png')]

if successful_ids:
    n_images = len(successful_ids)
    cols = 3
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten() if n_images > 1 else [axes]

    for idx, file_id in enumerate(successful_ids):
        img = Image.open(f'outputs/{file_id}/nlos_reconstruction_id{file_id}_output.png')
        axes[idx].imshow(img)
        axes[idx].axis('off')

        params = parameters[file_id]
        title = f"ID {file_id}: {params['scene']}"
        if params['exposure_ms']:
            title += f"\nExposure: {params['exposure_ms']}ms"
        axes[idx].set_title(title, fontsize=10)

    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/reconstruction_grid.png', dpi=150, bbox_inches='tight')
    print("Visualization grid saved to outputs/reconstruction_grid.png\n")

    # Create exposure time comparison for officescene variants (if available)
    exposure_ids = [i for i in [7, 8, 9, 10] if i in successful_ids]
    if len(exposure_ids) >= 2:
        fig, axes = plt.subplots(1, len(exposure_ids), figsize=(5*len(exposure_ids), 5))
        if len(exposure_ids) == 1:
            axes = [axes]

        for idx, file_id in enumerate(exposure_ids):
            img = Image.open(f'outputs/{file_id}/nlos_reconstruction_id{file_id}_output.png')
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(f"Exposure: {parameters[file_id]['exposure_ms']}ms", fontsize=12)

        plt.suptitle('Effect of Exposure Time on Reconstruction Quality', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('outputs/exposure_comparison.png', dpi=150, bbox_inches='tight')
        print("Exposure comparison saved to outputs/exposure_comparison.png\n")

print("Done! Check the outputs/ directory for visualizations.")
