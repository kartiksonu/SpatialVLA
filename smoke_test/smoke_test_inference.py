#!/usr/bin/env python3
"""
SpatialVLA Smoke Test: Episode inference script with BridgeDataV2.

Loads SpatialVLA model and runs inference on episodes from BridgeDataV2 dataset.
Generates comparison plots and GIFs for each episode.

Usage:
    # Run Episode 0 (first episode) - default
    python smoke_test/smoke_test_inference.py

    # Run Episode 5 (0-indexed, so 6th episode)
    python smoke_test/smoke_test_inference.py --episode_index 5

    # Run Episode 10 with custom output directory
    python smoke_test/smoke_test_inference.py --episode_index 10 --output_dir ./my_outputs

    # Count total episodes in the tfrecord file
    python smoke_test/smoke_test_inference.py --count_episodes

Note: Each tfrecord file contains multiple episodes (~50 episodes per file).
      Episode indices are 0-indexed (episode_index 0 = first episode).
"""

import os
import tensorflow as tf
import torch
import numpy as np
from PIL import Image
import io
from transformers import AutoModel, AutoProcessor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from pathlib import Path

# Get script directory and project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent  # smoke_test/ -> project root

TFRECORD_PATH = SCRIPT_DIR / "data/bridge_dataset-train.tfrecord-00000-of-01024"
MODEL_PATH = PROJECT_ROOT / "pretrained/spatialvla-4b-224-pt"

def decode_image(image_data):
    try:
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def count_episodes(tfrecord_path):
    """Count total number of episodes in a tfrecord file."""
    tfrecord_path = str(tfrecord_path)
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    count = 0
    for _ in dataset:
        count += 1
    return count

def extract_episode_from_tfrecord(tfrecord_path, episode_index=0):
    """
    Extract a specific episode (all steps) from TFRecord.

    Args:
        tfrecord_path: Path to the tfrecord file
        episode_index: Which episode to load (0-indexed). Default is 0 (first episode).

    Returns:
        images, gt_actions, instruction, ep_id, total_episodes
    """
    tfrecord_path = str(tfrecord_path)
    print(f"Attempting to read from {tfrecord_path}...")
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    # Iterate through episodes to find the target one
    target_data = None
    total_episodes = 0

    for idx, raw_record in enumerate(dataset):
        total_episodes += 1

        # Only process the episode we want
        if idx == episode_index:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            features = example.features.feature

            # Extract instruction
            instruction = "do something"
            if 'steps/language_instruction' in features:
                bytes_list = features['steps/language_instruction'].bytes_list.value
                if len(bytes_list) > 0:
                    instruction = bytes_list[0].decode('utf-8')

            # Extract all images from episode
            images = []
            if 'steps/observation/image_0' in features:
                bytes_list = features['steps/observation/image_0'].bytes_list.value
                for img_bytes in bytes_list:
                    img = decode_image(img_bytes)
                    if img is not None:
                        images.append(img)

            # Extract ground truth actions
            gt_actions = []
            if 'steps/action' in features:
                action_floats = features['steps/action'].float_list.value
                num_steps = len(images)
                if num_steps > 0:
                    dim = len(action_floats) // num_steps
                    gt_actions = np.array(action_floats).reshape(num_steps, dim)

            # Extract episode ID
            ep_id = "N/A"
            if 'episode_metadata/episode_id' in features:
                feat_id = features['episode_metadata/episode_id']
                if feat_id.int64_list.value:
                    ep_id = str(feat_id.int64_list.value[0])

            print(f"Episode Index: {episode_index}/{total_episodes-1} (0-indexed)")
            print(f"Episode ID: {ep_id}")
            print(f"Instruction: {instruction}")
            print(f"Number of steps: {len(images)}")

            if images:
                target_data = (images, gt_actions, instruction, ep_id, total_episodes)
                break

    if target_data is None:
        if episode_index >= total_episodes:
            print(f"ERROR: episode_index {episode_index} >= total episodes {total_episodes}")
        return None, None, None, None, total_episodes

    images, gt_actions, instruction, ep_id, total_episodes = target_data
    return images, gt_actions, instruction, ep_id, total_episodes

def run_inference_on_episode(model, processor, images, instruction, device, max_steps=None):
    """Run inference on all images in episode."""
    if max_steps:
        images = images[:max_steps]

    pred_actions = []
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    print(f"\nRunning inference on {len(images)} steps...")
    for i, image in enumerate(images):
        inputs = processor(images=[image.convert("RGB")], text=prompt, return_tensors="pt").to(device, torch.bfloat16)

        with torch.no_grad():
            generation_outputs = model.predict_action(inputs)

        actions_dict = processor.decode_actions(generation_outputs, unnorm_key="bridge_orig/1.0.0")
        # decode_actions returns dict with "actions" (numpy array) and "action_ids"
        # actions shape is (action_chunk_size, action_dim), we take the first action
        actions = actions_dict["actions"]
        pred_actions.append(actions[0])  # Extract first action from chunk

        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{len(images)} steps...")

    return np.array(pred_actions)

def generate_plots(images, gt_actions, pred_actions, instruction, output_dir, episode_name):
    """Generate comparison plots and GIF for an episode."""
    num_steps = len(images)

    # Compute cumulative EE positions from action deltas
    gt_positions = np.cumsum(gt_actions[:, :3], axis=0)
    pred_positions = np.cumsum(pred_actions[:, :3], axis=0)

    # ========== 1. Static comparison plot ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Episode: '{instruction}'\n{episode_name}", fontsize=12)

    # Position components
    for i, (label, color) in enumerate(zip(['X', 'Y', 'Z'], ['r', 'g', 'b'])):
        axes[0, i].plot(gt_actions[:, i], f'{color}-', label='Ground Truth', linewidth=2)
        axes[0, i].plot(pred_actions[:, i], f'{color}--', label='Predicted', linewidth=2)
        axes[0, i].set_xlabel('Step')
        axes[0, i].set_ylabel(f'{label} Delta (m)')
        axes[0, i].set_title(f'Position {label}')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)

    # Rotation components
    for i, (label, color) in enumerate(zip(['Roll', 'Pitch', 'Yaw'], ['r', 'g', 'b'])):
        axes[1, i].plot(gt_actions[:, i+3], f'{color}-', label='Ground Truth', linewidth=2)
        axes[1, i].plot(pred_actions[:, i+3], f'{color}--', label='Predicted', linewidth=2)
        axes[1, i].set_xlabel('Step')
        axes[1, i].set_ylabel(f'{label} Delta (rad)')
        axes[1, i].set_title(f'Rotation {label}')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'action_comparison.png', dpi=150)
    plt.close()

    # ========== 2. 3D Trajectory plot ==========
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2],
            'b-', linewidth=2, label='Ground Truth')
    ax.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2],
            'r--', linewidth=2, label='Predicted')

    ax.scatter(*gt_positions[0], c='green', s=150, marker='*', label='Start')
    ax.scatter(*gt_positions[-1], c='blue', s=100, marker='o', label='GT End')
    ax.scatter(*pred_positions[-1], c='red', s=100, marker='^', label='Pred End')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"3D End-Effector Trajectory\n'{instruction}'")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_3d.png', dpi=150)
    plt.close()

    # ========== 3. Gripper comparison ==========
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gt_actions[:, 6], 'b-', label='Ground Truth', linewidth=2, marker='o')
    ax.plot(pred_actions[:, 6], 'r--', label='Predicted', linewidth=2, marker='^')
    ax.axhline(y=0.5, color='gray', linestyle=':', label='Open/Close threshold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Gripper Value')
    ax.set_title(f"Gripper Command\n'{instruction}'")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(output_dir / 'gripper_comparison.png', dpi=150)
    plt.close()

    # ========== 4. Animated GIF ==========
    fig = plt.figure(figsize=(14, 6))
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_img = fig.add_subplot(122)

    all_pos = np.vstack([gt_positions, pred_positions])
    margin = 0.02
    x_min, x_max = all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin
    y_min, y_max = all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin
    z_min, z_max = all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin

    def update(frame):
        ax_3d.clear()
        ax_img.clear()

        ax_3d.set_xlim([x_min, x_max])
        ax_3d.set_ylim([y_min, y_max])
        ax_3d.set_zlim([z_min, z_max])

        if frame > 0:
            ax_3d.plot(gt_positions[:frame+1, 0], gt_positions[:frame+1, 1], gt_positions[:frame+1, 2],
                       'b-', linewidth=2, label='Ground Truth')
            ax_3d.plot(pred_positions[:frame+1, 0], pred_positions[:frame+1, 1], pred_positions[:frame+1, 2],
                       'r--', linewidth=2, label='Predicted')

        ax_3d.scatter(*gt_positions[frame], c='blue', s=100, marker='o', edgecolors='black', zorder=5)
        ax_3d.scatter(*pred_positions[frame], c='red', s=100, marker='^', edgecolors='black', zorder=5)
        ax_3d.scatter(*gt_positions[0], c='green', s=150, marker='*', edgecolors='black', label='Start')

        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_title(f'End-Effector Trajectory\nStep {frame+1}/{num_steps}')
        ax_3d.legend(loc='upper left')
        ax_3d.view_init(elev=20, azim=45 + frame * 2)

        ax_img.imshow(images[frame])
        ax_img.set_title(f"Camera View - Step {frame+1}\n'{instruction}'")
        ax_img.axis('off')

        gt_grip = "open" if gt_actions[frame, 6] > 0.5 else "close"
        pred_grip = "open" if pred_actions[frame, 6] > 0.5 else "close"
        info_text = f"GT: [{gt_actions[frame, 0]:.3f}, {gt_actions[frame, 1]:.3f}, {gt_actions[frame, 2]:.3f}] grip:{gt_grip}\n"
        info_text += f"Pred: [{pred_actions[frame, 0]:.3f}, {pred_actions[frame, 1]:.3f}, {pred_actions[frame, 2]:.3f}] grip:{pred_grip}"
        ax_img.text(0.5, -0.1, info_text, transform=ax_img.transAxes, fontsize=9,
                    ha='center', va='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        return []

    anim = animation.FuncAnimation(fig, update, frames=num_steps, interval=800, blit=False)
    anim.save(output_dir / 'trajectory.gif', writer='pillow', fps=1.5)
    plt.close()

    # ========== 5. Save statistics ==========
    stats = {
        'episode_name': episode_name,
        'instruction': instruction,
        'num_steps': num_steps,
        'position_errors': [np.linalg.norm(pred_actions[i, :3] - gt_actions[i, :3]) for i in range(num_steps)],
        'rotation_errors': [np.linalg.norm(pred_actions[i, 3:6] - gt_actions[i, 3:6]) for i in range(num_steps)],
        'total_errors': [np.linalg.norm(pred_actions[i] - gt_actions[i]) for i in range(num_steps)],
    }
    stats['mean_position_error'] = np.mean(stats['position_errors'])
    stats['mean_rotation_error'] = np.mean(stats['rotation_errors'])
    stats['mean_total_error'] = np.mean(stats['total_errors'])

    with open(output_dir / 'statistics.txt', 'w') as f:
        f.write(f"Episode: {episode_name}\n")
        f.write(f"Instruction: {instruction}\n")
        f.write(f"Number of steps: {num_steps}\n")
        f.write(f"\nMean Position Error: {stats['mean_position_error']:.4f}\n")
        f.write(f"Mean Rotation Error: {stats['mean_rotation_error']:.4f}\n")
        f.write(f"Mean Total Error: {stats['mean_total_error']:.4f}\n")
        f.write(f"\nPer-step errors:\n")
        for i in range(num_steps):
            f.write(f"  Step {i+1}: pos={stats['position_errors'][i]:.4f}, rot={stats['rotation_errors'][i]:.4f}, total={stats['total_errors'][i]:.4f}\n")

    return stats

def main(output_dir: str = None, episode_index: int = 0, count_episodes_only: bool = False):
    """
    Main function to run SpatialVLA inference on a specific episode.

    Args:
        output_dir: Directory to save outputs (default: scripts/outputs/episode_{index})
        episode_index: Episode index to process (0-indexed, default: 0)
        count_episodes_only: If True, only count episodes and exit
    """
    # Count episodes if requested
    if count_episodes_only:
        total = count_episodes(TFRECORD_PATH)
        print(f"\nTotal episodes in {TFRECORD_PATH.name}: {total}")
        print(f"Valid episode indices: 0 to {total-1}")
        return

    # Setup output directory
    if output_dir is None:
        output_dir = SCRIPT_DIR / "outputs" / f"episode_{episode_index}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("="*70)
    print(f"SPATIALVLA EPISODE {episode_index} INFERENCE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\n--- Loading Data ---")
    images, gt_actions, instruction, ep_id, total_episodes = extract_episode_from_tfrecord(TFRECORD_PATH, episode_index)

    if images is None or len(images) == 0:
        print("Failed to extract episode from TFRecord.")
        if episode_index >= total_episodes:
            print(f"Hint: Valid episode indices are 0 to {total_episodes-1}")
        return

    # 2. Load Model
    print("\n--- Loading Model ---")
    model_path_str = str(MODEL_PATH)
    print(f"Model path: {model_path_str}")

    try:
        processor = AutoProcessor.from_pretrained(model_path_str, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path_str,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).eval().cuda()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 3. Run Inference on All Steps
    print("\n--- Running Inference ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_actions = run_inference_on_episode(model, processor, images, instruction, device)

    # 4. Generate Plots and Visualizations
    print("\n--- Generating Plots ---")
    episode_name = f"episode_{episode_index}_id_{ep_id}"
    stats = generate_plots(images, gt_actions, pred_actions, instruction, output_dir, episode_name)
    print(f"Plots saved to: {output_dir}")

    # 5. Display Results Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    if len(gt_actions) > 0:
        print(f"\nMean Position Error: {stats['mean_position_error']:.4f}")
        print(f"Mean Rotation Error: {stats['mean_rotation_error']:.4f}")
        print(f"Mean Total Error: {stats['mean_total_error']:.4f}")

        print("\nStep-by-step comparison (Predicted vs Ground Truth):")
        print(f"{'Step':<6} {'Predicted XYZ':<30} {'GT XYZ':<30} {'Error':<10}")
        print("-"*76)

        for i in range(min(10, len(pred_actions))):  # Show first 10 steps
            pred = pred_actions[i]
            gt = gt_actions[i] if i < len(gt_actions) else None

            pred_xyz = f"[{pred[0]:.3f}, {pred[1]:.3f}, {pred[2]:.3f}]"

            if gt is not None:
                gt_xyz = f"[{gt[0]:.3f}, {gt[1]:.3f}, {gt[2]:.3f}]"
                error = np.linalg.norm(pred[:3] - gt[:3])
                print(f"{i+1:<6} {pred_xyz:<30} {gt_xyz:<30} {error:.4f}")

        if len(pred_actions) > 10:
            print(f"... ({len(pred_actions) - 10} more steps)")
    else:
        print("\nPredicted Actions (first 5 steps):")
        for i in range(min(5, len(pred_actions))):
            act = pred_actions[i]
            print(f"Step {i+1}: XYZ=[{act[0]:.3f}, {act[1]:.3f}, {act[2]:.3f}], "
                  f"RPY=[{act[3]:.3f}, {act[4]:.3f}, {act[5]:.3f}], "
                  f"Gripper={act[6]:.3f}")

    print(f"\nAll outputs saved to: {output_dir}")
    print("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="SpatialVLA Episode Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Episode 0 (first episode) - default
  python scripts/smoke_test_inference.py

  # Run Episode 5 (0-indexed, so 6th episode)
  python scripts/smoke_test_inference.py --episode_index 5

  # Run Episode 10 with custom output directory
  python scripts/smoke_test_inference.py --episode_index 10 --output_dir ./my_outputs

  # Count total episodes in the tfrecord file
  python scripts/smoke_test_inference.py --count_episodes
        """
    )
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for plots and GIFs (default: scripts/outputs/episode_{index})")
    parser.add_argument("--episode_index", type=int, default=0,
                       help="Episode index to process (0-indexed, default: 0). Each tfrecord contains ~50 episodes.")
    parser.add_argument("--count_episodes", action="store_true",
                       help="Count total episodes in tfrecord file and exit")
    args = parser.parse_args()

    main(output_dir=args.output_dir, episode_index=args.episode_index, count_episodes_only=args.count_episodes)
