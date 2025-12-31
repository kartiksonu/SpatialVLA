#!/usr/bin/env python3
"""
SpatialVLA Smoke Test: Episode inference on converted LeRobot data.

=============================================================================
PIPELINE OVERVIEW
=============================================================================

This script runs SpatialVLA inference on LeRobot SO101 data that has been
converted to RLDS format. The key challenge is that:

  - LeRobot SO101 data: 15 fps, tiny per-frame deltas (~0.003m)
  - SpatialVLA training data (Bridge): ~3 fps, larger deltas (~0.015m)

To make fair comparisons, we DOWNSAMPLE the LeRobot data and RECOMPUTE
action deltas as cumulative changes between downsampled frames.

=============================================================================
THE 5 STEPS
=============================================================================

  STEP 1: LOAD DATA
          Read episode from TFRecord (images, raw actions, states, instruction)

  STEP 2: DOWNSAMPLE & COMPUTE CUMULATIVE DELTAS
          - Take every Nth frame (e.g., N=7 for 15fps -> 2.14fps)
          - Recompute action deltas: action[i] = state[i] - state[i-1]
          - This makes deltas ~5x larger, matching Bridge scale

  STEP 3: LOAD MODEL
          Load SpatialVLA pretrained weights

  STEP 4: RUN INFERENCE
          For each frame, predict action using SpatialVLA
          Model outputs are unnormalized using Bridge statistics

  STEP 5: GENERATE VISUALIZATIONS
          - Action comparison plots (normalized space)
          - 3D trajectory visualization
          - Animated GIF
          - Statistics file

=============================================================================
USAGE
=============================================================================

  # Basic run (Episode 0, no downsampling)
  python smoke_test/smoke_test_lerobot.py

  # With downsampling to match Bridge scale (RECOMMENDED)
  python smoke_test/smoke_test_lerobot.py --episode_index 5 --skip_frames 7

  # Count episodes
  python smoke_test/smoke_test_lerobot.py --count_episodes

=============================================================================
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List
import io

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Default paths
DEFAULT_TFRECORD = SCRIPT_DIR / "data/so101_offline_eval.tfrecord"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "pretrained/spatialvla-4b-224-pt"

# Key for unnormalizing model outputs (when using Bridge stats)
# Must match the dataset key in SpatialVLA's statistics
UNNORM_KEY = "bridge_orig/1.0.0"


def compute_lerobot_norm_stats(gt_actions: np.ndarray) -> dict:
    """
    Compute normalization statistics (q01, q99) from LeRobot GT actions.

    These can be used to unnormalize model outputs using LeRobot's
    action distribution instead of Bridge's.

    Args:
        gt_actions: Ground truth actions [N, 7]

    Returns:
        Dict with 'q01', 'q99', 'mean', 'std' for each dimension
    """
    return {
        'q01': np.percentile(gt_actions, 1, axis=0),
        'q99': np.percentile(gt_actions, 99, axis=0),
        'mean': np.mean(gt_actions, axis=0),
        'std': np.std(gt_actions, axis=0),
    }


def compute_dataset_norm_stats(tfrecord_path: Path, skip_frames: int = 1) -> dict:
    """
    Compute normalization statistics from ALL episodes in a TFRecord.

    This gives more robust/stable normalization than per-episode stats.
    Applies the same downsampling logic to ensure stats match inference.

    Args:
        tfrecord_path: Path to TFRecord file
        skip_frames: Downsampling factor (must match inference)

    Returns:
        Dict with 'q01', 'q99', 'mean', 'std', 'num_episodes', 'num_samples'
    """
    print(f"\n{'='*70}")
    print("COMPUTING DATASET-WIDE NORMALIZATION STATS")
    print(f"{'='*70}")
    print(f"  TFRecord: {tfrecord_path}")
    print(f"  Skip frames: {skip_frames}")

    all_actions = []
    num_episodes = 0

    dataset = tf.data.TFRecordDataset(str(tfrecord_path))

    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        # Get number of frames from images
        num_frames = 0
        if 'steps/observation/image_0' in features:
            num_frames = len(features['steps/observation/image_0'].bytes_list.value)

        if num_frames == 0:
            continue

        # Extract states (needed for recomputing deltas after downsampling)
        states = np.zeros((num_frames, 7), dtype=np.float32)
        if 'steps/observation/state' in features:
            state_floats = features['steps/observation/state'].float_list.value
            dim = len(state_floats) // num_frames
            states = np.array(state_floats).reshape(num_frames, dim)

        # Apply downsampling
        if skip_frames > 1:
            states = states[::skip_frames]

            # Recompute deltas between downsampled states
            if len(states) > 1:
                actions = np.zeros_like(states)
                for i in range(1, len(states)):
                    # Position deltas
                    actions[i, :3] = states[i, :3] - states[i-1, :3]
                    # Rotation deltas (handle wraparound)
                    for j in range(3, 6):
                        delta = states[i, j] - states[i-1, j]
                        actions[i, j] = (delta + np.pi) % (2 * np.pi) - np.pi
                    # Gripper (absolute)
                    actions[i, 6] = states[i, 6]
                actions[0, 6] = states[0, 6]
            else:
                actions = np.zeros((1, 7), dtype=np.float32)
        else:
            # Use original actions
            actions = np.zeros((num_frames, 7), dtype=np.float32)
            if 'steps/action' in features:
                action_floats = features['steps/action'].float_list.value
                dim = len(action_floats) // num_frames
                actions = np.array(action_floats).reshape(num_frames, dim)

        all_actions.append(actions)
        num_episodes += 1

    # Concatenate all actions
    all_actions = np.concatenate(all_actions, axis=0)

    stats = {
        'q01': np.percentile(all_actions, 1, axis=0),
        'q99': np.percentile(all_actions, 99, axis=0),
        'mean': np.mean(all_actions, axis=0),
        'std': np.std(all_actions, axis=0),
        'num_episodes': num_episodes,
        'num_samples': len(all_actions),
    }

    print(f"  Episodes processed: {num_episodes}")
    print(f"  Total samples: {len(all_actions)}")
    print(f"  q01 (XYZ): {stats['q01'][:3].round(6)}")
    print(f"  q99 (XYZ): {stats['q99'][:3].round(6)}")
    print(f"  mean (XYZ): {stats['mean'][:3].round(6)}")
    print(f"  std (XYZ): {stats['std'][:3].round(6)}")

    return stats


def unnormalize_with_lerobot_stats(
    normalized_actions: np.ndarray,
    lerobot_stats: dict
) -> np.ndarray:
    """
    Unnormalize actions using LeRobot statistics instead of Bridge.

    Uses the same formula as SpatialVLA's decode_actions:
        action = 0.5 * (normalized + 1) * (q99 - q01) + q01

    Args:
        normalized_actions: Actions in [-1, 1] normalized space
        lerobot_stats: Dict with 'q01' and 'q99' from compute_lerobot_norm_stats

    Returns:
        Actions unnormalized to LeRobot's scale
    """
    q01 = lerobot_stats['q01']
    q99 = lerobot_stats['q99']
    return 0.5 * (normalized_actions + 1) * (q99 - q01) + q01


def renormalize_bridge_to_lerobot(
    bridge_actions: np.ndarray,
    bridge_stats: dict,
    lerobot_stats: dict
) -> np.ndarray:
    """
    Convert actions from Bridge scale to LeRobot scale.

    1. Re-normalize Bridge actions back to [-1, 1]
    2. Unnormalize using LeRobot stats

    Args:
        bridge_actions: Actions unnormalized with Bridge stats
        bridge_stats: Bridge q01/q99 (from processor.statistics)
        lerobot_stats: LeRobot q01/q99 (from compute_lerobot_norm_stats)

    Returns:
        Actions in LeRobot's scale
    """
    # Re-normalize to [-1, 1] using Bridge stats
    q01_b = bridge_stats['q01']
    q99_b = bridge_stats['q99']
    normalized = 2 * (bridge_actions - q01_b) / (q99_b - q01_b + 1e-8) - 1

    # Unnormalize with LeRobot stats
    return unnormalize_with_lerobot_stats(normalized, lerobot_stats)

# LeRobot original framerate
LEROBOT_FPS = 15


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EpisodeData:
    """Container for a single episode's data."""
    images: List[Image.Image]      # RGB images from camera
    actions: np.ndarray            # Action deltas [N, 7]: xyz_delta, rpy_delta, gripper
    states: np.ndarray             # Absolute states [N, 7]: xyz, rpy, gripper
    instruction: str               # Language instruction
    episode_id: str                # Episode identifier
    original_length: int           # Original number of frames (before downsampling)
    downsampled: bool              # Whether data was downsampled
    skip_frames: int               # Downsampling factor used
    effective_fps: float           # Effective framerate after downsampling


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

def decode_image(image_bytes: bytes) -> Optional[Image.Image]:
    """Decode JPEG bytes to PIL Image."""
    try:
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        print(f"  [WARNING] Failed to decode image: {e}")
        return None


def load_episode_from_tfrecord(
    tfrecord_path: Path,
    episode_index: int = 0
) -> Tuple[Optional[EpisodeData], int]:
    """
    STEP 1: Load raw episode data from TFRecord.

    This loads the data AS-IS from the TFRecord file, without any
    downsampling or action recomputation.

    Args:
        tfrecord_path: Path to TFRecord file
        episode_index: Which episode to load (0-indexed)

    Returns:
        Tuple of (EpisodeData or None, total_episode_count)

    TFRecord Structure (per episode):
        - steps/observation/image_0: JPEG bytes for each frame
        - steps/action: Flattened float array [N*7] of action deltas
        - steps/observation/state: Flattened float array [N*7] of absolute states
        - steps/language_instruction: Repeated instruction string
        - episode_metadata/episode_id: Episode identifier
    """
    print(f"\n{'='*70}")
    print("STEP 1: LOAD DATA")
    print(f"{'='*70}")
    print(f"  TFRecord: {tfrecord_path}")
    print(f"  Target episode index: {episode_index}")

    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    total_episodes = 0
    episode_data = None

    for idx, raw_record in enumerate(dataset):
        total_episodes += 1

        if idx != episode_index:
            continue

        # Parse the TFRecord example
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        # --- Extract instruction ---
        instruction = "do something"  # default
        if 'steps/language_instruction' in features:
            bytes_list = features['steps/language_instruction'].bytes_list.value
            if bytes_list:
                instruction = bytes_list[0].decode('utf-8')

        # --- Extract images ---
        images = []
        if 'steps/observation/image_0' in features:
            for img_bytes in features['steps/observation/image_0'].bytes_list.value:
                img = decode_image(img_bytes)
                if img is not None:
                    images.append(img)

        num_frames = len(images)
        if num_frames == 0:
            print(f"  [ERROR] No images found in episode {episode_index}")
            continue

        # --- Extract actions (7D: xyz_delta, rpy_delta, gripper) ---
        actions = np.zeros((num_frames, 7), dtype=np.float32)
        if 'steps/action' in features:
            action_floats = features['steps/action'].float_list.value
            dim = len(action_floats) // num_frames
            actions = np.array(action_floats).reshape(num_frames, dim)

        # --- Extract states (7D: xyz, rpy, gripper) ---
        states = np.zeros((num_frames, 7), dtype=np.float32)
        if 'steps/observation/state' in features:
            state_floats = features['steps/observation/state'].float_list.value
            dim = len(state_floats) // num_frames
            states = np.array(state_floats).reshape(num_frames, dim)

        # --- Extract episode ID ---
        episode_id = "N/A"
        if 'episode_metadata/episode_id' in features:
            feat = features['episode_metadata/episode_id']
            if feat.bytes_list.value:
                episode_id = feat.bytes_list.value[0].decode('utf-8')
            elif feat.int64_list.value:
                episode_id = str(feat.int64_list.value[0])

        episode_data = EpisodeData(
            images=images,
            actions=actions,
            states=states,
            instruction=instruction,
            episode_id=episode_id,
            original_length=num_frames,
            downsampled=False,
            skip_frames=1,
            effective_fps=LEROBOT_FPS
        )

        print(f"  Episode ID: {episode_id}")
        print(f"  Instruction: '{instruction}'")
        print(f"  Frames loaded: {num_frames}")
        print(f"  Action shape: {actions.shape}")
        print(f"  State shape: {states.shape}")
        break

    print(f"  Total episodes in file: {total_episodes}")

    if episode_data is None and episode_index >= total_episodes:
        print(f"  [ERROR] Episode index {episode_index} out of range (max: {total_episodes-1})")

    return episode_data, total_episodes


# =============================================================================
# STEP 2: DOWNSAMPLE & COMPUTE CUMULATIVE DELTAS
# =============================================================================

def downsample_episode(episode: EpisodeData, skip_frames: int) -> EpisodeData:
    """
    STEP 2: Downsample episode and recompute action deltas.

    WHY THIS IS NECESSARY:
    ----------------------
    LeRobot data is recorded at 15 fps, resulting in very small per-frame
    deltas (~0.003m per frame). SpatialVLA was trained on Bridge data at
    ~3 fps with larger deltas (~0.015m per frame).

    Simply skipping frames but keeping the original tiny deltas would give
    us the wrong action magnitudes. We need to ACCUMULATE the deltas.

    THE KEY INSIGHT:
    ----------------
    If we downsample from 15fps to 3fps (skip_frames=5), we're not just
    taking every 5th frame - we're taking every 5th STATE and computing
    the ACTION as the difference between consecutive downsampled states.

    Original 15fps:
        S0 --a0--> S1 --a1--> S2 --a2--> S3 --a3--> S4 --a4--> S5
           0.003m    0.003m    0.003m    0.003m    0.003m

    After downsampling (skip_frames=5):
        S0 -------------------- accumulated --------------------> S5
                            action = S5 - S0 = 0.015m

    This makes the action magnitude match Bridge's scale!

    Args:
        episode: Original episode data
        skip_frames: Take every Nth frame (1 = no downsampling)

    Returns:
        New EpisodeData with downsampled data and recomputed deltas
    """
    if skip_frames <= 1:
        print(f"\n{'='*70}")
        print("STEP 2: DOWNSAMPLE (SKIPPED - skip_frames=1)")
        print(f"{'='*70}")
        print("  No downsampling requested. Using original 15fps data.")
        print("  [WARNING] Action deltas will be ~5x smaller than Bridge scale!")
        return episode

    print(f"\n{'='*70}")
    print("STEP 2: DOWNSAMPLE & COMPUTE CUMULATIVE DELTAS")
    print(f"{'='*70}")
    print(f"  Original frames: {len(episode.images)}")
    print(f"  Skip factor: {skip_frames}")
    print(f"  Original fps: {LEROBOT_FPS} Hz")
    print(f"  Target fps: {LEROBOT_FPS / skip_frames:.2f} Hz")

    # --- Downsample images and states ---
    downsampled_images = episode.images[::skip_frames]
    downsampled_states = episode.states[::skip_frames]

    num_frames = len(downsampled_images)
    print(f"  Downsampled frames: {num_frames}")

    # --- Recompute action deltas between consecutive downsampled states ---
    #
    # This is the KEY step. Instead of using the original tiny deltas,
    # we compute: action[i] = state[i] - state[i-1]
    #
    # For position (XYZ): simple subtraction
    # For rotation (RPY): subtraction with angle wrapping to [-pi, pi]
    # For gripper: use current value (not a delta)

    new_actions = np.zeros((num_frames, 7), dtype=np.float32)

    for i in range(1, num_frames):
        # Position delta: xyz[i] - xyz[i-1]
        # This accumulates all the tiny deltas between frame i-1 and frame i
        new_actions[i, 0] = downsampled_states[i, 0] - downsampled_states[i-1, 0]  # X
        new_actions[i, 1] = downsampled_states[i, 1] - downsampled_states[i-1, 1]  # Y
        new_actions[i, 2] = downsampled_states[i, 2] - downsampled_states[i-1, 2]  # Z

        # Rotation delta with angle wrapping
        # Angles can wrap around (e.g., 179° to -179° is only 2° change, not 358°)
        for j in range(3, 6):  # Roll, Pitch, Yaw
            delta = downsampled_states[i, j] - downsampled_states[i-1, j]
            # Wrap delta to [-pi, pi]
            new_actions[i, j] = (delta + np.pi) % (2 * np.pi) - np.pi

        # Gripper: use absolute value (0=closed, 1=open), not a delta
        new_actions[i, 6] = downsampled_states[i, 6]

    # First frame has no previous frame, so deltas are 0
    # But we still want the gripper state
    new_actions[0, 6] = downsampled_states[0, 6]

    # --- Compute and display statistics ---
    xyz_magnitudes = np.linalg.norm(new_actions[1:, :3], axis=1)  # Skip frame 0 (zeros)
    mean_xyz_mag = np.mean(xyz_magnitudes) if len(xyz_magnitudes) > 0 else 0

    BRIDGE_XYZ_MAG = 0.0158  # Bridge dataset average XYZ delta magnitude
    ratio = mean_xyz_mag / BRIDGE_XYZ_MAG if BRIDGE_XYZ_MAG > 0 else 0

    print(f"\n  Action delta statistics (after recomputation):")
    print(f"    Mean XYZ magnitude: {mean_xyz_mag:.5f} m")
    print(f"    Bridge XYZ magnitude: {BRIDGE_XYZ_MAG:.5f} m")
    print(f"    Ratio to Bridge: {ratio:.0%}")

    if ratio < 0.7:
        print(f"    [WARNING] Deltas still smaller than Bridge. Consider larger skip_frames.")
    elif ratio > 1.5:
        print(f"    [WARNING] Deltas larger than Bridge. Consider smaller skip_frames.")
    else:
        print(f"    [OK] Deltas are in similar range to Bridge data.")

    return EpisodeData(
        images=downsampled_images,
        actions=new_actions,
        states=downsampled_states,
        instruction=episode.instruction,
        episode_id=episode.episode_id,
        original_length=episode.original_length,
        downsampled=True,
        skip_frames=skip_frames,
        effective_fps=LEROBOT_FPS / skip_frames
    )


# =============================================================================
# STEP 3: LOAD MODEL
# =============================================================================

def load_spatialvla_model(model_path: Path) -> Tuple[AutoModel, AutoProcessor, torch.device]:
    """
    STEP 3: Load SpatialVLA model and processor.

    Loads the pretrained SpatialVLA-4B model for inference.
    The model uses:
      - Adaptive Action Grids for spatial tokenization
      - Ego3D Position Encoding for 3D understanding
      - Bridge dataset statistics for action normalization

    Args:
        model_path: Path to pretrained model directory

    Returns:
        Tuple of (model, processor, device)
    """
    print(f"\n{'='*70}")
    print("STEP 3: LOAD MODEL")
    print(f"{'='*70}")
    print(f"  Model path: {model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    print("  Processor loaded.")

    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).eval().to(device)
    print("  Model loaded.")

    return model, processor, device


# =============================================================================
# STEP 4: RUN INFERENCE
# =============================================================================

def enable_sampling_mode(model, temperature: float = 0.7):
    """
    Monkey-patch the model to use sampling instead of greedy decoding.

    By default, SpatialVLA uses do_sample=False (greedy decoding), which
    means the same input always produces the same output. This makes
    multi-query mode useless since all queries return identical results.

    This function patches predict_action to use do_sample=True with
    temperature, enabling stochastic sampling from the action distribution.

    Args:
        model: SpatialVLA model
        temperature: Sampling temperature (0.5-1.0 typical)
                    Lower = more deterministic, Higher = more random
    """
    original_predict = model.predict_action

    def predict_action_with_sampling(model_inputs):
        model_inputs = model_inputs.to(torch.bfloat16).to(model.device)
        input_len = model_inputs["input_ids"].shape[-1]
        generation_outputs = model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=temperature,
        )
        return generation_outputs[:, input_len:]

    # Replace the method
    import types
    model.predict_action = types.MethodType(
        lambda self, inputs: predict_action_with_sampling(inputs),
        model
    )

    return original_predict  # Return original in case we want to restore


def run_inference(
    model: AutoModel,
    processor: AutoProcessor,
    episode: EpisodeData,
    device: torch.device,
    num_queries: int = 1,
    temperature: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    STEP 4: Run SpatialVLA inference on episode.

    For each frame in the episode:
      1. Prepare input (image + text prompt)
      2. Run model forward pass N times (num_queries) to measure variance
      3. Decode predicted action tokens - get FULL action chunk (size 4)
      4. Unnormalize using Bridge statistics

    The model outputs actions in NORMALIZED space (trained on Bridge).
    We unnormalize them using Bridge statistics to get real-world deltas.

    MULTI-QUERY MODE:
    -----------------
    When num_queries > 1, we run inference multiple times per frame.
    This captures the stochastic nature of the model's predictions.

    IMPORTANT: By default the model uses greedy decoding (do_sample=False),
    which means all queries return identical results! Set temperature > 0
    to enable sampling mode for actual variance.

    For example, with:
      - 20 frames
      - 10 queries per frame
      - Action chunk size of 4

    We get: 20 * 10 * 4 = 800 total action predictions!

    Args:
        model: SpatialVLA model
        processor: SpatialVLA processor (handles tokenization)
        episode: Episode data with images and instruction
        device: Torch device
        num_queries: Number of times to query model per frame (default: 1)
        temperature: Sampling temperature (0=greedy, 0.5-1.0=stochastic)

    Returns:
        Tuple of:
          - pred_actions [N, 7]: Mean of first action from each query
          - all_chunks [N, num_queries, chunk_size, 7]: Full chunks from ALL queries
    """
    print(f"\n{'='*70}")
    print("STEP 4: RUN INFERENCE")
    print(f"{'='*70}")
    print(f"  Instruction: '{episode.instruction}'")
    print(f"  Frames to process: {len(episode.images)}")
    print(f"  Queries per frame: {num_queries}")

    # Enable sampling mode if temperature > 0
    if temperature > 0 and num_queries > 1:
        print(f"  Temperature: {temperature} (SAMPLING MODE)")
        enable_sampling_mode(model, temperature)
    else:
        if num_queries > 1:
            print(f"  Temperature: 0 (GREEDY - all queries will be identical!)")
        else:
            print(f"  Temperature: 0 (greedy)")

    print(f"  Unnormalization key: {UNNORM_KEY}")

    # Get chunk size from processor
    chunk_size = getattr(processor, 'action_chunk_size', 4)
    print(f"  Action chunk size: {chunk_size}")
    print(f"  Total predictions: {len(episode.images)} * {num_queries} * {chunk_size} = "
          f"{len(episode.images) * num_queries * chunk_size}")

    # Build prompt for the model
    prompt = f"In: What action should the robot take to {episode.instruction.lower()}?\nOut:"

    pred_actions = []      # Mean of first actions across queries [N, 7]
    all_chunks = []        # ALL chunks from ALL queries [N, num_queries, chunk_size, 7]

    for i, image in enumerate(episode.images):
        # Prepare inputs (once per frame)
        inputs = processor(
            images=[image.convert("RGB")],
            text=prompt,
            return_tensors="pt"
        ).to(device, torch.bfloat16)

        # Query model multiple times to capture variance
        frame_chunks = []      # [num_queries, chunk_size, 7]
        frame_first_actions = []  # [num_queries, 7]

        for q in range(num_queries):
            # Run inference
            # NOTE: torch.manual_seed() is intentionally NOT called
            # This allows the model to produce varied outputs across queries
            with torch.no_grad():
                generation_outputs = model.predict_action(inputs)

            # Decode action tokens to continuous values
            # actions shape: [chunk_size, 7]
            actions_dict = processor.decode_actions(generation_outputs, unnorm_key=UNNORM_KEY)
            actions = actions_dict["actions"]  # [chunk_size, 7]

            frame_chunks.append(actions)
            frame_first_actions.append(actions[0])

        frame_chunks = np.array(frame_chunks)            # [num_queries, chunk_size, 7]
        frame_first_actions = np.array(frame_first_actions)  # [num_queries, 7]

        # Use mean of first actions as the prediction for this frame
        pred_actions.append(np.mean(frame_first_actions, axis=0))
        all_chunks.append(frame_chunks)

        # Progress indicator
        if (i + 1) % 5 == 0 or i == len(episode.images) - 1:
            print(f"  Processed {i+1}/{len(episode.images)} frames...")

    pred_actions = np.array(pred_actions)  # [N, 7]
    all_chunks = np.array(all_chunks)      # [N, num_queries, chunk_size, 7]

    print(f"  Predicted actions shape: {pred_actions.shape}")
    print(f"  All chunks shape: {all_chunks.shape}")

    # Display sample predictions with variance
    print(f"\n  Sample predictions (first 3 frames):")
    for i in range(min(3, len(pred_actions))):
        xyz_mean = pred_actions[i, :3]
        if num_queries > 1:
            xyz_std = np.std(all_chunks[i, :, 0, :3], axis=0)
            print(f"    Frame {i}: XYZ=[{xyz_mean[0]:+.4f}±{xyz_std[0]:.4f}, "
                  f"{xyz_mean[1]:+.4f}±{xyz_std[1]:.4f}, {xyz_mean[2]:+.4f}±{xyz_std[2]:.4f}]")
        else:
            grip = "open" if pred_actions[i, 6] > 0.5 else "closed"
            print(f"    Frame {i}: XYZ=[{xyz_mean[0]:+.4f}, {xyz_mean[1]:+.4f}, {xyz_mean[2]:+.4f}] gripper={grip}")

    return pred_actions, all_chunks


# =============================================================================
# STEP 5: GENERATE VISUALIZATIONS
# =============================================================================

def compute_action_stats(actions: np.ndarray) -> tuple:
    """
    Compute mean and std from action data.

    Used to normalize both GT and predicted actions to the same space.
    By computing stats from the downsampled GT, we get a data-driven
    normalization that doesn't rely on hardcoded values.

    Args:
        actions: Action array [N, 7]

    Returns:
        Tuple of (mean, std) arrays
    """
    mean = np.mean(actions, axis=0)
    std = np.std(actions, axis=0)
    # Prevent division by zero for dimensions with no variance
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def normalize_actions(actions: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize actions to zero mean, unit variance using provided stats."""
    return (actions - mean) / std


def normalize_trajectory_to_unit_scale(trajectory: np.ndarray) -> np.ndarray:
    """Normalize trajectory to unit scale (center at origin, unit variance)."""
    centered = trajectory - trajectory.mean(axis=0)
    scale = np.std(centered)
    if scale > 1e-6:
        centered = centered / scale
    return centered


def generate_visualizations(
    episode: EpisodeData,
    pred_actions: np.ndarray,
    all_chunks: np.ndarray,
    output_dir: Path
) -> dict:
    """
    STEP 5: Generate comparison plots and animations.

    Generates:
      1. action_comparison.png - Per-dimension action comparison (normalized)
      2. trajectory_3d.png - 3D trajectory visualization
      3. gripper_comparison.png - Gripper state over time
      4. action_chunks.png - Full action chunks from all queries (NEW)
      5. trajectory.gif - Animated visualization
      6. statistics.txt - Numerical metrics

    NORMALIZATION STRATEGY:
    -----------------------
    We compute mean/std from the DOWNSAMPLED GT actions and use those
    to normalize both GT and predictions. This works because:

      1. With proper downsampling (skip_frames=7), GT deltas ≈ Bridge scale
      2. Model predictions are in Bridge scale (after unnormalization)
      3. Both are now in similar magnitude, so same stats work for both

    This approach:
      - Removes hardcoded magic numbers
      - Is self-documenting (stats come from actual data)
      - Works correctly as long as downsampling matches Bridge fps

    Args:
        episode: Episode data with GT actions/states
        pred_actions: Model predictions [N, 7]
        all_chunks: Full action chunks [N, num_queries, chunk_size, 7]
        output_dir: Directory to save outputs

    Returns:
        Dictionary of computed statistics
    """
    print(f"\n{'='*70}")
    print("STEP 5: GENERATE VISUALIZATIONS")
    print(f"{'='*70}")
    print(f"  Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    gt_actions = episode.actions
    gt_states = episode.states
    images = episode.images
    instruction = episode.instruction
    num_steps = len(images)

    episode_name = f"episode_{episode.episode_id}"
    if episode.downsampled:
        episode_name += f"_skip{episode.skip_frames}"

    # --- Compute normalization stats from downsampled GT ---
    # This is the key: we use GT stats to normalize BOTH GT and predictions.
    # With proper downsampling, GT and predictions are in similar scale,
    # so the same normalization works for fair comparison.
    gt_mean, gt_std = compute_action_stats(gt_actions)

    print(f"\n  Normalization stats (computed from GT):")
    print(f"    Mean XYZ: [{gt_mean[0]:.6f}, {gt_mean[1]:.6f}, {gt_mean[2]:.6f}]")
    print(f"    Std XYZ:  [{gt_std[0]:.6f}, {gt_std[1]:.6f}, {gt_std[2]:.6f}]")

    if not episode.downsampled:
        print(f"    [WARNING] No downsampling applied! Stats may not match Bridge scale.")
        print(f"              Consider using --skip_frames 7 for proper comparison.")

    # Normalize both GT and predictions using GT stats
    gt_actions_norm = normalize_actions(gt_actions, gt_mean, gt_std)
    pred_actions_norm = normalize_actions(pred_actions, gt_mean, gt_std)

    # --- Compute positions (for trajectory visualization) ---
    # GT: Use absolute positions from FK-computed states
    gt_positions = gt_states[:, :3]

    # Predicted: Integrate predictions starting from GT start position
    pred_positions = np.cumsum(pred_actions[:, :3], axis=0)
    pred_positions = pred_positions - pred_positions[0] + gt_positions[0]

    # --- 1. ACTION COMPARISON PLOT (Normalized Space) ---
    print("  Generating action_comparison.png...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    title = f"Action Comparison: '{instruction}'\n"
    title += f"Episode: {episode_name} | {num_steps} steps @ {episode.effective_fps:.1f} Hz\n"
    title += "(Normalized using GT statistics)"
    fig.suptitle(title, fontsize=11)

    for i, (label, color) in enumerate(zip(['X', 'Y', 'Z'], ['r', 'g', 'b'])):
        axes[0, i].plot(gt_actions_norm[:, i], f'{color}-', label='Ground Truth', linewidth=2)
        axes[0, i].plot(pred_actions_norm[:, i], f'{color}--', label='Predicted', linewidth=2)
        axes[0, i].set_xlabel('Step')
        axes[0, i].set_ylabel(f'{label} (normalized)')
        axes[0, i].set_title(f'Position {label}')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)

    for i, (label, color) in enumerate(zip(['Roll', 'Pitch', 'Yaw'], ['r', 'g', 'b'])):
        axes[1, i].plot(gt_actions_norm[:, i+3], f'{color}-', label='Ground Truth', linewidth=2)
        axes[1, i].plot(pred_actions_norm[:, i+3], f'{color}--', label='Predicted', linewidth=2)
        axes[1, i].set_xlabel('Step')
        axes[1, i].set_ylabel(f'{label} (normalized)')
        axes[1, i].set_title(f'Rotation {label}')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'action_comparison.png', dpi=150)
    plt.close()

    # --- 2. 3D TRAJECTORY PLOT ---
    print("  Generating trajectory_3d.png...")

    # Normalize trajectories to unit scale for shape comparison
    gt_traj_norm = normalize_trajectory_to_unit_scale(gt_positions)
    pred_traj_norm = normalize_trajectory_to_unit_scale(pred_positions)

    fig = plt.figure(figsize=(14, 6))

    # Left panel: Normalized comparison (shape)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(gt_traj_norm[:, 0], gt_traj_norm[:, 1], gt_traj_norm[:, 2],
             'b-', linewidth=2, label='Ground Truth')
    ax1.plot(pred_traj_norm[:, 0], pred_traj_norm[:, 1], pred_traj_norm[:, 2],
             'r--', linewidth=2, label='Predicted')
    ax1.scatter(*gt_traj_norm[0], c='green', s=150, marker='*', label='Start')
    ax1.scatter(*gt_traj_norm[-1], c='blue', s=100, marker='o', label='GT End')
    ax1.scatter(*pred_traj_norm[-1], c='red', s=100, marker='^', label='Pred End')
    ax1.set_xlabel('X (normalized)')
    ax1.set_ylabel('Y (normalized)')
    ax1.set_zlabel('Z (normalized)')
    ax1.set_title('Trajectory SHAPE Comparison\n(both normalized to unit scale)')
    ax1.legend(fontsize=8)

    # Right panel: GT actual trajectory (meters)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2],
             'b-', linewidth=2, label='Ground Truth (FK)')
    ax2.scatter(*gt_positions[0], c='green', s=150, marker='*', label='Start')
    ax2.scatter(*gt_positions[-1], c='blue', s=100, marker='o', label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title(f'GT Trajectory (actual meters)\n"{instruction}"')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_3d.png', dpi=150)
    plt.close()

    # --- 3. GRIPPER COMPARISON ---
    print("  Generating gripper_comparison.png...")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gt_actions[:, 6], 'b-', label='Ground Truth', linewidth=2, marker='o', markersize=3)
    ax.plot(pred_actions[:, 6], 'r--', label='Predicted', linewidth=2, marker='^', markersize=3)
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

    # --- 4. ACTION CHUNKS PLOT (Full chunks from all queries) ---
    # all_chunks shape: [N, num_queries, chunk_size, 7]
    num_queries = all_chunks.shape[1]
    chunk_size = all_chunks.shape[2]

    if num_queries > 1:
        print(f"  Generating action_chunks.png ({num_queries} queries x {chunk_size} chunk size)...")

        # Dynamic alpha: more queries = lower alpha for density visualization
        # This creates a "probability distribution" effect where:
        #   - Dense regions (many overlapping samples) appear darker
        #   - Sparse regions appear lighter
        # Formula: alpha ≈ 3 / num_queries, clamped to [0.02, 0.3]
        plot_alpha = max(0.02, min(0.3, 3.0 / num_queries))
        print(f"    Using alpha={plot_alpha:.3f} for density visualization")

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f"Action Chunks: {num_queries} queries × {chunk_size} chunk size\n"
                     f"Total predictions per frame: {num_queries * chunk_size} | "
                     f"alpha={plot_alpha:.2f}\n"
                     f"'{instruction}'", fontsize=11)

        dim_labels = ['X', 'Y', 'Z']
        colors = ['red', 'green', 'blue']

        for dim_idx, (label, color) in enumerate(zip(dim_labels, colors)):
            ax_chunks = axes[dim_idx, 0]
            ax_variance = axes[dim_idx, 1]

            # LEFT PLOT: All chunk predictions at each timestep
            # Each query produces a chunk of `chunk_size` actions
            # We plot them all, offset by their position in the chunk
            # With many queries (50-100), this creates a probability distribution visualization

            for frame_idx in range(num_steps):
                for query_idx in range(num_queries):
                    # Get chunk for this frame and query: [chunk_size, 7]
                    chunk = all_chunks[frame_idx, query_idx, :, dim_idx]

                    # X-axis: frame_idx + small offset for each action in chunk
                    # This shows how the chunk "looks ahead" from each frame
                    x_vals = frame_idx + np.arange(chunk_size) * 0.2

                    # Plot with dynamic alpha - many overlapping lines create density
                    ax_chunks.plot(x_vals, chunk, color=color, alpha=plot_alpha, linewidth=0.8)

            # Overlay GT actions (thick black line)
            ax_chunks.plot(range(num_steps), gt_actions[:, dim_idx], 'k-',
                          label='GT', linewidth=2.5, marker='o', markersize=4)

            # Overlay mean prediction (first action only, dashed)
            ax_chunks.plot(range(num_steps), pred_actions[:, dim_idx],
                          color=color, linestyle='--', label='Pred (mean)', linewidth=2)

            ax_chunks.set_xlabel('Frame')
            ax_chunks.set_ylabel(f'{label} delta (m)')
            ax_chunks.set_title(f'{label}: All {num_queries * chunk_size} predictions per frame')
            ax_chunks.legend(fontsize=8)
            ax_chunks.grid(True, alpha=0.3)

            # RIGHT PLOT: Variance across queries for each chunk position
            # Shows how much the model's predictions vary
            # all_chunks[:, :, :, dim_idx] -> [N, num_queries, chunk_size]
            variance_per_chunk_pos = np.std(all_chunks[:, :, :, dim_idx], axis=1)  # [N, chunk_size]

            for chunk_pos in range(chunk_size):
                ax_variance.plot(range(num_steps), variance_per_chunk_pos[:, chunk_pos],
                               label=f't+{chunk_pos}', linewidth=1.5, alpha=0.8,
                               marker='o', markersize=2)

            ax_variance.set_xlabel('Frame')
            ax_variance.set_ylabel(f'{label} std across {num_queries} queries')
            ax_variance.set_title(f'{label}: Variance by chunk position (t+0 to t+{chunk_size-1})')
            ax_variance.legend(fontsize=8)
            ax_variance.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'action_chunks.png', dpi=150)
        plt.close()
    else:
        print("  Skipping action_chunks.png (only 1 query, no variance to show)")

    # --- 5. COMPUTE ALL TRAJECTORY SAMPLES ---
    # For multi-query mode, compute trajectory for EACH query
    # all_chunks shape: [N, num_queries, chunk_size, 7]
    # We integrate the first action (t+0) from each query to get trajectory samples

    num_queries = all_chunks.shape[1]
    all_trajectories_norm = []  # Will hold [num_queries] normalized trajectories

    if num_queries > 1:
        print(f"  Computing {num_queries} trajectory samples...")

        for q in range(num_queries):
            # Get first action (t+0) from each frame for this query
            # all_chunks[:, q, 0, :3] -> [N, 3] XYZ deltas for query q
            query_actions = all_chunks[:, q, 0, :3]  # [N, 3]

            # Integrate to get positions (starting from GT start)
            query_positions = np.cumsum(query_actions, axis=0)
            query_positions = query_positions - query_positions[0] + gt_positions[0]

            # Normalize to unit scale (same as gt_traj_norm and pred_traj_norm)
            query_traj_norm = normalize_trajectory_to_unit_scale(query_positions)
            all_trajectories_norm.append(query_traj_norm)

        all_trajectories_norm = np.array(all_trajectories_norm)  # [num_queries, N, 3]

        # Dynamic alpha and linewidth for trajectory cloud
        # Goal: overlapping lines create visible density patterns
        #
        # The key insight: with N lines at alpha A, full overlap gives ~N*A opacity
        # We want ~2-4 overlapping lines to reach ~50-80% opacity for visibility
        #
        # num_queries=10:  alpha=0.5  -> 2 overlaps = 100% (bold)
        # num_queries=30:  alpha=0.25 -> 3 overlaps = 75%
        # num_queries=50:  alpha=0.15 -> 4 overlaps = 60%
        # num_queries=100: alpha=0.10 -> 5 overlaps = 50%

        if num_queries <= 10:
            traj_alpha = 0.5
            traj_linewidth = 1.5
        elif num_queries <= 30:
            traj_alpha = 0.25
            traj_linewidth = 1.2
        elif num_queries <= 50:
            traj_alpha = 0.15
            traj_linewidth = 1.0
        elif num_queries <= 100:
            traj_alpha = 0.10
            traj_linewidth = 0.8
        else:
            traj_alpha = max(0.05, 5.0 / num_queries)
            traj_linewidth = 0.6

        print(f"    Trajectory cloud: alpha={traj_alpha:.3f}, linewidth={traj_linewidth}")

    # --- 6. TRAJECTORY CLOUD PLOT (static) ---
    if num_queries > 1:
        print("  Generating trajectory_cloud.png...")

        fig = plt.figure(figsize=(14, 6))

        # Left: All trajectory samples
        ax1 = fig.add_subplot(121, projection='3d')

        # Plot all sampled trajectories as semi-transparent lines
        for q in range(num_queries):
            ax1.plot(all_trajectories_norm[q, :, 0],
                    all_trajectories_norm[q, :, 1],
                    all_trajectories_norm[q, :, 2],
                    'r-', alpha=traj_alpha, linewidth=traj_linewidth)

        # Overlay GT trajectory (thick blue)
        ax1.plot(gt_traj_norm[:, 0], gt_traj_norm[:, 1], gt_traj_norm[:, 2],
                'b-', linewidth=3, label='Ground Truth')

        # Overlay mean prediction (thick red dashed)
        ax1.plot(pred_traj_norm[:, 0], pred_traj_norm[:, 1], pred_traj_norm[:, 2],
                'r--', linewidth=2, label='Pred (mean)')

        ax1.scatter(*gt_traj_norm[0], c='green', s=200, marker='*', label='Start', zorder=10)
        ax1.scatter(*gt_traj_norm[-1], c='blue', s=100, marker='o', label='GT End', zorder=10)

        ax1.set_xlabel('X (norm)')
        ax1.set_ylabel('Y (norm)')
        ax1.set_zlabel('Z (norm)')
        ax1.set_title(f'Trajectory Cloud ({num_queries} samples)\n"{instruction}"')
        ax1.legend(fontsize=8)

        # Right: Top-down view (X-Y plane)
        ax2 = fig.add_subplot(122)

        for q in range(num_queries):
            ax2.plot(all_trajectories_norm[q, :, 0],
                    all_trajectories_norm[q, :, 1],
                    'r-', alpha=traj_alpha, linewidth=traj_linewidth)

        ax2.plot(gt_traj_norm[:, 0], gt_traj_norm[:, 1],
                'b-', linewidth=3, label='Ground Truth')
        ax2.plot(pred_traj_norm[:, 0], pred_traj_norm[:, 1],
                'r--', linewidth=2, label='Pred (mean)')

        ax2.scatter(gt_traj_norm[0, 0], gt_traj_norm[0, 1], c='green', s=200, marker='*', label='Start', zorder=10)
        ax2.scatter(gt_traj_norm[-1, 0], gt_traj_norm[-1, 1], c='blue', s=100, marker='o', label='GT End', zorder=10)

        ax2.set_xlabel('X (norm)')
        ax2.set_ylabel('Y (norm)')
        ax2.set_title(f'Top-Down View (X-Y)\n{num_queries} trajectory samples')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(output_dir / 'trajectory_cloud.png', dpi=150)
        plt.close()

    # --- 7. ANIMATED GIF (with trajectory cloud if multi-query) ---
    print("  Generating trajectory.gif...")
    fig = plt.figure(figsize=(14, 6))
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_img = fig.add_subplot(122)

    # Compute bounds including all trajectory samples
    if num_queries > 1:
        all_pos = np.vstack([gt_traj_norm, pred_traj_norm] +
                           [all_trajectories_norm[q] for q in range(num_queries)])
    else:
        all_pos = np.vstack([gt_traj_norm, pred_traj_norm])

    margin = 0.2
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
            # Plot trajectory cloud (all sampled trajectories)
            if num_queries > 1:
                for q in range(num_queries):
                    ax_3d.plot(all_trajectories_norm[q, :frame+1, 0],
                              all_trajectories_norm[q, :frame+1, 1],
                              all_trajectories_norm[q, :frame+1, 2],
                              'r-', alpha=traj_alpha, linewidth=traj_linewidth)

            # GT trajectory (thick blue)
            ax_3d.plot(gt_traj_norm[:frame+1, 0], gt_traj_norm[:frame+1, 1], gt_traj_norm[:frame+1, 2],
                       'b-', linewidth=2.5, label='Ground Truth')

            # Mean prediction (red dashed)
            ax_3d.plot(pred_traj_norm[:frame+1, 0], pred_traj_norm[:frame+1, 1], pred_traj_norm[:frame+1, 2],
                       'r--', linewidth=2, label='Pred (mean)')

        ax_3d.scatter(*gt_traj_norm[frame], c='blue', s=100, marker='o', edgecolors='black', zorder=5)
        ax_3d.scatter(*pred_traj_norm[frame], c='red', s=100, marker='^', edgecolors='black', zorder=5)
        ax_3d.scatter(*gt_traj_norm[0], c='green', s=150, marker='*', edgecolors='black', label='Start')

        ax_3d.set_xlabel('X (norm)')
        ax_3d.set_ylabel('Y (norm)')
        ax_3d.set_zlabel('Z (norm)')

        if num_queries > 1:
            ax_3d.set_title(f'Trajectory Cloud ({num_queries} samples)\nStep {frame+1}/{num_steps}')
        else:
            ax_3d.set_title(f'Trajectory\nStep {frame+1}/{num_steps}')

        ax_3d.legend(loc='upper left', fontsize=8)
        ax_3d.view_init(elev=20, azim=45 + frame * 2)

        ax_img.imshow(images[frame])
        ax_img.set_title(f"Camera View - Step {frame+1}\n'{instruction}'")
        ax_img.axis('off')

        gt_grip = "open" if gt_actions[frame, 6] > 0.5 else "close"
        pred_grip = "open" if pred_actions[frame, 6] > 0.5 else "close"
        info_text = f"GT: [{gt_actions_norm[frame, 0]:+.2f}, {gt_actions_norm[frame, 1]:+.2f}, {gt_actions_norm[frame, 2]:+.2f}] {gt_grip}\n"
        info_text += f"Pred: [{pred_actions_norm[frame, 0]:+.2f}, {pred_actions_norm[frame, 1]:+.2f}, {pred_actions_norm[frame, 2]:+.2f}] {pred_grip}"
        ax_img.text(0.5, -0.1, info_text, transform=ax_img.transAxes, fontsize=9,
                    ha='center', va='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        return []

    anim = animation.FuncAnimation(fig, update, frames=num_steps, interval=500, blit=False)
    anim.save(output_dir / 'trajectory.gif', writer='pillow', fps=2)
    plt.close()

    # --- 5. STATISTICS ---
    print("  Computing statistics...")

    # Errors in normalized space
    position_errors = [np.linalg.norm(pred_actions_norm[i, :3] - gt_actions_norm[i, :3])
                       for i in range(num_steps)]
    rotation_errors = [np.linalg.norm(pred_actions_norm[i, 3:6] - gt_actions_norm[i, 3:6])
                       for i in range(num_steps)]
    total_errors = [np.linalg.norm(pred_actions_norm[i, :6] - gt_actions_norm[i, :6])
                    for i in range(num_steps)]

    # Cosine similarity (direction agreement)
    cosine_sims = []
    for i in range(num_steps):
        gt_vec = gt_actions_norm[i, :3]
        pred_vec = pred_actions_norm[i, :3]
        gt_norm = np.linalg.norm(gt_vec)
        pred_norm = np.linalg.norm(pred_vec)
        if gt_norm > 0.01 and pred_norm > 0.01:
            cosine_sims.append(np.dot(gt_vec, pred_vec) / (gt_norm * pred_norm))

    stats = {
        'episode_name': episode_name,
        'instruction': instruction,
        'num_steps': num_steps,
        'effective_fps': episode.effective_fps,
        'skip_frames': episode.skip_frames,
        'gt_mean': gt_mean,
        'gt_std': gt_std,
        'mean_position_error': np.mean(position_errors),
        'mean_rotation_error': np.mean(rotation_errors),
        'mean_total_error': np.mean(total_errors),
        'mean_cosine_similarity': np.mean(cosine_sims) if cosine_sims else 0.0,
        'position_errors': position_errors,
        'rotation_errors': rotation_errors,
        'total_errors': total_errors,
    }

    # Write statistics file
    with open(output_dir / 'statistics.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SPATIALVLA INFERENCE RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Episode: {episode_name}\n")
        f.write(f"Instruction: {instruction}\n")
        f.write(f"Steps: {num_steps}\n")
        f.write(f"Effective FPS: {episode.effective_fps:.2f} Hz\n")
        f.write(f"Downsampling: skip_frames={episode.skip_frames}\n\n")

        f.write("-" * 70 + "\n")
        f.write("NORMALIZATION STATS (computed from downsampled GT)\n")
        f.write("-" * 70 + "\n")
        f.write(f"GT Mean: [{gt_mean[0]:.6f}, {gt_mean[1]:.6f}, {gt_mean[2]:.6f}, "
                f"{gt_mean[3]:.6f}, {gt_mean[4]:.6f}, {gt_mean[5]:.6f}, {gt_mean[6]:.6f}]\n")
        f.write(f"GT Std:  [{gt_std[0]:.6f}, {gt_std[1]:.6f}, {gt_std[2]:.6f}, "
                f"{gt_std[3]:.6f}, {gt_std[4]:.6f}, {gt_std[5]:.6f}, {gt_std[6]:.6f}]\n\n")

        f.write("-" * 70 + "\n")
        f.write("ERRORS (normalized using GT stats)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean Position Error: {stats['mean_position_error']:.4f}\n")
        f.write(f"Mean Rotation Error: {stats['mean_rotation_error']:.4f}\n")
        f.write(f"Mean Total Error: {stats['mean_total_error']:.4f}\n")
        f.write(f"Mean Cosine Similarity: {stats['mean_cosine_similarity']:.4f}\n")
        f.write("  (1.0 = perfect direction match, 0 = orthogonal, -1 = opposite)\n\n")

        f.write("-" * 70 + "\n")
        f.write("NOTE ON INTERPRETATION\n")
        f.write("-" * 70 + "\n")
        f.write("This is ZERO-SHOT inference on OUT-OF-DISTRIBUTION data.\n")
        f.write("SpatialVLA was trained on Bridge data, not LeRobot/SO101.\n")
        f.write("Normalization uses GT stats (data-driven, no hardcoded values).\n")
        f.write("With proper downsampling (skip_frames~7), GT ≈ Bridge scale.\n\n")

        f.write("-" * 70 + "\n")
        f.write("PER-STEP ERRORS\n")
        f.write("-" * 70 + "\n")
        for i in range(num_steps):
            f.write(f"Step {i+1:3d}: pos={position_errors[i]:.4f}, "
                    f"rot={rotation_errors[i]:.4f}, total={total_errors[i]:.4f}\n")

    print(f"  Statistics saved to statistics.txt")

    # --- 9. SAVE DATA FOR LATER PLOTTING ---
    # Save all arrays to .npz so plots can be regenerated without re-running inference
    print("  Saving data to inference_data.npz...")

    save_dict = {
        # Episode metadata
        'episode_name': episode_name,
        'instruction': instruction,
        'num_steps': num_steps,
        'effective_fps': episode.effective_fps,
        'skip_frames': episode.skip_frames,
        'num_queries': num_queries,

        # Ground truth
        'gt_actions': gt_actions,              # [N, 7] raw GT actions
        'gt_states': gt_states,                # [N, 7] raw GT states
        'gt_positions': gt_positions,          # [N, 3] XYZ positions from FK

        # Predictions
        'pred_actions': pred_actions,          # [N, 7] mean predicted actions
        'all_chunks': all_chunks,              # [N, num_queries, chunk_size, 7] all predictions

        # Normalized versions (for plotting)
        'gt_actions_norm': gt_actions_norm,    # [N, 7] normalized GT actions
        'pred_actions_norm': pred_actions_norm,# [N, 7] normalized pred actions
        'gt_mean': gt_mean,                    # [7] normalization mean
        'gt_std': gt_std,                      # [7] normalization std

        # Trajectories (normalized to unit scale)
        'gt_traj_norm': gt_traj_norm,          # [N, 3] normalized GT trajectory
        'pred_traj_norm': pred_traj_norm,      # [N, 3] normalized pred trajectory
        'pred_positions': pred_positions,      # [N, 3] raw integrated positions
    }

    # Add all trajectory samples if multi-query
    if num_queries > 1 and len(all_trajectories_norm) > 0:
        save_dict['all_trajectories_norm'] = all_trajectories_norm  # [num_queries, N, 3]

    np.savez(output_dir / 'inference_data.npz', **save_dict)
    print(f"  Data saved to inference_data.npz")
    print(f"  All outputs saved to: {output_dir}")

    return stats


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def count_episodes(tfrecord_path: Path) -> int:
    """Count total episodes in a TFRecord file."""
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    return sum(1 for _ in dataset)


def main(
    output_dir: Optional[str] = None,
    episode_index: int = 0,
    count_episodes_only: bool = False,
    skip_frames: int = 1,
    num_queries: int = 1,
    temperature: float = 0.0,
    use_lerobot_unnorm: bool = False,
    tfrecord_path: Optional[str] = None,
    model_path: Optional[str] = None
):
    """
    Main entry point for SpatialVLA inference on LeRobot data.

    Executes the 5-step pipeline:
      1. Load data from TFRecord
      2. Downsample and compute cumulative deltas
      3. Load SpatialVLA model
      4. Run inference (optionally with multiple queries per frame)
      5. (Optional) Re-unnormalize with LeRobot stats
      6. Generate visualizations

    Args:
        use_lerobot_unnorm: If True, re-unnormalize model outputs using
                           LeRobot's action distribution instead of Bridge's.
                           This converts predictions to LeRobot's scale.
    """
    tfrecord = Path(tfrecord_path) if tfrecord_path else DEFAULT_TFRECORD
    model = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    # Handle count-only mode
    if count_episodes_only:
        total = count_episodes(tfrecord)
        print(f"\nTotal episodes in {tfrecord.name}: {total}")
        print(f"Valid episode indices: 0 to {total-1}")
        return

    # Determine output directory
    if output_dir is None:
        suffix = f"_skip{skip_frames}" if skip_frames > 1 else ""
        if num_queries > 1:
            suffix += f"_q{num_queries}"
        output_dir = SCRIPT_DIR / "outputs_lerobot" / f"episode_{episode_index}{suffix}"
    else:
        output_dir = Path(output_dir)

    # Print header
    print("\n" + "=" * 70)
    print("SPATIALVLA INFERENCE ON LEROBOT DATA")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Episode index: {episode_index}")
    print(f"  Skip frames: {skip_frames} ({LEROBOT_FPS/skip_frames:.2f} Hz)")
    print(f"  Queries per frame: {num_queries}")
    print(f"  Temperature: {temperature} {'(sampling)' if temperature > 0 else '(greedy)'}")
    print(f"  Use LeRobot unnorm: {use_lerobot_unnorm}")
    print(f"  TFRecord: {tfrecord}")
    print(f"  Model: {model}")
    print(f"  Output: {output_dir}")

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    episode_data, total_episodes = load_episode_from_tfrecord(tfrecord, episode_index)

    if episode_data is None:
        print("\n[ERROR] Failed to load episode. Exiting.")
        return

    # =========================================================================
    # STEP 2: DOWNSAMPLE & COMPUTE CUMULATIVE DELTAS
    # =========================================================================
    episode_data = downsample_episode(episode_data, skip_frames)

    # =========================================================================
    # STEP 3: LOAD MODEL
    # =========================================================================
    model_obj, processor, device = load_spatialvla_model(model)

    # =========================================================================
    # STEP 4: RUN INFERENCE
    # =========================================================================
    pred_actions, all_chunks = run_inference(
        model_obj, processor, episode_data, device,
        num_queries=num_queries, temperature=temperature
    )

    # =========================================================================
    # STEP 4.5: (OPTIONAL) RE-UNNORMALIZE WITH LEROBOT STATS
    # =========================================================================
    if use_lerobot_unnorm:
        # Compute LeRobot's normalization stats from ALL episodes in TFRecord
        # (not just the current episode - gives more robust statistics)
        lerobot_stats = compute_dataset_norm_stats(tfrecord, skip_frames)

        print(f"\n{'='*70}")
        print("STEP 4.5: RE-UNNORMALIZE WITH LEROBOT STATS")
        print(f"{'='*70}")
        print(f"  Using dataset-wide stats ({lerobot_stats['num_episodes']} episodes, "
              f"{lerobot_stats['num_samples']} samples)")

        # Get Bridge stats from processor
        bridge_stats = processor.statistics[UNNORM_KEY]["action"]
        bridge_stats_dict = {
            'q01': np.array(bridge_stats['q01']),
            'q99': np.array(bridge_stats['q99']),
        }
        print(f"  Bridge q01: {bridge_stats_dict['q01'][:3].round(6)} (XYZ)")
        print(f"  Bridge q99: {bridge_stats_dict['q99'][:3].round(6)} (XYZ)")

        # Convert predictions from Bridge scale to LeRobot scale
        print("  Converting predictions: Bridge scale → LeRobot scale")
        pred_actions = renormalize_bridge_to_lerobot(
            pred_actions, bridge_stats_dict, lerobot_stats
        )

        # Convert all_chunks as well
        original_shape = all_chunks.shape
        all_chunks_flat = all_chunks.reshape(-1, 7)
        all_chunks_flat = renormalize_bridge_to_lerobot(
            all_chunks_flat, bridge_stats_dict, lerobot_stats
        )
        all_chunks = all_chunks_flat.reshape(original_shape)

        print(f"  Predictions now in LeRobot scale")
        print(f"  Sample pred XYZ: {pred_actions[0, :3].round(6)}")
        print(f"  Sample GT XYZ:   {episode_data.actions[0, :3].round(6)}")

    # =========================================================================
    # STEP 5: GENERATE VISUALIZATIONS
    # =========================================================================
    stats = generate_visualizations(episode_data, pred_actions, all_chunks, output_dir)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Episode: {episode_data.episode_id}")
    print(f"  Instruction: '{episode_data.instruction}'")
    print(f"  Frames: {len(episode_data.images)} @ {episode_data.effective_fps:.2f} Hz")
    print(f"\n  Errors (normalized Bridge space):")
    print(f"    Position: {stats['mean_position_error']:.4f}")
    print(f"    Rotation: {stats['mean_rotation_error']:.4f}")
    print(f"    Total: {stats['mean_total_error']:.4f}")
    print(f"    Direction (cosine): {stats['mean_cosine_similarity']:.4f}")
    print(f"\n  Outputs saved to: {output_dir}")
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SpatialVLA Inference on Converted LeRobot Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
================================================================================
EXAMPLES
================================================================================

  # Basic run (Episode 0, no downsampling)
  python smoke_test/smoke_test_lerobot.py

  # Run Episode 5 with downsampling to match Bridge scale (RECOMMENDED)
  python smoke_test/smoke_test_lerobot.py --episode_index 5 --skip_frames 7

  # Run with multiple queries to see prediction variance
  python smoke_test/smoke_test_lerobot.py --episode_index 5 --skip_frames 7 --num_queries 10

  # Use custom TFRecord (e.g., top camera only)
  python smoke_test/smoke_test_lerobot.py --tfrecord data/so101_offline_eval_top.tfrecord

  # Count total episodes
  python smoke_test/smoke_test_lerobot.py --count_episodes

================================================================================
DOWNSAMPLING GUIDE
================================================================================

  LeRobot data is 15 fps with tiny deltas (~0.003m per frame).
  SpatialVLA expects Bridge-scale deltas (~0.015m per frame).

  Recommended skip_frames values:
    --skip_frames 5  -> 3.0 Hz, ~76% of Bridge scale
    --skip_frames 6  -> 2.5 Hz, ~89% of Bridge scale
    --skip_frames 7  -> 2.1 Hz, ~103% of Bridge scale (BEST MATCH)
    --skip_frames 8  -> 1.9 Hz, ~116% of Bridge scale

================================================================================
MULTI-QUERY MODE (with sampling)
================================================================================

  Use --num_queries N to run N inference passes per frame.
  Use --temperature T to enable stochastic sampling (required for variance!)

  Without temperature > 0, the model uses greedy decoding and all queries
  return identical results!

  Example with variance:
    python smoke_test_lerobot.py --num_queries 50 --temperature 0.7

  Each query returns an action CHUNK (typically 4 actions).
  With 20 frames, 50 queries, chunk size 4:
    Total predictions = 20 * 50 * 4 = 4000

  Generates action_chunks.png showing all predictions and variance.

================================================================================
UNNORMALIZATION OPTIONS
================================================================================

  By default, model outputs are unnormalized using Bridge statistics
  (the dataset SpatialVLA was trained on).

  Use --use_lerobot_unnorm to re-unnormalize using LeRobot's statistics:
    1. Model outputs normalized [-1, 1] values
    2. Default: unnormalize with Bridge q01/q99 → Bridge-scale actions
    3. With flag: convert to LeRobot scale using dataset-wide GT statistics

  Statistics are computed from ALL episodes in the TFRecord (not just the
  current episode) for more robust/stable normalization.

  This puts predictions in the SAME SCALE as your LeRobot GT data.

================================================================================
        """
    )
    parser.add_argument("--tfrecord", type=str, default=None,
                        help="Path to TFRecord file")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to SpatialVLA model")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--episode_index", type=int, default=0,
                        help="Episode index (0-indexed)")
    parser.add_argument("--skip_frames", type=int, default=1,
                        help="Downsample factor (1=15fps, 5=3fps, 7=2.1fps)")
    parser.add_argument("--num_queries", type=int, default=1,
                        help="Queries per frame for variance analysis (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0=greedy, 0.5-1.0=stochastic). "
                             "Required for variance in multi-query mode.")
    parser.add_argument("--use_lerobot_unnorm", action="store_true",
                        help="Re-unnormalize model outputs using LeRobot stats instead of Bridge")
    parser.add_argument("--count_episodes", action="store_true",
                        help="Count episodes and exit")

    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        episode_index=args.episode_index,
        count_episodes_only=args.count_episodes,
        skip_frames=args.skip_frames,
        num_queries=args.num_queries,
        temperature=args.temperature,
        use_lerobot_unnorm=args.use_lerobot_unnorm,
        tfrecord_path=args.tfrecord,
        model_path=args.model_path
    )
