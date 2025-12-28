#!/usr/bin/env python3
"""
Streamlit Dashboard for Bridge V2 Dataset Exploration

Interactive dashboard to explore Bridge V2 dataset episodes with:
- TFRecord file selection
- Episode ID selection
- Step-by-step visualization
- Episode GIF generation

To run on remote server and access from local PC:
1. Run: streamlit run smoke_test/dataset_explorer.py --server.address 0.0.0.0 --server.port 8501
2. Set up SSH port forwarding: ssh -L 8501:localhost:8501 user@remote_host
3. Open browser: http://localhost:8501
"""

import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from pathlib import Path
from PIL import Image
import imageio
import urllib.request
import io
import base64

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set page config
st.set_page_config(page_title="Bridge V2 Dataset Explorer", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
TFRECORD_BASE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/"

@st.cache_data
def list_tfrecord_files():
    """List all TFRecord files in data directory."""
    if not DATA_DIR.exists():
        return []
    return sorted([f.name for f in DATA_DIR.glob("*.tfrecord*")])

@st.cache_data
def get_episode_ids(tfrecord_file):
    """Extract all episode IDs and instructions from a TFRecord file."""
    filepath = DATA_DIR / tfrecord_file
    if not filepath.exists():
        return []

    episodes_metadata = []
    raw_dataset = tf.data.TFRecordDataset(str(filepath))

    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        # Extract ID
        ep_id = "0000"
        if 'episode_metadata/episode_id' in features:
            feat_id = features['episode_metadata/episode_id']
            if feat_id.int64_list.value:
                ep_id = str(feat_id.int64_list.value[0])
            elif feat_id.bytes_list.value:
                ep_id_val = feat_id.bytes_list.value[0].decode('utf-8')
                if ep_id_val and ep_id_val.lower() != "n/a":
                    ep_id = ep_id_val

        # Extract Instruction
        instruction = "N/A"
        if 'steps/language_instruction' in features:
            vals = features['steps/language_instruction'].bytes_list.value
            if vals:
                instruction = vals[0].decode('utf-8')

        episodes_metadata.append({'id': ep_id, 'instruction': instruction})

    return episodes_metadata

@st.cache_data
def load_episode(tfrecord_file, episode_index):
    """Load a specific episode from TFRecord."""
    filepath = DATA_DIR / tfrecord_file
    raw_dataset = tf.data.TFRecordDataset(str(filepath))

    for idx, raw_record in enumerate(raw_dataset):
        if idx == episode_index:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            features = example.features.feature

            # Extract metadata
            ep_id = "0000"
            if 'episode_metadata/episode_id' in features:
                feat_id = features['episode_metadata/episode_id']
                if feat_id.int64_list.value:
                    ep_id = str(feat_id.int64_list.value[0])
                elif feat_id.bytes_list.value:
                    ep_id = feat_id.bytes_list.value[0].decode('utf-8')
                    if not ep_id or ep_id.lower() == "n/a":
                        ep_id = "0000"

            file_path = "N/A"
            if 'episode_metadata/file_path' in features:
                vals = features['episode_metadata/file_path'].bytes_list.value
                if vals: file_path = vals[0].decode('utf-8')

            instruction = "N/A"
            if 'steps/language_instruction' in features:
                vals = features['steps/language_instruction'].bytes_list.value
                if vals: instruction = vals[0].decode('utf-8')

            # Extract images
            images_dict = {}
            for i in range(4):
                key = f'steps/observation/image_{i}'
                if key in features:
                    img_bytes_list = list(features[key].bytes_list.value)  # Convert protobuf to list
                    images_dict[i] = []
                    for b in img_bytes_list:
                        try:
                            img = tf.image.decode_image(b).numpy()
                            images_dict[i].append(img)  # Keep as numpy array (Streamlit can handle this)
                        except:
                            images_dict[i].append(None)

            # Extract actions (convert protobuf repeated fields to lists)
            actions = []
            if 'steps/action' in features:
                action_floats = list(features['steps/action'].float_list.value)  # Convert protobuf to list
                num_steps = len(images_dict[0]) if 0 in images_dict and images_dict[0] else len(action_floats) // 7
                if num_steps > 0:
                    actions = np.array(action_floats).reshape(num_steps, 7)  # Keep as numpy array

            # Extract states (convert protobuf repeated fields to lists)
            states = []
            if 'steps/observation/state' in features:
                state_floats = list(features['steps/observation/state'].float_list.value)  # Convert protobuf to list
                num_steps = len(images_dict[0]) if 0 in images_dict and images_dict[0] else len(state_floats) // 7
                if num_steps > 0:
                    state_dim = len(state_floats) // num_steps
                    states = np.array(state_floats).reshape(num_steps, state_dim)  # Keep as numpy array

            # Extract step flags (convert protobuf repeated fields to lists for pickling)
            is_first = list(features['steps/is_first'].int64_list.value) if 'steps/is_first' in features else []
            is_last = list(features['steps/is_last'].int64_list.value) if 'steps/is_last' in features else []
            is_terminal = list(features['steps/is_terminal'].int64_list.value) if 'steps/is_terminal' in features else []
            rewards = list(features['steps/reward'].float_list.value) if 'steps/reward' in features else []
            discounts = list(features['steps/discount'].float_list.value) if 'steps/discount' in features else []

            num_steps = len(images_dict[0]) if 0 in images_dict and images_dict[0] else len(actions)

            return {
                'episode_id': ep_id,
                'file_path': file_path,
                'instruction': instruction,
                'images': images_dict,
                'actions': actions,
                'states': states,
                'is_first': is_first,
                'is_last': is_last,
                'is_terminal': is_terminal,
                'rewards': rewards,
                'discounts': discounts,
                'num_steps': num_steps
            }

    return None

def download_tfrecord(filename):
    """Download a TFRecord file from the Bridge V2 dataset."""
    url = TFRECORD_BASE_URL + filename
    filepath = DATA_DIR / filename

    if filepath.exists():
        return True, f"{filename} already exists"

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, filepath)
        size_mb = filepath.stat().st_size / 1024 / 1024
        return True, f"Downloaded {filename} ({size_mb:.1f} MB)"
    except Exception as e:
        if filepath.exists():
            filepath.unlink()  # Remove partial download
        return False, f"Failed to download {filename}: {e}"

def create_episode_gif(images_dict, actions, output_path, instruction="", episode_id=""):
    """Create GIF from episode images with 3D trajectory visualization."""
    if 0 not in images_dict or not images_dict[0]:
        return None

    images = []
    for img in images_dict[0]:
        if img is not None:
            images.append(Image.fromarray(img))

    if not images:
        return None

    # Compute cumulative positions from action deltas
    positions = np.cumsum(actions[:, :3], axis=0) if len(actions) > 0 else np.zeros((len(images), 3))

    # Create frames with image and 3D trajectory
    frames = []
    num_steps = len(images)

    # Set up 3D plot bounds
    if len(positions) > 0:
        margin = 0.02
        x_min, x_max = positions[:, 0].min() - margin, positions[:, 0].max() + margin
        y_min, y_max = positions[:, 1].min() - margin, positions[:, 1].max() + margin
        z_min, z_max = positions[:, 2].min() - margin, positions[:, 2].max() + margin
    else:
        x_min, x_max, y_min, y_max, z_min, z_max = -0.1, 0.1, -0.1, 0.1, -0.1, 0.1

    # Use Agg backend for faster non-interactive saving
    plt.switch_backend('Agg')

    for frame_idx in range(num_steps):
        # Create small, low-res figure for speed
        fig = plt.figure(figsize=(8, 4), dpi=50)
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

        # Left: Image
        ax_img = fig.add_subplot(gs[0])
        ax_img.imshow(images[frame_idx])
        ax_img.set_title(f"Step {frame_idx+1}/{num_steps}", fontsize=8)
        ax_img.axis('off')

        # Add action info text (simplified)
        if frame_idx < len(actions):
            act = actions[frame_idx]
            gt_grip = "open" if act[6] > 0.5 else "close"
            info_text = f"XYZ: [{act[0]:.2f}, {act[1]:.2f}, {act[2]:.2f}]\n"
            info_text += f"Grip: {gt_grip}"
            ax_img.text(0.5, -0.1, info_text, transform=ax_img.transAxes, fontsize=7,
                       ha='center', va='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Right: 3D Trajectory
        ax_3d = fig.add_subplot(gs[1], projection='3d')
        ax_3d.set_xlim([x_min, x_max])
        ax_3d.set_ylim([y_min, y_max])
        ax_3d.set_zlim([z_min, z_max])

        if frame_idx > 0:
            ax_3d.plot(positions[:frame_idx+1, 0], positions[:frame_idx+1, 1], positions[:frame_idx+1, 2],
                      'b-', linewidth=1, label='Traj')

        # Current position
        ax_3d.scatter(*positions[frame_idx], c='red', s=50, marker='o', edgecolors='black', zorder=5)

        # Start position
        if frame_idx > 0:
            ax_3d.scatter(*positions[0], c='green', s=50, marker='*', edgecolors='black')

        ax_3d.set_title(f'Trajectory', fontsize=8)
        ax_3d.view_init(elev=20, azim=45 + frame_idx * 2)

        # Remove axis labels to save space/time
        ax_3d.set_xticklabels([])
        ax_3d.set_yticklabels([])
        ax_3d.set_zticklabels([])

        plt.tight_layout()

        # Convert figure to image using BytesIO
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=50, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img.load()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        frames.append(img)
        plt.close(fig)
        buf.close()

    # Save as GIF using imageio for better compatibility
    if frames:
        # Ensure all frames have the same size
        if len(frames) > 1:
            # Use the size of the first frame as reference
            target_size = frames[0].size
            resized_frames = []
            for frame in frames:
                if frame.size != target_size:
                    # Use NEAREST for speed
                    frame = frame.resize(target_size, Image.Resampling.NEAREST)
                resized_frames.append(frame)
            frames = resized_frames

        # Convert PIL Images to numpy arrays for imageio
        frame_arrays = [np.array(frame) for frame in frames]
        imageio.mimsave(
            output_path,
            frame_arrays,
            duration=0.8,  # 800ms per frame
            loop=0,
            format='GIF'
        )
        return output_path
    return None

def main():
    st.title("🔍 Bridge V2 Dataset Explorer")
    st.markdown("Explore Bridge V2 dataset episodes interactively")

    # Sidebar for file and episode selection
    st.sidebar.header("📁 Dataset Selection")

    # Cache Control
    if st.sidebar.button("🧹 Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")

    # 1. TFRecord file selection (type or choose)
    tfrecord_files = list_tfrecord_files()

    # Allow user to type or select
    input_method = st.sidebar.radio("File Selection Method", ["Select from list", "Type filename"], index=0)

    if input_method == "Select from list":
        if not tfrecord_files:
            st.sidebar.warning(f"No TFRecord files found in {DATA_DIR}")
            st.sidebar.info("Use 'Type filename' to download a file")
            selected_file = None
        else:
            selected_file = st.sidebar.selectbox(
                "Select TFRecord File",
                tfrecord_files,
                index=0
            )
    else:
        filename_input = st.sidebar.text_input(
            "Enter TFRecord filename",
            value="bridge_dataset-train.tfrecord-00000-of-01024",
            help="Enter the full filename (e.g., bridge_dataset-train.tfrecord-00000-of-01024)"
        )
        selected_file = filename_input.strip() if filename_input else None

    # Check if file exists, offer download if not
    if selected_file:
        filepath = DATA_DIR / selected_file
        if not filepath.exists():
            st.sidebar.warning(f"⚠️ File not found: {selected_file}")
            if st.sidebar.button(f"📥 Download {selected_file}"):
                with st.spinner(f"Downloading {selected_file}..."):
                    success, message = download_tfrecord(selected_file)
                    if success:
                        st.sidebar.success(message)
                        st.cache_data.clear()  # Clear cache to refresh file list
                        st.rerun()
                    else:
                        st.sidebar.error(message)
            return

    if not selected_file:
        st.info("Please select or enter a TFRecord filename")
        return

    # 2. Episode Selection
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Episode Selection")

        with st.spinner("Loading episode metadata..."):
            episodes_metadata = get_episode_ids(selected_file)

        if not episodes_metadata:
            st.warning("No episodes found in this file")
            return

        # Search functionality
        search_query = st.text_input("🔍 Search Instruction", help="Filter by instruction text").lower()

        # Filter episodes based on search query
        if search_query:
            filtered_indices = [
                i for i, ep in enumerate(episodes_metadata)
                if search_query in ep['instruction'].lower()
            ]
        else:
            filtered_indices = list(range(len(episodes_metadata)))

        if not filtered_indices:
            st.warning(f"No episodes found matching '{search_query}'")
            return

        # Select from filtered list
        selected_filtered_idx = st.selectbox(
            f"Select Episode ({len(filtered_indices)} matches)",
            range(len(filtered_indices)),
            index=0,
            format_func=lambda i: f"Ep {episodes_metadata[filtered_indices[i]]['id']} (Idx {filtered_indices[i]}): {episodes_metadata[filtered_indices[i]]['instruction'][:40]}"
        )

        # Get actual index in the file
        episode_index = filtered_indices[selected_filtered_idx]
        selected_ep_id = episodes_metadata[episode_index]['id']

    # Load episode data
    with st.spinner(f"Loading Episode {selected_ep_id}..."):
        episode_data = load_episode(selected_file, episode_index)

    if episode_data is None:
        st.error("Failed to load episode")
        return

    # Display metadata
    st.header("📋 Episode Metadata")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Episode ID", episode_data['episode_id'])
    with col2:
        st.metric("Number of Steps", episode_data['num_steps'])

    # Display instruction with full text wrapping
    instruction = episode_data['instruction']
    if not instruction or instruction == "N/A":
        st.info("**Instruction:** _No instruction provided_")
    else:
        st.info(f"**Instruction:** {instruction}")

    st.text(f"File Path: {episode_data['file_path']}")

    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["📈 Step-by-Step Visualization", "🎬 Episode GIF", "📊 Episode Summary"])

    with tab1:
        st.header("Step-by-Step Visualization")

        # Step selector - dropdown instead of slider
        step_options = list(range(episode_data['num_steps']))
        step_idx = st.selectbox(
            "Select Step",
            step_options,
            index=0,
            format_func=lambda x: f"Step {x+1}/{episode_data['num_steps']}"
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            # Display images for selected step
            images_to_show = []
            for img_idx in range(4):
                if img_idx in episode_data['images'] and step_idx < len(episode_data['images'][img_idx]):
                    img = episode_data['images'][img_idx][step_idx]
                    if img is not None:
                        images_to_show.append((f"image_{img_idx}", img))

            if images_to_show:
                cols = len(images_to_show)
                fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
                if cols == 1:
                    axes = [axes]
                for idx, (name, img) in enumerate(images_to_show):
                    axes[idx].imshow(img)
                    axes[idx].set_title(f"{name}\n{img.shape}")
                    axes[idx].axis('off')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("No images available for this step")

        with col2:
            st.subheader("Step Metadata")
            if step_idx < len(episode_data['is_first']):
                st.write(f"**is_first:** {bool(episode_data['is_first'][step_idx])}")
            if step_idx < len(episode_data['is_last']):
                st.write(f"**is_last:** {bool(episode_data['is_last'][step_idx])}")
            if step_idx < len(episode_data['is_terminal']):
                st.write(f"**is_terminal:** {bool(episode_data['is_terminal'][step_idx])}")
            if step_idx < len(episode_data['rewards']):
                st.write(f"**reward:** {episode_data['rewards'][step_idx]:.4f}")
            if step_idx < len(episode_data['discounts']):
                st.write(f"**discount:** {episode_data['discounts'][step_idx]:.4f}")

            st.subheader("Action (7D)")
            if step_idx < len(episode_data['actions']):
                act = episode_data['actions'][step_idx]
                st.write(f"**Position XYZ:**")
                st.write(f"  X: {act[0]:.4f}")
                st.write(f"  Y: {act[1]:.4f}")
                st.write(f"  Z: {act[2]:.4f}")
                st.write(f"**Rotation RPY:**")
                st.write(f"  Roll: {act[3]:.4f}")
                st.write(f"  Pitch: {act[4]:.4f}")
                st.write(f"  Yaw: {act[5]:.4f}")
                st.write(f"**Gripper:** {act[6]:.4f} ({'Open' if act[6] > 0.5 else 'Closed'})")

            if step_idx < len(episode_data['states']):
                state = episode_data['states'][step_idx]
                st.write(f"**State ({len(state)}D):**")
                st.write(f"{state[:5]}..." if len(state) > 5 else str(state))

    with tab2:
        st.header("Episode GIF Generation")
        st.markdown("Generate an animated GIF with camera views and 3D trajectory visualization")

        if 0 in episode_data['images'] and episode_data['images'][0]:
            if st.button("Generate GIF"):
                with st.spinner("Creating GIF with 3D trajectory..."):
                    gif_path = Path("/tmp") / f"episode_{episode_data['episode_id']}.gif"
                    result = create_episode_gif(
                        episode_data['images'],
                        episode_data['actions'],
                        gif_path,
                        instruction=episode_data['instruction'],
                        episode_id=episode_data['episode_id']
                    )

                    if result:
                        st.success("GIF created successfully!")
                        with open(gif_path, "rb") as f:
                            gif_bytes = f.read()
                            st.download_button(
                                label="Download GIF",
                                data=gif_bytes,
                                file_name=f"episode_{episode_data['episode_id']}.gif",
                                mime="image/gif"
                            )

                        # Display GIF using base64 to ensure animation plays
                        b64_gif = base64.b64encode(gif_bytes).decode('utf-8')
                        st.markdown(
                            f'<img src="data:image/gif;base64,{b64_gif}" width="100%">',
                            unsafe_allow_html=True
                        )
                    else:
                        st.error("Failed to create GIF")
        else:
            st.warning("No images available to create GIF")

    with tab3:
        st.header("Episode Summary")

        # Plot action trajectories
        if len(episode_data['actions']) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f"Episode {episode_data['episode_id']}: {episode_data['instruction']}", fontsize=12)

            # Position components
            for i, (label, color) in enumerate(zip(['X', 'Y', 'Z'], ['r', 'g', 'b'])):
                axes[0, i].plot(episode_data['actions'][:, i], f'{color}-', linewidth=2)
                axes[0, i].set_xlabel('Step')
                axes[0, i].set_ylabel(f'{label} Delta (m)')
                axes[0, i].set_title(f'Position {label}')
                axes[0, i].grid(True, alpha=0.3)

            # Rotation components
            for i, (label, color) in enumerate(zip(['Roll', 'Pitch', 'Yaw'], ['r', 'g', 'b'])):
                axes[1, i].plot(episode_data['actions'][:, i+3], f'{color}-', linewidth=2)
                axes[1, i].set_xlabel('Step')
                axes[1, i].set_ylabel(f'{label} Delta (rad)')
                axes[1, i].set_title(f'Rotation {label}')
                axes[1, i].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Gripper plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(episode_data['actions'][:, 6], 'b-', linewidth=2, marker='o')
            ax.axhline(y=0.5, color='gray', linestyle=':', label='Open/Close threshold')
            ax.set_xlabel('Step')
            ax.set_ylabel('Gripper Value')
            ax.set_title(f"Gripper Command")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)
            st.pyplot(fig)
            plt.close()

if __name__ == "__main__":
    main()
