# Smoke Test Directory

This directory contains all files related to smoke testing SpatialVLA inference on Bridge V2 dataset episodes.

## Structure

```
smoke_test/
├── dataset_explorer.py       # Streamlit interactive dashboard ⭐
├── smoke_test_inference.py   # Main inference script
├── visualize_dataset.ipynb   # Jupyter notebook for dataset visualization
├── download_bridge_episodes.py # Script to download dataset episodes
├── data/                      # Bridge V2 dataset files
│   └── bridge_dataset-train.tfrecord-*.tfrecord
└── outputs/                   # Generated plots and statistics
    ├── episode_0/
    ├── episode_1/
    └── ...
```

## Usage

### Interactive Dataset Explorer (Streamlit Dashboard) ⭐

Launch the interactive dashboard with dropdowns for file and episode selection:

**For Remote Access (SSH from local PC to thor server):**

1. **On remote server (thor):**
   ```bash
   streamlit run smoke_test/dataset_explorer.py --server.address 0.0.0.0 --server.port 8501
   ```

2. **On your local PC (in a new terminal):**
   Set up SSH port forwarding:
   ```bash
   ssh -L 8501:localhost:8501 thor@10.0.0.103
   ```
   (Replace with your actual SSH connection details)

3. **Open in your local browser:**
   ```
   http://localhost:8501
   ```

**For Local Access (if running directly on thor):**
```bash
streamlit run smoke_test/dataset_explorer.py
```

Features:
- **TFRecord File Dropdown**: Select which dataset file to explore
- **Episode ID Dropdown**: Select specific episode by ID
- **Step-by-Step Visualization**: Browse through all steps in an episode with images and actions
- **Episode GIF Generation**: Create and download animated GIFs of episode images
- **Episode Summary**: View action trajectories and statistics

### Inference Script

See the docstring in `smoke_test_inference.py` for detailed usage examples.

Quick start:
```bash
# Run Episode 0 (first episode)
python smoke_test/smoke_test_inference.py

# Run Episode 5
python smoke_test/smoke_test_inference.py --episode_index 5

# Count total episodes
python smoke_test/smoke_test_inference.py --count_episodes
```

### Dataset Visualization (Jupyter Notebook)

Open `visualize_dataset.ipynb` in Jupyter to explore and visualize Bridge V2 dataset episodes, including images, actions, states, and metadata.

## Outputs

Each episode run generates:
- `action_comparison.png` - Position and rotation component comparisons
- `trajectory_3d.png` - 3D end-effector trajectory visualization
- `gripper_comparison.png` - Gripper open/close comparison
- `trajectory.gif` - Animated GIF showing trajectory evolution
- `statistics.txt` - Detailed error statistics
