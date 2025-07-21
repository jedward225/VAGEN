# SPOC Environment Integration for VAGEN

## Overview

This document presents a comprehensive integration of the **SPOC (Situated Procedures for Object manipulation and Commonsense) environment** into the VAGEN framework. The SPOC environment enables training and evaluation of embodied AI agents on complex object manipulation tasks using the Stretch robot in realistic household scenarios powered by AI2-THOR physics simulation.

**How to SetUp:**

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

conda create -n vagen python=3.10 -y
conda activate vagen

git clone https://github.com/allenai/spoc-robot-training.git

cd spoc-robot-training

pip install -r requirements.txt
pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+5d0ab8ab8760eb584c5ae659c2b2b951cab23246

pip install h5py
pip install objathor
cd ../

git clone https://github.com/JamesKrW/verl.git
cd verl
pip install -e .
cd ../

git clone https://github.com/jedward225/VAGEN.git
cd VAGEN
bash scripts/install.sh
sudo apt-get update
sudo apt-get install -y vulkan-utils libvulkan1 mesa-vulkan-drivers
sudo apt-get install -y xvfb mesa-utils
sudo apt install net-tools
sudo apt-get install -y libvulkan1
sudo apt install x11-apps
sudo apt install xvfb -y

python -c 'from transformers import AutoTokenizer, AutoModel;
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct");
AutoModel.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
from transformers import AutoImageProcessor; 
AutoImageProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")'


export SPOC_DATA_PATH=/root/spoc_data/fifteen
export FORCE_HEADLESS=1
export AI2THOR_ALLOW_RESOLUTION_MISMATCH=1
./quick_restart.sh
```

## Technical Architecture

### Core Components

#### Environment Class (`SpocEnv`)
The central environment class inherits from `BaseEnv` and implements:
- **Initialization Pipeline**: Multi-stage AI2-THOR controller setup with fallback mechanisms for different rendering platforms
- **Episode Management**: Deterministic episode loading from SPOC dataset with proper scene initialization and agent teleportation
- **Action Execution**: Comprehensive discrete action mapping to AI2-THOR's continuous control API
- **Observation Generation**: Dual-camera image concatenation with proprioceptive arm state integration
- **Success Evaluation**: Object-holding based success criteria for fetch tasks

#### Service Architecture (`SpocService`)
Implements parallel environment management with:
- **Concurrent Operations**: ThreadPoolExecutor-based parallel environment creation, reset, and stepping
- **Error Recovery**: Robust error handling with automatic environment recreation on failures
- **GPU Load Balancing**: Intelligent GPU allocation across multiple devices to optimize resource utilization
- **Batch Processing**: Efficient batch operations for training and evaluation scenarios

#### Configuration System
- **`SpocEnvConfig`**: Comprehensive environment configuration covering dataset paths, task types, visual settings, prompt formats, and reward parameters
- **`SpocServiceConfig`**: Service-level configuration for multi-environment deployment including worker limits and device allocation

#### Task Loading Pipeline (`ChoresDataset`)
- **HDF5 Integration**: Direct integration with SPOC's native HDF5 format for efficient data loading
- **Scene Mapping**: Intelligent mapping between SPOC's arbitrary house indices and AI2-THOR's supported FloorPlan ranges (1-30, 201-230, 301-330, 401-430)
- **JSON Processing**: Robust parsing of templated task specifications with error handling for corrupted data
- **Instruction Generation**: Automatic natural language instruction generation from structured task specifications

### Prompt Engineering Framework

#### Format Taxonomy
1. **Free Think**: Basic thinking-then-action format for straightforward reasoning
2. **Grounding**: Structured observation-reasoning-action pipeline for explicit scene understanding
3. **Worldmodeling**: Reasoning-prediction-action format for forward planning capabilities
4. **Grounding + Worldmodeling**: Comprehensive observation-reasoning-prediction-action pipeline for maximum interpretability
5. **No Think**: Direct action format for fast inference and simplified training

#### Action Space Design
The 20-action discrete space provides hierarchical control:
- **Navigation Actions (8)**: Base movement and rotation with both coarse (30°, 0.2m) and fine (6°) control
- **Manipulation Actions (12)**: Complete arm control including extension/retraction, height adjustment, wrist rotation, and object interaction

#### Observation Engineering
- **Visual Component**: Side-by-side concatenated 512×512 images from navigation and manipulation cameras with 6-pixel border cropping for consistency
- **Proprioceptive Component**: Structured arm state including 3D position, wrist orientation, and gripper status
- **Instruction Integration**: Natural language task descriptions seamlessly integrated into prompt templates

## Implementation Highlights

### Reliability Features
- **Multi-Platform Initialization**: Graceful fallback between CloudRendering and Linux64 platforms for maximum compatibility
- **Recursive Reset Protection**: Intelligent episode selection with retry limits to prevent infinite recursion on corrupted scenes
- **Memory Management**: Explicit cleanup protocols for AI2-THOR controllers to prevent memory leaks in long-running training

### Performance Optimizations
- **Parallel Environment Creation**: Concurrent initialization of multiple AI2-THOR instances with proper synchronization
- **Efficient Image Processing**: Optimized NumPy operations for real-time image concatenation and format conversion
- **Lazy Loading**: On-demand dataset loading to minimize memory footprint during large-scale training

### Development Features
- **Comprehensive Logging**: Detailed debug output for action parsing, environment state, and reward computation
- **Interactive Testing**: Built-in interactive mode for manual environment exploration and debugging
- **Visualization Support**: Integrated bounding box drawing utilities for object detection validation

## Dataset Integration

### SPOC ChoresDataset Support
- **Full Compatibility**: Native support for SPOC's HDF5-based ChoresDataset format
- **Task Type Filtering**: Automatic filtering by task type (FetchType, ObjectNavType, etc.) and dataset split
- **Robust Scene Handling**: Intelligent scene name construction with validation and fallback mechanisms
- **Metadata Extraction**: Complete extraction of agent poses, object targets, and task specifications

### Evaluation Datasets
The implementation includes five curated evaluation datasets:
- **Base**: Core object fetching scenarios
- **Common Sense**: Tasks requiring logical reasoning about object properties
- **Complex Instruction**: Multi-step manipulation sequences
- **Visual Appearance**: Object identification based on visual features
- **Long Horizon**: Extended manipulation tasks requiring planning

## Usage and Integration

### Basic Environment Setup
```python
from vagen.env.spoc import SpocEnv, SpocEnvConfig

config = SpocEnvConfig(
    data_path="/path/to/spoc_data/fifteen_type",
    task_type="FetchType",
    prompt_format="grounding_worldmodeling",
    resolution=512
)

env = SpocEnv(config)
obs, info = env.reset(seed=42)
```

### Service-Level Deployment
```python
from vagen.env.spoc import SpocService, SpocServiceConfig

service_config = SpocServiceConfig(
    max_workers=4,
    devices=[0, 1, 2, 3]
)

service = SpocService(service_config)
# Parallel environment management for training
```

### Training Integration
The SPOC environment seamlessly integrates with VAGEN's training pipeline, supporting:
- **Multi-GPU Training**: Automatic GPU allocation across available devices
- **Batch Operations**: Efficient parallel rollout collection
- **Reward Engineering**: Configurable reward functions for different learning objectives
- **Prompt Format Selection**: Easy switching between reasoning paradigms for research

## Future Extensions

The modular architecture supports several potential extensions:
- **Additional Task Types**: Easy integration of new SPOC task categories
- **Enhanced Observation Spaces**: Support for depth, semantic segmentation, or additional camera views
- **Custom Action Spaces**: Extension to continuous control or hierarchical action spaces
- **Multi-Agent Scenarios**: Framework extension for collaborative manipulation tasks

## Technical Requirements

- **AI2-THOR**: Version 5.0.0+ with Stretch robot support
- **Python Dependencies**: h5py, numpy, Pillow, OpenCV for dataset loading and image processing
- **Hardware**: CUDA-compatible GPUs for efficient parallel environment execution
- **Dataset**: SPOC ChoresDataset in HDF5 format

This SPOC integration represents a significant advancement in embodied AI research infrastructure, providing researchers with a robust, scalable platform for training and evaluating sophisticated manipulation agents in realistic household environments.