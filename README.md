# 🔧 Post-Processing Recovery Enhanced Video Tracking

Post-processing recovery enhanced video tracking system - avoiding runtime loop issues

## 📋 Project Overview

This project is an Agentic AI-based video object tracking system, specifically designed for semantic tracking of specific targets. The system integrates YOLO detection, ByteTrack tracking, CLIP semantic matching, and intelligent post-processing recovery capabilities.

### 🎯 Core Features

- **Specific Object Tracking**: Focuses on semantically described targets, not general multi-object tracking
- **CLIP Semantic Validation**: Uses CLIP model for semantic matching validation
- **Bridge Mechanism**: Handles target loss and reappearance scenarios
- **Global ID Management**: Ensures target identity consistency
- **Post-processing Recovery**: Intelligently optimizes tracking results
- **Multimodal LLM Support**: Optional MLLM parameter recommendations

## 🏗️ System Architecture

### Core Components

```
📁 Project Structure
├── main.py                    # Main entry program
├── adapter/                   # Adapter modules
│   ├── clip_matcher.py       # CLIP semantic matching
│   ├── detection_api.py      # Detection API (local + cloud)
│   ├── editing.py            # Video editing and export
│   ├── global_id.py          # Global ID management
│   ├── nlp_parser.py         # Natural language parsing
│   └── tracking.py           # ByteTrack tracking
├── agent/                     # Agent modules
│   ├── graph_build.py        # Graph building (basic version)
│   ├── graph_build_mllm.py   # Graph building (MLLM version)
│   ├── multimodal_advisor.py # Multimodal LLM advisor
│   ├── planner.py            # Planner
│   ├── policies.py           # Policies
│   └── state_types.py        # State types
├── tracker/                   # Tracker
│   ├── byte_tracker.py       # ByteTrack implementation
│   ├── kalman_filter.py      # Kalman filtering
│   └── matching.py           # Matching algorithms
├── configs/                   # Configuration files
│   └── agent.yaml            # Agent configuration
├── testcase/                  # Test cases
│   ├── *.mp4                 # Test videos
│   ├── groundtruth.json      # Ground truth annotations
│   └── *_frames/             # Extracted frames
├── testcase_results/          # Test results
└── yolo11*.pt                # YOLO model files
```

### Workflow

1. **Input Parsing**: Parse natural language instructions
2. **Object Detection**: Use YOLO to detect targets
3. **Object Tracking**: Use ByteTrack to track targets
4. **Semantic Validation**: Use CLIP to validate semantic matching
5. **Global ID Assignment**: Assign and manage global IDs
6. **Post-processing Recovery**: Intelligently optimize tracking results
7. **Video Export**: Export final tracking video

## 🚀 Quick Start

### Requirements

- Python 3.10+
- CUDA support (recommended)
- 8GB+ RAM

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Basic tracking
python main.py --video "testcase/dog.mp4" --text "the brown teddy dog" --output_dir "./output"

# Use configuration file
python main.py --video "testcase/dog.mp4" --config "configs/agent.yaml" --output_dir "./output"

# Interactive input
python main.py --video "testcase/dog.mp4" --ask --output_dir "./output"
```

### Parameter Description

- `--video`: Input video file path (required)
- `--text`: Target description text (optional, overrides prompt in config file)
- `--config`: Configuration file path (default: configs/agent.yaml)
- `--output_dir`: Output directory (default: ./output)
- `--ask`: Interactive input instructions

## 🧪 Test Cases

### Run Tests

```bash
# Run all test cases
python validate_testcases.py

# Run short tests (only first few videos)
python validate_testcases.py --short
```

### Test Results

Test results are saved in the `testcase_results/` directory:

- `tracking_results.json`: Detailed tracking results
- `tracking_summary.md`: Test summary report
- `*/`: Output videos and logs for each test case

## 🔧 Configuration

### Main Configuration Items (configs/agent.yaml)

```yaml
# Default model
default_model: "yolo11n"

# Default prompt
prompt: "track the target object"

# CLIP threshold
clip_threshold: 0.25

# Tracking parameters
track_buffer: 30
match_thresh: 0.8
track_thresh: 0.3

# Bridge parameters
max_bridge: 50
gap: 2
```

## 📊 Performance Metrics

### Evaluation Metrics

- **Detection Rate**: Proportion of frames successfully detecting targets
- **CLIP Score**: Average semantic matching score
- **Bridge Interval**: Recovery time after target loss
- **ID Consistency**: Stability of global IDs
- **Processing Speed**: FPS processing performance

### Typical Performance

- **Detection Rate**: 85-95%
- **CLIP Score**: 0.25-0.35
- **Processing Speed**: 15-25 FPS (depends on hardware)
- **Memory Usage**: 2-4GB

## 🎯 Use Cases

### Suitable Scenarios

- **Specific Object Tracking**: Track specific objects in videos
- **Semantic Search**: Find targets based on natural language descriptions
- **Video Analysis**: Analyze target behavior in videos
- **Content Creation**: Extract video segments of specific targets

### Unsuitable Scenarios

- **General Multi-object Tracking**: Track multiple different targets simultaneously
- **Real-time Tracking**: Applications requiring extremely low latency
- **Large-scale Video Processing**: Process TB-level video data

## 🔍 Troubleshooting

### Common Issues

1. **Low Detection Rate**
   - Check if target description is accurate
   - Adjust CLIP threshold
   - Try different YOLO models

2. **Tracking Loss**
   - Increase track_buffer parameter
   - Adjust max_bridge parameter
   - Check video quality

3. **Insufficient Memory**
   - Use smaller YOLO model (yolo11n)
   - Reduce batch size
   - Disable unnecessary features

### Debug Mode

```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
python main.py --video "test.mp4" --text "target"
```

## 📈 Development Plan

### Completed Features

- ✅ Basic tracking system
- ✅ CLIP semantic validation
- ✅ Global ID management
- ✅ Post-processing recovery
- ✅ Multimodal LLM support
- ✅ Test case validation

### Planned Features

- 🔄 Real-time tracking optimization
- 🔄 More detection model support
- 🔄 Cloud deployment support
- 🔄 Web interface development

## 📄 License

This project is for academic research purposes only.

## 👥 Contributing

Welcome to submit Issues and Pull Requests to improve the project.

## 📞 Contact

If you have any questions, please contact through GitHub Issues.

---

**Note**: This project focuses on semantic tracking of specific targets, which is fundamentally different from general multi-object tracking systems. Please ensure you understand the system's working principles and applicable scenarios before use.