# 🎯 Agentic AI for Video Object Tracking

> **A pioneering multi-modal AI system that autonomously optimizes video object tracking parameters through intelligent agent-based decision making.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-enabled-green.svg)](https://github.com/langchain-ai/langgraph)
[![Multi-Modal](https://img.shields.io/badge/Multi--Modal-CLIP%2BLLM-orange.svg)](https://github.com/mlfoundations/open_clip)

## 🌟 Key Innovation

This project introduces the **first Agentic AI approach** to video object tracking, where AI agents autonomously analyze video content and optimize tracking parameters without manual tuning.

## 🏗️ Architecture Overview

```
🖥️  User Interface Layer
    ├── agent_main.py              # Traditional System  
    └── agent_main_mllm.py         # MLLM-Enhanced System

🧠  Agent Layer (Decision & Strategy)
    ├── agent/planner.py           # Parameter Optimization
    ├── agent/policies.py          # Decision Policies
    ├── multimodal_advisor.py      # MLLM-based Analysis
    └── agent/state_types.py       # State Management

🔧  Adapter Layer (Tool Implementation)  
    ├── adapter/detection_api.py   # YOLO Integration
    ├── adapter/tracking.py        # ByteTrack Wrapper
    ├── adapter/clip_matcher.py    # Semantic Matching
    ├── adapter/editing.py         # Video Processing
    └── adapter/global_id.py       # Identity Management

⚙️  Foundation Layer
    ├── tracker/                   # ByteTrack Implementation
    └── configs/                   # System Configuration
```

For detailed architecture documentation, see [ARCHITECTURE.md](./ARCHITECTURE.md).

## 🚀 Features

- **🎯 Semantic Object Tracking**: Natural language description to video object mapping
- **🧠 Intelligent Decision Making**: Dynamic parameter adjustment based on video analysis  
- **🔗 Multi-Modal Integration**: YOLO + ByteTrack + CLIP + LangGraph
- **📊 Global Identity Management**: Consistent object tracking across frames
- **🎬 Intelligent Video Editing**: Automatic segment merging with bridge algorithms
- **⚡ Performance Optimization**: Model caching and batch processing

## 🛠️ Technology Stack

- **Computer Vision**: YOLOv11, ByteTrack
- **Multi-Modal AI**: OpenCLIP  
- **Workflow Orchestration**: LangGraph
- **Video Processing**: OpenCV, MoviePy
- **Architecture Pattern**: Agent-Adapter Separation

## 📦 Installation

```bash
# Create conda environment
conda create -n capstone310 python=3.10
conda activate capstone310

# Install dependencies
pip install ultralytics
pip install open-clip-torch
pip install langgraph
pip install opencv-python
pip install moviepy
```

## 🎮 Usage

### 🤖 MLLM-Enhanced System (Recommended)
```bash
# Intelligent parameter optimization with multi-modal LLM
python agent_main_mllm.py --video demo2.mp4 --text "track the woman in white clothes dancing" --output_dir ./output_mllm
```

### 🔧 Traditional System (For Comparison)
```bash
# Manual parameter configuration
python agent_main.py --video demo2.mp4 --text "track the woman in white clothes dancing" --output_dir ./output_traditional
```

### 📊 Key Differences

| Feature | Traditional System | MLLM-Enhanced System |
|---------|-------------------|---------------------|
| **Parameter Tuning** | Manual configuration | Fully automated |
| **Video Analysis** | Rule-based | AI-powered analysis |
| **Target Detection** | Basic matching | Semantic understanding |
| **Adaptation** | Static parameters | Dynamic optimization |
| **User Experience** | Technical expertise required | Natural language input |

### Parameters
- `--video`: Input video file path
- `--text`: Natural language description of target object  
- `--output_dir`: Output directory for processed video

## 🧠 System Workflow

1. **Detection**: YOLO identifies potential objects in each frame
2. **Tracking**: ByteTrack maintains object trajectories across frames  
3. **Semantic Matching**: CLIP matches objects to text description
4. **Global Identity**: Assigns consistent IDs to tracked objects
5. **Intelligent Planning**: Dynamically adjusts parameters based on video characteristics
6. **Video Editing**: Merges tracking segments and exports final video

## 📊 Key Components

### Agent Layer (Decision Logic)
- **Planner**: Dynamic strategy adjustment (bridge values, thresholds)
- **State Manager**: Unified state management across workflow nodes
- **Global ID Manager**: Object identity consistency

### Adapter Layer (Tool Implementation)  
- **Detection Adapter**: YOLO model integration
- **Tracking Adapter**: ByteTrack algorithm wrapper
- **CLIP Adapter**: Multi-modal semantic matching
- **Editing Adapter**: Video processing and export

### LangGraph Orchestration
- **Stateful Workflow**: Node-based processing pipeline
- **Dynamic Routing**: Conditional workflow paths
- **Error Handling**: Robust error recovery mechanisms

## 🎯 Academic Contributions

- **Agent-Adapter Architecture**: Clean separation of concerns for AI systems
- **Multi-Modal Integration**: Seamless combination of vision and language models
- **Dynamic Parameter Optimization**: Intelligent adaptation to video characteristics
- **Global Identity Management**: Novel approach to cross-frame object consistency

## 📈 Performance Features

- **Model Caching**: Avoid repeated model loading
- **Batch Processing**: Efficient frame processing
- **Memory Optimization**: Intelligent state management
- **Adaptive Thresholding**: Dynamic quality control

## 🔬 Research Applications

This system demonstrates:
- **Agentic AI Principles**: Autonomous decision-making and adaptation
- **Multi-Modal AI Integration**: Vision-language model coordination  
- **Workflow Orchestration**: Complex AI system management
- **Real-World Problem Solving**: Practical video processing challenges

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{agentic_video_tracking_2024,
  title={Agentic Video Object Tracking and Editing System},
  author={[Your Name]},
  year={2024},
  note={Master's Thesis Project}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaborations, please contact: [Your Email]
