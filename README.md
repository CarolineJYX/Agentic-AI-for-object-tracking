# 🎯 Agentic Video Object Tracking & Editing System

A sophisticated multi-modal AI system for intelligent video object tracking and editing, built with an agent-adapter architecture and LangGraph orchestration.

## 🏗️ Architecture

```
📁 Project Structure
├── agent/              # 🧠 Decision Logic Layer
│   ├── planner.py         # Strategy Planning
│   ├── graph_build.py     # Workflow Orchestration  
│   └── state_types.py     # State Management
├── adapter/            # 🔧 Implementation Layer
│   ├── detection.py       # YOLO Detection Adapter
│   ├── tracking.py        # ByteTrack Tracking Adapter
│   ├── clip_matcher.py    # CLIP Semantic Matching
│   └── editing.py         # Video Editing Functions
├── configs/            # ⚙️ Configuration Files
└── agent_main.py       # 📱 Unified Interface
```

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

### Basic Usage
```bash
python agent_main.py --video demo.mp4 --text "track the brown teddy dog" --output_dir ./output
```

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
