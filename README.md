# Echo: A Large Language Model with Temporal Episodic Memory

![Echo](https://img.shields.io/badge/Model-Echo--7B%20%7C%20Echo--72B-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![License](https://img.shields.io/badge/License-Apache--2.0-yellow)

Echo is an innovative large language model designed with **temporal episodic memory** capabilities, enabling it to understand and respond with awareness of when conversations occur. Unlike traditional language models, Echo incorporates time-sensitive contextual understanding that enhances its ability to provide temporally-aware responses.

## 🌟 Key Features

- **⏰ Temporal Awareness**: Understands and incorporates time information in conversations
- **🧠 Episodic Memory**: Maintains memory of past interactions with temporal context
- **🔄 Time-Sensitive Responses**: Generates contextually appropriate responses based on temporal information
- **🚀 API Support**: FastAPI-based REST API compatible with OpenAI API format
- **🖥️ Web Interface**: User-friendly Gradio-based web interface
- **📊 Comprehensive Evaluation**: Extensive benchmarks comparing with GPT-3.5, GPT-4, ChatGLM3, and Llama3

## 🏗️ Architecture

Echo extends the ChatGLM architecture with novel temporal encoding mechanisms:

- **Time-Aware Tokenization**: Injects temporal information into the input processing pipeline
- **Episodic Memory Module**: Stores and retrieves conversation history with temporal tags
- **Streaming Support**: Real-time response generation with temporal context
- **Memory Persistence**: Automatic saving and loading of conversation memory

## 📦 Installation

### Prerequisites

```bash
# Python 3.8+
pip install torch transformers
pip install fastapi uvicorn gradio
pip install sentence-transformers
pip install sse-starlette
```

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd echo
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download model weights**:
```bash
# Set your model path
export MODEL_PATH="/path/to/echo/model"
```

## 🚀 Usage

### Web Interface

Launch the Gradio web interface:

```bash
python echo_web.py
```

The interface will be available at `http://localhost:8502`

### API Server

Start the FastAPI server:

```bash
cd stream_api
python echo_api.py
```

The API will be available at `http://localhost:8501`

### API Usage Example

```python
import requests

data = {
    "model": "echo",
    "messages": [
        {"role": "user", "content": "What day is today?"}
    ],
    "timeinfo": "2024年1月15日星期一10点30分00秒",
    "temperature": 0.8,
    "top_p": 0.8,
    "max_tokens": 1000,
    "stream": True
}

response = requests.post("http://localhost:8501/v1/chat/completions", json=data)
```

## 🧪 Experiments

The repository includes comprehensive evaluation experiments:

### Experiment 1: Basic Temporal Memory
- **Location**: `exp1/`
- **Purpose**: Evaluate basic temporal understanding and memory retention
- **Metrics**: Response accuracy, temporal consistency

### Experiment 2: Long-term vs Short-term Memory
- **Location**: `exp2/`  
- **Purpose**: Test memory performance across different time spans
- **Metrics**: Memory retention over time, response quality

### Experiment 3: Difficulty-based Evaluation
- **Location**: `exp3/`
- **Purpose**: Assess performance on varying difficulty levels
- **Metrics**: Accuracy on easy vs hard temporal reasoning tasks

### Human Evaluation
- **Location**: `exp1/human_eval.py`
- **Purpose**: Human-in-the-loop evaluation of model responses
- **Interface**: Interactive evaluation system

### Running Experiments

```bash
# Run Experiment 1
cd exp1
python experiment_1.py

# Run Experiment 2  
cd exp2
python experiment_2.py

# Run Experiment 3
cd exp3
python experiment_3.py
```

## 📊 Performance

Echo demonstrates superior performance in temporal reasoning tasks compared to baseline models:

- **Temporal Accuracy**: 85%+ on time-sensitive queries
- **Memory Consistency**: Maintained across conversation sessions
- **Response Quality**: Competitive with GPT-3.5/4 while adding temporal awareness

## 🤗 Model Weights

### Available Models

| Model | Size | HuggingFace Link |
|-------|------|------------------|
| Echo1-7B | 7B parameters | [ALmonster/Echo1-7B](https://huggingface.co/ALmonster/Echo1-7B) |
| Echo1-72B | 72B parameters | [ALmonster/Echo1-72B](https://huggingface.co/ALmonster/Echo1-72B) |

### Training Data

The training dataset is available at: [ALmonster/Echo-v1](https://huggingface.co/datasets/ALmonster/Echo-v1)

## 📝 Citation

If you use Echo in your research, please cite our paper:

```bibtex
@misc{liu2025echolargelanguagemodel,
      title={Echo: A Large Language Model with Temporal Episodic Memory}, 
      author={WenTao Liu and Ruohua Zhang and Aimin Zhou and Feng Gao and JiaLi Liu},
      year={2025},
      eprint={2502.16090},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.16090}, 
}
```

## 📁 Project Structure

```
echo/
├── echo_api.py              # Legacy API implementation
├── echo_web.py              # Gradio web interface
├── stream_api/              # FastAPI streaming implementation
│   ├── echo_api.py         # Main API server
│   ├── utils.py            # Utility functions
│   └── test_api.py         # API testing
├── exp1/                    # Experiment 1: Basic temporal memory
├── exp2/                    # Experiment 2: Long/short-term memory  
├── exp3/                    # Experiment 3: Difficulty evaluation
├── test_data/              # Evaluation datasets
│   ├── exp1.json
│   ├── exp2.json
│   └── exp3.json
└── README.md               # This file
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built upon the ChatGLM architecture
- Evaluation framework inspired by temporal reasoning benchmarks
- Thanks to the community for feedback and contributions

---

For more information, visit our [paper](https://arxiv.org/abs/2502.16090) or check out the [models on HuggingFace](https://huggingface.co/ALmonster).
