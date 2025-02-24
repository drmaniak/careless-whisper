# Careless Whisper 🎙️

A toolkit for fine-tuning OpenAI's Whisper model for speech diarization, creating searchable audio transcript datasets from YouTube videos and podcasts.

## Overview

Careless Whisper is an end-to-end solution that combines the power of OpenAI's Whisper model with custom fine-tuning for speaker diarization. The project enables users to:

- Fine-tune Whisper for improved speaker diarization
- Process YouTube videos and podcasts for speaker identification
- Create searchable datasets from transcribed and diarized content
- Query and explore audio content through a user-friendly interface

## Features

- 🔊 **Enhanced Whisper Model**: Fine-tuned version of OpenAI's Whisper optimized for speaker diarization
- 📥 **Media Processing**: Support for YouTube videos and podcast downloads
- 👥 **Speaker Diarization**: Accurate speaker identification and separation
- 🔍 **Search Engine**: Full-text search capabilities across transcribed content
- 🌐 **Web Interface**: User-friendly frontend for exploring the dataset
- 🔄 **API Support**: RESTful API for integration with other services

## Project Structure

careless-whisper/
├── careless_whisper/ # Main package directory
├── data/ # Dataset storage
├── models/ # Fine-tuned model checkpoints
├── notebooks/ # Development and analysis notebooks
└── scripts/ # Utility scripts

## Requirements

- Python ≥ 3.12
- PyTorch
- Transformers
- Datasets\[audio\]
- FastAPI
- Streamlit
- Additional dependencies listed in pyproject.toml

## Installation

First, ensure you have `uv` installed. If not, you can install it using:

```bash
# Install uv using pip
pip install uv

# Or using curl (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then proceed with the installation:

```bash
# Clone the repository
git clone https://github.com/drmaniak/careless-whisper
cd careless-whisper

# Create and activate a virtual environment using uv
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
# .venv\Scripts\activate  # On Windows

# Install dependencies using uv
uv pip install -e .

# Verify installation
python -c "import careless_whisper; print(careless_whisper.__version__)"
```

## Usage

[Coming Soon] This section will be updated with detailed usage instructions as the project develops.

## Development Roadmap

- [ ] Initial project setup and dependency configuration
- [ ] Data collection and preprocessing pipeline
- [ ] Whisper model fine-tuning implementation
- [ ] Speaker diarization integration
- [ ] Search engine development
- [ ] Web interface creation
- [ ] API development
- [ ] Documentation and examples

## Authors

- Manick Vennimalai
- Tolu Ojo
- Gianfranco Ameri
- Cameron B

## Acknowledgments

- OpenAI for the Whisper model
- [Additional acknowledgments to be added as the project evolves]
