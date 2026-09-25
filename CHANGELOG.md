# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Naming history:** this project was released as *LIT for Voice* (v1.0.0), renamed
> to *ECHO*, and is now developed as **AudioLens**, an extension of
> [ECHO](https://github.com/AnasSAV/ECHO). Entries below v1.0.0 describe the
> project under its earlier names and are kept as historical record.

## [Unreleased] — AudioLens

Extends ECHO with an asynchronous execution plane and a broader set of
interpretability analyses.

### Added
- **Dataset fetch scripts** - `download_ravdess.py`, `download_saa.py`,
  `prepare_l2arctic.py`, `prepare_common_voice.py` and a `fetch_datasets.py`
  wrapper, driven by selection manifests in `scripts/manifests/`. Datasets are
  fetched from their official sources into `Backend/data/` instead of being
  described as bundled; no dataset audio is redistributed
- **Asynchronous job execution** - Celery workers as a separate model-execution
  plane, with `gpu-fast`, `gpu-large`, and `cpu` queues; the FastAPI control plane
  no longer imports the ML runtime
- **Job lifecycle API** - job submission, status polling, and authorized result
  retrieval, with session-scoped ownership checks
- **Object storage abstraction** - shared filesystem locally, S3-compatible
  storage in production, with a 24-hour lifecycle policy for transient objects
- **Redis topology** - separate logical databases for sessions/cache, job
  metadata, Celery broker, and results, configured persistent and non-evicting
- **Scheduled cleanup** - Celery Beat task expiring transient local objects
- **Dataset exploratory data analysis (EDA)** for built-in and custom datasets
- **Embedding analytics** - clustering and nearest-neighbour semantic retrieval
- **Saliency faithfulness evaluation** alongside saliency generation
- **Linguistic versus acoustic influence analysis**
- **Whisper hallucination detection**
- **Layer-wise representation analysis (probing)**
- **Jacobian Lens** - decoder-only, position-resolved lens over Whisper decoder
  layers (see [JACOBIAN_LENS.md](JACOBIAN_LENS.md))
- **Accent and language fairness analysis**
- **Internal word activation explorer**
- **Custom model ingestion** from Hugging Face, with compatibility constraints
- **Model and dataset comparison**
- **LibriSpeech-1000 and SAVEE dataset support** with import helper scripts
- **Developer scripts** - `scripts/start.sh`, `scripts/stop.sh`, and
  `scripts/queue-status.sh`
- **GPU profiles in Compose** - optional CUDA and ROCm workers, plus native MPS
  worker instructions for macOS

### Changed
- **Docker Compose now runs the full stack** (API, workers, scheduler, Redis,
  frontend) from the repository root, replacing the Redis-only setup
- **Documentation** - added [ARCHITECTURE.md](ARCHITECTURE.md),
  [PROJECT.md](PROJECT.md), and [FEATURES.md](FEATURES.md)

## [1.0.0] - 2024-10-13

### Added
- **Initial stable release of LIT for Voice** - A comprehensive Learning Interpretability Tool for Audio Models
- **Frontend Application** (React 18 + TypeScript + Vite)
  - Interactive audio waveform visualization with playback controls
  - Model prediction analysis dashboard
  - Attention pattern visualization for transformer-based audio models
  - High-dimensional audio embedding visualization in 2D/3D space
  - Gradient-based saliency mapping for audio inputs
  - Comprehensive perturbation tools for model robustness testing
  - Responsive UI built with Tailwind CSS and shadcn/ui components
  - Audio file upload and management system
  
- **Backend API** (FastAPI + Python 3.11)
  - RESTful API for audio processing and model inference
  - Support for transformer-based audio models (Whisper, Wav2Vec2)
  - Redis caching for predictions and analysis results
  - Audio perturbation service with various transformation techniques
  - Custom dataset handling and management
  - Session-based file management
  - Comprehensive test suite with pytest
  
- **Core Features**
  - Audio data management with metadata support
  - Interactive waveform viewer with zoom and navigation
  - Model prediction analysis with confidence scores
  - Attention mechanism visualization
  - Embedding space exploration tools
  - Saliency map generation for interpretability
  - Multiple audio perturbation techniques
  - Real-time model inference and analysis
  
- **Infrastructure**
  - Docker Compose setup for Redis
  - Comprehensive development environment setup
  - CI/CD ready project structure
  - Detailed documentation and setup guides
  
- **Documentation**
  - Comprehensive README with installation instructions
  - Contributing guidelines for developers
  - Code of conduct for community participation
  - Security policy for responsible disclosure
  - Project structure documentation
  
### Technical Stack
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui, TanStack Query
- **Backend**: FastAPI, Python 3.11, Redis, PyTorch, Transformers, Librosa
- **Audio Processing**: Web Audio API, Librosa, SoundFile
- **Visualization**: Custom React components, Chart.js integration
- **Development**: ESLint, Prettier, pytest, Docker

### Supported Models
- OpenAI Whisper (speech recognition)
- Facebook Wav2Vec2 (speech representation learning)
- Custom transformer-based audio models

### Sample Datasets
- Common Voice validation subset
- RAVDESS emotion recognition subset
- Support for custom audio dataset uploads

---

**Full Changelog**: https://github.com/AnasSAV/LIT-for-Voice/commits/v1.0.0