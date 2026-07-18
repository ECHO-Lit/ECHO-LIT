<p align="center">
  <img width="1920" height="1080" alt="ECHO" src="https://github.com/user-attachments/assets/9db0b7eb-a701-4f25-aa27-ec950762bd57" />
</p>

# ECHO - Explainable Computation for Hearing Outputs

<p align="center">
  <a href="https://github.com/AnasSAV/ECHO">
    <img src="https://img.shields.io/badge/version-v1.0-blue" alt="Version"/>
  </a>
  <a href="https://github.com/AnasSAV/ECHO/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/AnasSAV/ECHO" alt="License"/>
  </a>
  <a href="https://github.com/AnasSAV/ECHO/stargazers">
    <img src="https://img.shields.io/github/stars/AnasSAV/ECHO" alt="Stars"/>
  </a>
  <a href="https://github.com/AnasSAV/ECHO/network/members">
    <img src="https://img.shields.io/github/forks/AnasSAV/ECHO" alt="Forks"/>
  </a>
  <a href="https://github.com/AnasSAV/ECHO/issues">
    <img src="https://img.shields.io/github/issues/AnasSAV/ECHO" alt="Issues"/>
  </a>
</p>

> **Learning Interpretability Tool for Audio Models**

Interpreting how deep learning models make decisions is crucial, especially in high-stakes applications like speech recognition, emotion detection, and speaker identification. While the Learning Interpretability Tool (LIT) enables exploration of text and tabular models, there's a lack of equivalent tools for voice-based models. Voice data poses additional challenges due to its temporal nature and multi-modal representations (e.g., waveform, spectrogram).

ECHO extends the interpretability paradigm to audio models, providing researchers and developers with tools to analyze and debug speech models with greater transparency. Through interactive visualizations, attention mechanisms, and perturbation analyses, you can gain deeper insights into how your audio models make decisions.

## Features

* **Audio Data Management**: Upload and manage audio datasets with metadata
* **Waveform Visualization**: Interactive waveform viewer with playback controls
* **Model Prediction Analysis**: Examine model predictions and confidence scores
* **Attention Visualization**: Explore attention patterns in transformer-based audio models
* **Embedding Analysis**: Visualize high-dimensional audio embeddings in 2D/3D space
* **Saliency Mapping**: Identify important regions in audio input using gradient-based methods
* **Perturbation Tools**: Apply various audio perturbations to test model robustness
* **Interactive Dashboard**: Comprehensive interface for exploring model behavior

## Tech Stack

* **Frontend**: React 18 + TypeScript + Vite
* **UI Framework**: Tailwind CSS + shadcn/ui components
* **State Management**: TanStack Query
* **Data Visualization**: Custom React components with Chart.js integration
* **Audio Processing**: Web Audio API
* **Backend**: FastAPI + Python 3.11
* **Models**: Transformer-based audio models (Whisper, Wav2Vec2)
* **Execution**: Celery workers with Redis as the durable, non-evicting broker
* **Storage**: Shared filesystem locally and S3-compatible object storage in production

For S3 deployments, apply the included 24-hour lifecycle policy:
`aws s3api put-bucket-lifecycle-configuration --bucket <bucket> --lifecycle-configuration file://Backend/s3-lifecycle.json`.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) or Docker Engine + Compose plugin (Linux)
- **Windows**: enable the WSL 2 backend in Docker Desktop settings
- **NVIDIA users**: driver 555+, then verify with `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`
- **AMD users**: Linux with a ROCm-supported GPU and host driver
- **Mac GPU users**: run the backend natively; Docker Desktop does not expose MPS

## Quickstart (Docker)

```bash
# 1. Clone
git clone https://github.com/AnasSAV/ECHO.git
cd ECHO

# 2. Copy env files (edit if needed — defaults work out of the box)
cp Backend/.env.example Backend/.env
cp Frontend/.env.example Frontend/.env

# 3. Boot the lightweight stack (no model libraries or model downloads)
docker compose --profile mock up --build
```

The default `mock` execution profile starts a CPU API control plane and a
deterministic worker built from the API dependencies. It exercises the real
Redis queues, job states, storage, caching, cancellation, and result contracts
without installing PyTorch or downloading model weights.

- **Frontend**: http://localhost:8080
- **API**: http://localhost:8000/health
- **Redis**: localhost:6379

### Execution profiles

| Profile | Worker | Use case |
| --- | --- | --- |
| `mock` | Lightweight Docker worker | Everyday UI/API development |
| `mps` | Native macOS PyTorch worker | Real models on Apple Metal |
| `cpu` | Docker model worker | Real-model CPU fallback |
| `cloud-gpu` | CUDA/ROCm worker | Production only |

Mock results are isolated from real results in the analysis cache. Task
envelopes also carry their execution profile, and a worker rejects tasks from a
different profile.

#### Apple Metal (MPS)

Docker Desktop on macOS cannot pass the Metal GPU into a Linux container. Keep
the lightweight control plane in Compose, then run one native worker for the
CPU and model queues:

```bash
# One-time native worker setup
cd Backend
python3.12 -m venv .venv-mps
.venv-mps/bin/pip install -r requirements-worker.txt
cd ..

# Start Docker services configured to dispatch real model jobs
EXECUTION_PROFILE=mps docker compose up --build -d \
  redis api scheduler frontend

# Foreground native worker; Ctrl-C stops model execution
./scripts/run-mps-worker.sh
```

The launcher verifies that `torch.backends.mps.is_available()` is true, forces
`ML_DEVICE=mps`, uses Celery's single-process `solo` pool, and retains at most
one model-registry entry by default. Unsupported MPS operations may fall back
to CPU through `PYTORCH_ENABLE_MPS_FALLBACK=1`.

Switch back to the model-free stack with:

```bash
docker compose down
docker compose --profile mock up --build
```

#### CPU and production GPU modes

```bash
# Real-model CPU fallback
EXECUTION_PROFILE=cpu docker compose --profile real --profile cpu-model up --build

# NVIDIA production worker (Linux/WSL 2 with NVIDIA Container Toolkit)
ENVIRONMENT=production EXECUTION_PROFILE=cloud-gpu ALLOW_PAID_EXECUTION=true \
  docker compose --profile real --profile gpu up --build \
  redis api scheduler frontend worker-cpu worker-gpu

# AMD ROCm production worker
ENVIRONMENT=production EXECUTION_PROFILE=cloud-gpu ALLOW_PAID_EXECUTION=true \
  docker compose --profile real --profile amd up --build \
  redis api scheduler frontend worker-cpu worker-amd
```

`ML_DEVICE=auto` selects NVIDIA CUDA or AMD ROCm first, Apple MPS second, and
CPU as a fallback. Set `ML_DEVICE=cpu`, `mps`, `nvidia`, `amd`, or `cuda:1` to
override selection. For a native AMD install, install the ROCm build of
`torch` and `torchaudio` from the version-matched PyTorch ROCm wheel index
before installing `requirements.txt`.

The API `/health` response reports Redis, storage, and worker-heartbeat state.
Accelerator selection and model loading happen only in worker processes.

### Common operations

```bash
# Stop everything
docker compose down

# Reset all volumes (clears Redis, HF model cache, uploads)
docker compose down -v

# Rebuild after changing requirements
docker compose build --no-cache api worker-mock worker-cpu worker-model-local

# Pre-warm the HF model cache without starting the full stack
docker compose run --rm worker-model-local python3 -c \
  "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-base')"

# Run backend tests in a local virtual environment
cd Backend
python3 -m pip install -r requirements-dev.txt
pytest
```

### Windows tips

- Use the **WSL 2 backend** in Docker Desktop (Settings → General) for faster
  bind-mount I/O and GPU passthrough.
- Keep the repo at a short path (`C:\dev\ECHO`) to avoid `MAX_PATH` issues.
- If Vite HMR stops firing, confirm `CHOKIDAR_USEPOLLING=true` is set in
  `docker-compose.yml` (it already is by default).
- Dataset paths inside the Linux container are **case-sensitive**:
  `data/common_voice_valid_dev` and `data/ravdess_subset` must match exactly.

### Access the Application
Open your browser and navigate to [http://localhost:8080](http://localhost:8080)


## Project Structure

```
ECHO/
├── Frontend/                # React frontend application
│   ├── components/          # React components
│   │   ├── analysis/        # Analysis and perturbation tools
│   │   ├── audio/           # Audio visualization components
│   │   ├── layout/          # Layout components
│   │   ├── panels/          # Dashboard panels
│   │   ├── ui/              # Reusable UI components
│   │   └── visualization/   # Data visualization components
│   ├── hooks/               # Custom React hooks
│   ├── lib/                 # Utility functions
│   └── pages/               # Page components
│
├── Backend/                 # FastAPI backend application
│   ├── app/                 # Application code
│   │   ├── api/             # API routes and endpoints
│   │   ├── core/            # Core functionality
│   │   └── services/        # Business logic services
│   ├── data/                # Sample datasets
│   ├── tests/               # Backend tests
│   └── uploads/             # User-uploaded audio files
│
├── CODE_OF_CONDUCT.md       # Community guidelines
├── CONTRIBUTING.md          # Contribution guidelines
├── LICENSE                  # MIT License
├── README.md                # Project documentation
└── SECURITY.md              # Security policy
```

## Available Scripts

### Frontend
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run lint` - Run ESLint
- `npm run preview` - Preview production build

### Backend
- `pytest` - Run backend tests
- `uvicorn app.main:app --reload` - Start the API server in development mode

## Usage

1. **Upload Audio Data**: Use the audio uploader to load your audio files
2. **Select Models**: Choose from available audio models for analysis
3. **Explore Visualizations**:
   - Examine waveforms and spectrograms
   - View model predictions and confidence scores
   - Explore attention patterns and embedding spaces
   - Generate saliency maps to highlight important audio regions
4. **Apply Perturbations**: Test model robustness with various audio perturbations
5. **Analyze Results**: Use the interactive dashboard to gain insights

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## Security

For security-related issues, please refer to our [Security Policy](SECURITY.md).

## Authors

- **Anas Hussaindeen** - [GitHub Profile](https://github.com/AnasSAV)
- **Chandupa Ambepitiya** - [GitHub Profile](https://github.com/Chand2103)
- **Dewmike Amarasinghe** - [GitHub Profile](https://github.com/DewmikeAmarasinghe)

## Mentor
- **Dr Uthayasanker Thayasivam** - NLP Researcher & Senior Lecturer and Head of Department at Computer Science & Engineering, University of Moratuwa, Sri Lanka

## Acknowledgments

- Inspired by Google's [Learning Interpretability Tool (LIT)](https://github.com/PAIR-code/lit)
- Built with modern React ecosystem and TypeScript
- Special thanks to the open-source community for the amazing tools and libraries

## Roadmap

- [ ] Backend API enhancements for model serving
- [ ] Support for more audio model architectures
- [ ] Advanced perturbation techniques
- [ ] Real-time audio processing capabilities
- [ ] Export functionality for visualizations
- [ ] Multi-language support
- [ ] Plugin system for custom analysis tools

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built for audio model interpretability</sub>
</p>
