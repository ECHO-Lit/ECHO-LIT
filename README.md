<p align="center">
  <img width="1920" height="1080" alt="AudioLens" src="https://raw.githubusercontent.com/ECHO-Lit/AudioLens-LandingPage/refs/heads/main/public/assets/AudioLens.png" />
</p>

# AudioLens

<p align="center">
  <img src="https://img.shields.io/badge/version-v1.0-blue" alt="Version"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License"/>
</p>

> **Learning Interpretability Tool for Audio Models**

Interpreting how deep learning models make decisions is crucial, especially in high-stakes applications like speech recognition, emotion detection, and speaker identification. While the Learning Interpretability Tool (LIT) enables exploration of text and tabular models, there's a lack of equivalent tools for voice-based models. Voice data poses additional challenges due to its temporal nature and multi-modal representations (e.g., waveform, spectrogram).

AudioLens extends the interpretability paradigm to audio models, providing researchers and developers with tools to analyze and debug speech models with greater transparency. Through interactive visualizations, attention mechanisms, and perturbation analyses, you can gain deeper insights into how your audio models make decisions.

AudioLens is built on and extends [ECHO](https://github.com/AnasSAV/ECHO), an MIT-licensed audio interpretability tool created by Anas Hussaindeen, Chandupa Ambepitiya, and Dewmike Amarasinghe. See [Authors](#authors) and [Contributors](#contributors) below.

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

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) or Docker Engine + Compose plugin (Linux); Docker Compose 2.24 or later
- **Windows**: enable the WSL 2 backend in Docker Desktop settings
- **NVIDIA users**: driver 555+, then verify with `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`
- **AMD users**: Linux with a ROCm-supported GPU and host driver
- **Mac GPU users**: run the backend natively; Docker Desktop does not expose MPS

## Quickstart (Docker)

```bash
# 1. Clone (fork of ECHO, cloned into an AudioLens directory)
git clone https://github.com/AnasSAV/ECHO.git AudioLens
cd AudioLens

# 2. Copy env files (edit if needed — defaults work out of the box)
cp Backend/.env.example Backend/.env
cp Frontend/.env.example Frontend/.env

# 3. Boot the full stack
docker compose up --build
```

First boot starts a CPU API control plane and a separate local worker. Model
weights are downloaded only by the worker into `hf-cache`; the API image does
not contain or import the ML runtime.

`Backend/.env` configures the API, the scheduler and every worker: session and
job TTLs, upload limits, cookie flags, CORS origins (`ALLOWED_ORIGINS`) and the
worker tunables. Every key is documented in `Backend/.env.example`, including a
production block. Compose itself pins only what depends on the container
network and mounts (the Redis URLs and the storage root), so those lines stay
commented. `Frontend/.env` is optional: with `VITE_API_BASE_URL` unset, the
client calls port 8000 on whatever host served the page.

To add workers, for throughput or so that queued work continues when one fails,
scale them: `docker compose up -d --scale worker-cpu=2 --scale worker-model-local=2`.
Redis refuses writes rather than evicting data once it reaches `REDIS_MAXMEMORY`
(default `1gb`; set it in your shell or a root `.env`).

- **Frontend**: http://localhost:8080
- **API**: http://localhost:8000/health
- **Redis**: localhost:6379

### GPU modes

```bash
# NVIDIA (Linux or WSL 2 with NVIDIA Container Toolkit). Disable the local
# all-queue worker so only the GPU worker consumes GPU queues.
docker compose --profile gpu up --build --scale worker-model-local=0 redis api scheduler frontend worker-cpu worker-gpu

# AMD ROCm (Linux with a supported ROCm host driver)
docker compose --profile amd up --build --scale worker-model-local=0 redis api scheduler frontend worker-cpu worker-amd
```

Docker Desktop on macOS cannot pass the Metal GPU into a Linux container. Keep
the API in Compose, without the local CPU model worker, and run the GPU-queue
worker natively to use MPS:

```bash
docker compose up -d --build --scale worker-model-local=0 redis api scheduler frontend worker-cpu
cd Backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
STORAGE_LOCAL_ROOT=shared-storage ML_DEVICE=mps \
  celery -A app.core.celery_app:celery_app worker \
  --queues=gpu-fast,gpu-large --concurrency=1 --prefetch-multiplier=1
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
docker compose build --no-cache api worker-cpu worker-model-local

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
- Keep the repo at a short path (`C:\dev\AudioLens`) to avoid `MAX_PATH` issues.
- If Vite HMR stops firing, confirm `CHOKIDAR_USEPOLLING=true` is set in
  `docker-compose.yml` (it already is by default).
- Dataset paths inside the Linux container are **case-sensitive**:
  `data/common_voice_valid_dev` and `data/ravdess_subset` must match exactly.

### Access the Application
Open your browser and navigate to [http://localhost:8080](http://localhost:8080)


## Project Structure

```
AudioLens/
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

Original creators of ECHO, the project AudioLens is built on:

- **Anas Hussaindeen** - [GitHub Profile](https://github.com/AnasSAV)
- **Chandupa Ambepitiya** - [GitHub Profile](https://github.com/Chand2103)
- **Dewmike Amarasinghe** - [GitHub Profile](https://github.com/DewmikeAmarasinghe)

## Contributors

AudioLens is developed and maintained by Group 16, University of Moratuwa, extending ECHO's codebase and interpretability paradigm (full list in [CONTRIBUTORS.md](CONTRIBUTORS.md)):

- **Januda Lelwala**
- **Janith Mahanama**
- **Hesandi Mallawarachchi**

## Mentor
- **Dr Uthayasanker Thayasivam** - NLP Researcher & Senior Lecturer and Head of Department at Computer Science & Engineering, University of Moratuwa, Sri Lanka

## Acknowledgments

- Built on [ECHO](https://github.com/AnasSAV/ECHO) by Anas Hussaindeen, Chandupa Ambepitiya, and Dewmike Amarasinghe, used and extended here under its MIT license
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
  <sub>AudioLens — built for audio model interpretability, extending ECHO</sub>
</p>
