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
* **Storage**: Redis for caching predictions and results

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) or Docker Engine + Compose plugin (Linux)
- **Windows**: enable the WSL 2 backend in Docker Desktop settings
- **GPU users**: NVIDIA driver 555+, then verify with `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`

## Quickstart (Docker)

```bash
# 1. Clone
git clone https://github.com/AnasSAV/ECHO.git
cd ECHO

# 2. Copy env files (edit if needed — defaults work out of the box)
cp Backend/.env.example Backend/.env
cp Frontend/.env.example Frontend/.env

# 3. Boot the full stack
docker compose up --build
```

First boot downloads Whisper + Wav2Vec2 models (~3.4 GB) into the `hf-cache`
volume. Subsequent boots reuse it — no re-download.

- **Frontend**: http://localhost:8080
- **Backend**: http://localhost:8000/health
- **Redis**: localhost:6379

### GPU mode (NVIDIA opt-in)

```bash
# Start GPU backend instead of CPU backend (avoids port 8000 conflict)
docker compose --profile gpu up redis frontend backend-gpu --build
```

### Custom models

ECHO supports Hugging Face audio models by **task and architecture**, never by
model name. Register one from the toolbar (**Custom Models → Add Model**) with
its repo ID; ECHO derives the task, input tensor, sampling rate, label map and
tokenizer from the repository config alone.

| Task | Loaded with | Analyses |
|---|---|---|
| Sequence-to-sequence ASR | `AutoModelForSpeechSeq2Seq` | transcription, encoder/decoder activations, cross-attention, word projections, embeddings, saliency, activation interventions |
| CTC ASR | `AutoModelForCTC` | transcription, frame-level token probabilities, encoder activations, embeddings, saliency |
| Audio classification | `AutoModelForAudioClassification` | class probabilities, embeddings, saliency, perturbation analysis, label interventions |

CTC models expose no decoder analyses because they contain no text decoder.

Registration validates the repo against every constraint in two phases: config
and processor first (fast, no weights), then a real forward pass on synthetic
audio to confirm the model returns `logits` and honours
`output_hidden_states=True`. Limits (parameter count, size, inference time,
models per session) live in `Backend/app/core/settings.py` under
`CUSTOM_MODEL_*` and can be overridden by environment variable.

> **Downloads:** `docker-compose.yml` sets `HF_HUB_OFFLINE=1`, so only
> already-cached repos can be registered. To add new ones, start the backend
> with `HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0`.

Verified examples, one per task:

```bash
docker compose up -d
# Allow downloads for the registration step
docker compose run --rm -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 backend \
  python3 -c "from transformers import AutoModelForCTC, AutoProcessor; \
    AutoModelForCTC.from_pretrained('facebook/wav2vec2-base-960h'); \
    AutoProcessor.from_pretrained('facebook/wav2vec2-base-960h')"
```

| Repo | Detected as |
|---|---|
| `openai/whisper-base` | sequence-to-sequence ASR |
| `facebook/wav2vec2-base-960h` | CTC ASR |
| `superb/hubert-base-superb-er` | audio classification (4 labels) |

The API is also usable directly:

```bash
# Config-only compatibility check (no weights downloaded)
curl -c cookies.txt -X POST http://localhost:8000/custom-models/validate \
  -H "Content-Type: application/json" \
  -d '{"model_id":"facebook/wav2vec2-base-960h"}'

# Full validation + registration
curl -b cookies.txt -X POST http://localhost:8000/custom-models/register \
  -H "Content-Type: application/json" \
  -d '{"name":"ctc-asr","model_id":"facebook/wav2vec2-base-960h"}'

# Run it through the normal inference route
curl -b cookies.txt -X POST http://localhost:8000/inferences/run \
  -H "Content-Type: application/json" \
  -d '{"model":"custom:<session_id>:ctc-asr","dataset":"common-voice","dataset_file":"sample-000037.mp3"}'
```

Other endpoints: `GET /custom-models/list`, `GET /custom-models/{name}`,
`DELETE /custom-models/{name}`, and `GET /custom-models/capabilities` (the full
support matrix, constraints and limits — the UI renders its documentation from it).

### Common operations

```bash
# Stop everything
docker compose down

# Reset all volumes (clears Redis, HF model cache, uploads)
docker compose down -v

# Rebuild after changing requirements.txt
docker compose build --no-cache backend

# Pre-warm the HF model cache without starting the full stack
docker compose run --rm backend python3 -c \
  "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-base')"

# Run backend tests
docker compose run --rm backend pytest
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
