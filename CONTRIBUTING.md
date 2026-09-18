# Contributing to AudioLens

Thank you for your interest in contributing to AudioLens! This document provides guidelines and instructions to help you get started with contributing to this project.

AudioLens extends [ECHO](https://github.com/AnasSAV/ECHO) under its MIT license — see [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setting Up the Development Environment](#setting-up-the-development-environment)
- [Development Workflow](#development-workflow)
  - [Branching Strategy](#branching-strategy)
  - [Making Changes](#making-changes)
  - [Testing](#testing)
  - [Code Style and Linting](#code-style-and-linting)
- [Pull Request Process](#pull-request-process)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- **Docker Desktop** (Windows/Mac) or Docker Engine + Compose plugin (Linux), Compose 2.24 or later — this runs the whole stack
- **Frontend development** (optional, for native UI work):
  - Node.js (v18 or higher)
  - npm or bun package manager
- **Backend development** (optional, for native API/worker work):
  - Python 3.11

See the [README](README.md#prerequisites) for GPU-specific requirements (NVIDIA/AMD/Mac).

### Setting Up the Development Environment

The whole stack — API, Celery workers, scheduler, Redis, and frontend — runs from
the root `docker-compose.yml`. There is no separate Redis-only compose file.

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/AudioLens.git
   cd AudioLens
   ```

2. **Copy the env files** (defaults work out of the box):
   ```bash
   cp Backend/.env.example Backend/.env
   cp Frontend/.env.example Frontend/.env
   ```

3. **Boot the stack**:
   ```bash
   docker compose up --build
   ```

   Or use the helper scripts, which build, start, and wait until both the API and
   the UI answer:
   ```bash
   ./scripts/start.sh   # start and wait for readiness
   ./scripts/stop.sh    # stop, preserving volumes and cached models
   ```

   - **Frontend**: http://localhost:8080
   - **API**: http://localhost:8000/health

   See the [README](README.md#gpu-modes) for GPU profiles and worker scaling.

4. **Optional — native frontend dev server** (hot reload against the Dockerised API):
   ```bash
   cd Frontend
   npm install
   npm run dev
   ```

5. **Optional — native backend** (for debugging the API or a worker directly):
   ```bash
   cd Backend
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Unix or MacOS
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

   Redis must still be running for this to work — start it with
   `docker compose up -d redis`. Inference runs in Celery workers, not in the API
   process, so a worker also has to be running to execute jobs.

### Useful scripts

| Script | Purpose |
|--------|---------|
| `scripts/start.sh` | Build and start the stack, wait for API and UI readiness |
| `scripts/stop.sh` | Stop the stack, preserving Docker volumes and local data |
| `scripts/queue-status.sh` | Inspect the current Celery job queues |
| `scripts/init-custom-datasets.sh` | Initialise custom dataset directories |

## Development Workflow

### Branching Strategy

- `main` is the primary branch and should always be stable
- Create feature branches from `main` using the following naming convention:
  - `feature/short-description` for new features
  - `bugfix/issue-number` for bug fixes
  - `docs/description` for documentation changes
  - `refactor/description` for code refactoring

### Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them with descriptive messages:
   ```bash
   git add .
   git commit -m "Add detailed description of changes"
   ```

3. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

### Testing

- **Frontend**: We use Vitest with Testing Library for React components. Run tests with:
  ```bash
  cd Frontend
  npm run test           # single run
  npm run test:watch     # watch mode
  npm run test:coverage  # with coverage report
  ```

- **Backend**: We use pytest for testing API endpoints, services, and workers. Run tests with:
  ```bash
  cd Backend
  python -m pip install -r requirements-dev.txt
  pytest
  ```

### Code Style and Linting

- **Frontend**: We use ESLint with TypeScript configuration. Run linting with:
  ```bash
  cd Frontend
  npm run lint
  ```

- **Backend**: We follow PEP 8 guidelines. Consider using tools like `flake8` or `black` for formatting.

## Pull Request Process

1. Ensure your code follows our style guidelines and passes all tests
2. Update documentation if necessary
3. Submit a pull request to the `main` branch with a clear title and description
4. Reference any related issues in your PR description using the keyword "Fixes #issue_number"
5. Wait for code review and address any requested changes
6. After approval, a maintainer will merge your PR

## Documentation

- Update the README.md if you're changing functionality or adding features
- [PROJECT.md](PROJECT.md) is the full code reference — update it when you add a service, route, or config knob
- [ARCHITECTURE.md](ARCHITECTURE.md) covers the runtime split between the FastAPI control plane and the Celery execution plane — update it if you change queues, storage, or job flow
- Add comments to your code, especially for complex logic

## Issue Reporting

When reporting issues, please include:

- A clear and descriptive title
- A detailed description of the issue
- Steps to reproduce the problem
- Expected behavior and actual behavior
- Screenshots if applicable
- Environment details (OS, browser, versions, etc.)

## Feature Requests

We welcome feature requests! Please provide:

- A clear and detailed description of the feature
- The motivation and use cases for the feature
- Any potential implementation ideas you might have
- Mockups or examples if applicable

Thank you for contributing to AudioLens!