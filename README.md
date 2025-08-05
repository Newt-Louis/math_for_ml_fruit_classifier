# 🍍 Fruit Classifier Backend

A minimal **FastAPI** backend project for fruit classification. This backend is ready to integrate with mobile or web apps and includes a health check route (`/ping`). Project dependencies are managed using [Poetry](https://python-poetry.org/).

---

## 🚀 Features

- ✅ FastAPI + Uvicorn for modern async API
- ✅ `/ping` route for health check
- ✅ CORS enabled (ready for mobile/web frontend)
- ✅ Poetry dependency management (no virtualenv mess)
- ✅ Clean, lightweight starter structure

---

## 🚀 Setup and Installation
Follow these steps to get your development environment set up.

## 🧰 Requirements

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)

---

## ⚙️ Setup Instructions

### 2. Install Poetry (if not yet installed)
Poetry is used to manage project dependencies. If you don't have it installed, follow the instructions for your operating system.
#### macOS / Linux
```
curl -sSL [https://install.python-poetry.org](https://install.python-poetry.org) | python3 -
```
#### Windows (in PowerShell)
```
(Invoke-WebRequest -Uri [https://install.python-poetry.org](https://install.python-poetry.org) -UseBasicParsing).Content | py -
```

### Install Project Dependencies
With Poetry installed, run the following command in the project root directory. This will create a virtual environment and install all the required libraries specified in the pyproject.toml and poetry.lock files.

```
poetry install
```

#### macOS / Linux
```
cp .env.example .env
```
   
#### Windows
```
copy .env.example .env
```

---

### 3. Activate virtual environment (optional)

```bash
poetry env list --full-path    # Copy path to venv
source /path/to/venv/bin/activate
```

Or install shell plugin if needed:

```bash
poetry self add poetry-plugin-shell
poetry shell
```

---

### 4. Run the FastAPI server

```bash
poetry run uvicorn main:app --reload
```

Or if using a subpackage:

```bash
poetry run uvicorn fruit_classifier_backend.main:app --reload
```

Then visit: [http://localhost:8000/ping](http://localhost:8000/ping)

---

## 🧪 Test Endpoint

| Method | Path  | Description                                 |
| ------ | ----- | ------------------------------------------- |
| GET    | /ping | Health check. Returns `{"message": "pong"}` |

---

## 📘 Next Steps

* [ ] Add `/classify` endpoint to receive image and return prediction
* [ ] Integrate ML model using `torch`, `onnx`, or `tensorflow`
* [ ] Add authentication or rate-limiting if needed