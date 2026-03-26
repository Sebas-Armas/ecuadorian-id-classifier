# Document Classifier API — Ecuadorian Documents

Flask microservice that accepts an image and returns the document type plus confidence score.

**Response contract:**
```json
{
  "tipo_documento": "cedula | pasaporte | desconocido",
  "confianza": 0.95
}
```

---

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the Server](#running-the-server)
- [API Reference](#api-reference)
  - [POST /predict](#post-predict)
  - [GET /health](#get-health)
- [Web UI](#web-ui)
- [Integration Examples](#integration-examples)
- [Error Reference](#error-reference)

---

## Project Structure

```
app/
├── app.py          ← Flask app + inference logic
├── templates/
│   └── index.html  ← Drag-and-drop web UI
└── README.md
```

The app resolves shared resources from the project root automatically:

```
ecuadorian-id-classifier/   ← project root (two levels up from app/)
├── global/
│   ├── config.py           ← loaded at startup
│   └── utils.py
└── best_model.pth          ← model checkpoint
```

---

## Prerequisites

Complete the [root-level setup](../README.md#setup) first so the `ecuadorian-id` conda environment exists and dependencies are installed.

---

## Setup

```bash
# Activate the project environment (if not already active)
conda activate ecuadorian-id
```

No additional setup is required — the app resolves all paths relative to the repo root at startup.

---

## Running the Server

From the **project root** or from inside `app/`:

```bash
# From project root (recommended)
conda activate ecuadorian-id
python app/app.py

# Or from inside app/
cd app
python app.py
```

The server starts at `http://localhost:5000` in debug mode (`debug=True`, bound to `0.0.0.0:5000`).

> **Production note:** Replace `app.run(debug=True, ...)` with a production WSGI server such as **Gunicorn**:
> ```bash
> pip install gunicorn
> gunicorn -w 2 -b 0.0.0.0:5000 app:app
> ```

---

## API Reference

### POST /predict

Classifies a single document image.

| Property | Value |
|---|---|
| URL | `POST /predict` |
| Content-Type | `multipart/form-data` |
| Field name | `image` |
| Allowed formats | `jpg`, `jpeg`, `png` |
| Max file size | 10 MB |
| Min image size | 50 × 50 px |

#### Success response — `200 OK`

```json
{
  "tipo_documento": "cedula",
  "confianza": 0.9731
}
```

#### Debug response — `200 OK` (append `?debug=1`)

```json
{
  "tipo_documento": "cedula",
  "confianza": 0.9731,
  "debug": {
    "cedula_prob":      0.9731,
    "pasaporte_prob":   0.0214,
    "desconocido_prob": 0.0055,
    "threshold":        0.7
  }
}
```

#### cURL

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/cedula.jpg"
```

```bash
# With debug info
curl -X POST "http://localhost:5000/predict?debug=1" \
  -F "image=@/path/to/doc.jpg"
```

#### Python (`requests`)

```python
import requests

with open("cedula.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:5000/predict",
        files={"image": f},
    )

print(response.json())
# {"tipo_documento": "cedula", "confianza": 0.9731}
```

---

### GET /health

Liveness check — verifies the server is up and reports the active compute device.

```bash
curl http://localhost:5000/health
```

```json
{"status": "ok", "device": "cpu"}
```

---

## Web UI

Open `http://localhost:5000` in your browser.

- **Drag & drop** an image (or click to select) and press **Clasificar**.
- Enable the **debug** checkbox to display per-class probabilities alongside the result.

---

## Integration Examples

### PHP

```php
$ch = curl_init('http://localhost:5000/predict');
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, [
    'image' => new CURLFile('/path/to/doc.jpg'),
]);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$result = json_decode(curl_exec($ch), true);

echo $result['tipo_documento']; // "cedula"
echo $result['confianza'];      // 0.9731
```

### .NET Core (C#)

```csharp
using var form = new MultipartFormDataContent();
form.Add(new StreamContent(File.OpenRead("doc.jpg")), "image", "doc.jpg");
var response = await httpClient.PostAsync("http://localhost:5000/predict", form);
var result   = await response.Content.ReadFromJsonAsync<JsonObject>();
// result["tipo_documento"], result["confianza"]
```

---

## Error Reference

| HTTP Status | Cause |
|---|---|
| `400` | Missing `image` field or empty filename |
| `413` | File exceeds 10 MB |
| `415` | Unsupported file extension (not jpg/png) |
| `422` | Corrupt image or image smaller than 50 × 50 px |
| `500` | Internal inference error |