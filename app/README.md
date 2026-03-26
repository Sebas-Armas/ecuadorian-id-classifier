# Document Classifier API — Ecuadorian Documents

Flask service that accepts an image and returns the document type + confidence score,
exactly as specified:

```json
{
  "tipo_documento": "cedula | pasaporte | desconocido",
  "confianza": 0.95
}
```

---

## Project structure

```
doc_classifier/app
                ├── app.py               ← Flask app + inference logic
                ├── requirements.txt
                │   └── index.html       ← Web UI (drag & drop)
├── templates/
└── README.md
```

The app expects your project's `global/` folder (with `config.py` and `utils.py`)
two levels above this directory — the same convention used in the notebooks:

```
project_root/
├── global/
│   ├── config.py
│   └── utils.py
├── best_model.pth
└── notebooks/
    └── doc_classifier/   ← this folder lives here
        └── app.py
```

---

## Setup

In Root Project:
```bash
pip install -r requirements.txt
```

---

## Run

In app folder:
```bash
python app.py
```

Server starts at `http://localhost:5000`.

---

## Usage

### Web UI
Open `http://localhost:5000` in your browser — drag & drop an image, click **Clasificar**.
Enable the **debug** checkbox to see per-class probabilities.

### REST API

**Endpoint:** `POST /predict`
**Content-Type:** `multipart/form-data`
**Field:** `image` (jpg / jpeg / png, max 10 MB)

#### cURL
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/cedula.jpg"
```

#### Python
```python
import requests

with open("cedula.jpg", "rb") as f:
    r = requests.post("http://localhost:5000/predict", files={"image": f})
print(r.json())
# {"tipo_documento": "cedula", "confianza": 0.9731}
```

#### With debug info
```bash
curl -X POST "http://localhost:5000/predict?debug=1" \
  -F "image=@/path/to/doc.jpg"
```
Returns the spec output plus per-class probabilities:
```json
{
  "tipo_documento": "cedula",
  "confianza": 0.9731,
  "debug": {
    "cedula_prob": 0.9731,
    "pasaporte_prob": 0.0214,
    "desconocido_prob": 0.0055,
    "threshold": 0.6
  }
}
```

#### Health check
```bash
curl http://localhost:5000/health
# {"status": "ok", "device": "cpu"}
```

---

## Notes on PHP / .NET Core integration

Since the role uses PHP and .NET Core, this Flask service acts as a **microservice**.
You can call it from either stack over HTTP:

**PHP**
```php
$ch = curl_init('http://localhost:5000/predict');
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, ['image' => new CURLFile('/path/to/doc.jpg')]);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = json_decode(curl_exec($ch), true);
// $response['tipo_documento'], $response['confianza']
```

**.NET Core (C#)**
```csharp
using var form = new MultipartFormDataContent();
form.Add(new StreamContent(File.OpenRead("doc.jpg")), "image", "doc.jpg");
var response = await httpClient.PostAsync("http://localhost:5000/predict", form);
var result = await response.Content.ReadFromJsonAsync<JsonObject>();
```