# OmniParser Host Service

This repository hosts an OmniParser-based UI element inference service.
It exposes a FastAPI HTTP endpoint that accepts an image and returns
clickable UI elements detected on the screen, optionally with an
exploration (random selection) mode.

The service is designed to:

- Load all models once at startup
- Perform fast inference per request
- Run on a GPU cluster or locally
- Be accessed via SSH port forwarding

---

## Features

- YOLO-based UI icon detection
- OCR-assisted text-aware filtering
- Florence-2 based icon semantic captioning
- Deterministic explore mode with fixed seed
- Returns:
  - Clickable elements with normalized bounding boxes
  - Center coordinates
  - Overlay image with bounding boxes
- JSON-based API (no multipart required)

---

## Installation

### Create environment

```bash
conda create -n omni python=3.12 -y
conda activate omni
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

All configuration is centralized in `config.yaml`.

---

## Running the Service

```bash
python omniparser_host.py
```

---

## API

### POST /infer

Request JSON:

```json
{
  "image_base64": "<BASE64_IMAGE>",
  "explore": false,
  "ocr": true
}
```

Response (explore mode):

```json
{
  "action": "click",
  "text": null,
  "element": {
    "index": 0,
    "type": "icon",
    "bbox_ratio": [0.4, 0.1, 0.5, 0.2],
    "center_ratio": [0.45, 0.15],
    "content": "Normal"
  },
  "overlay_image_base64": "<BASE64_PNG>"
}
```

---

## Cluster Access

Use SSH port forwarding:

```bash
ssh -L 8090:<compute-node>:8090 <user>@<login-node>
```