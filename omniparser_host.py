import io
import base64
import random
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

import torch
from util.utils import (
    get_som_labeled_img,
    check_ocr_box,
    get_caption_model_processor,
    get_yolo_model,
)

CONFIG_PATH = "config.yaml"


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config(CONFIG_PATH)

PORT = int(cfg["server"]["port"])
HOST = str(cfg["server"]["host"])
SEED = int(cfg["server"]["seed"])

MODEL_PATH = str(cfg["models"]["yolo_weights"])
CAPTION_MODEL_PATH = str(cfg["models"]["caption_weights"])
DEVICE = str(cfg["models"]["device"]).lower()

BOX_THRESHOLD = float(cfg["infer"]["box_threshold"])
IOU_THRESHOLD = float(cfg["infer"]["iou_threshold"])
BATCH_SIZE = int(cfg["infer"]["batch_size"])

OCR_ENABLED = bool(cfg["ocr"]["enabled"])
OCR_BACKEND = str(cfg["ocr"]["backend"]).lower()
TEXT_THRESHOLD = float(cfg["ocr"]["text_threshold"])

_rng = random.Random(SEED)
torch.manual_seed(SEED)

som_model = get_yolo_model(MODEL_PATH)
try:
    som_model.to(DEVICE)
except Exception:
    pass

caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path=CAPTION_MODEL_PATH,
    device=DEVICE,
)

app = FastAPI(title="OmniParser Host API", version="0.6.0")


class InferRequest(BaseModel):
    image_base64: str = Field(...)
    explore: bool = Field(False)
    ocr: Optional[bool] = Field(None)


def center_ratio_xyxy(b: List[float]) -> List[float]:
    x1, y1, x2, y2 = b
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def normalize_element(e: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    bbox = e.get("bbox")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None

    bbox_ratio = [float(x) for x in bbox]

    return {
        "index": index,
        "type": e.get("type"),
        "bbox_ratio": bbox_ratio,
        "center_ratio": center_ratio_xyxy(bbox_ratio),
        "interactivity": bool(e.get("interactivity")),
        "content": e.get("content"),
        "source": e.get("source"),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "seed": SEED,
    }


@app.post("/infer")
def infer(req: InferRequest):
    try:
        img_bytes = base64.b64decode(req.image_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    ocr_enabled_this = OCR_ENABLED if req.ocr is None else bool(req.ocr)

    ocr_text: List[str] = []
    ocr_bbox: Optional[List[List[int]]] = None

    if ocr_enabled_this:
        use_paddleocr = OCR_BACKEND == "paddleocr"
        (ocr_text, ocr_bbox), _ = check_ocr_box(
            img,
            display_img=False,
            output_bb_format="xyxy",
            goal_filtering=None,
            easyocr_args={"paragraph": False, "text_threshold": TEXT_THRESHOLD},
            use_paddleocr=use_paddleocr,
        )

    with torch.inference_mode():
        overlay_img_b64, _, elements = get_som_labeled_img(
            img,
            som_model,
            BOX_TRESHOLD=BOX_THRESHOLD,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            caption_model_processor=caption_model_processor,
            ocr_text=ocr_text,
            use_local_semantics=True,
            iou_threshold=IOU_THRESHOLD,
            scale_img=False,
            batch_size=BATCH_SIZE,
        )

    clickable_raw = [e for e in elements if bool(e.get("interactivity"))]

    clickable: List[Dict[str, Any]] = []
    for idx, e in enumerate(clickable_raw):
        ne = normalize_element(e, idx)
        if ne is not None:
            clickable.append(ne)

    if not clickable:
        raise HTTPException(status_code=404, detail="No clickable elements found")

    if req.explore:
        chosen = _rng.choice(clickable)
        return JSONResponse(
            {
                "action": "click",
                "text": None,
                "element": chosen,
                "overlay_image_base64": overlay_img_b64,
            }
        )

    return JSONResponse(
        {
            "elements": clickable,
            "overlay_image_base64": overlay_img_b64,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("omniparser_host:app", host=HOST, port=PORT, workers=1)
