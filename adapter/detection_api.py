# detection_api.py
import os, cv2, requests
from typing import List, Dict
# adapter/detection_api.py 里，放在 build_detector 之上 -----------------
from typing import Dict, Any, List, Optional

class LocalYoloDetector:
    """
    本地 YOLO 推理器：alias -> weights.pt 映射
    依赖: pip install ultralytics
    """
    def __init__(self, alias2weights: Dict[str, str], device: str = "cpu", imgsz: int = 640):
        try:
            from ultralytics import YOLO  # 延迟导入，避免模块级失败
        except Exception as e:
            raise RuntimeError("需要本地 YOLO，请先 `pip install ultralytics`") from e
        self._YOLO = YOLO
        self.alias2weights = alias2weights
        self.device = device
        self.imgsz = imgsz
        self._models: Dict[str, Any] = {}  # 缓存不同权重对应的模型

    def _get_model(self, alias: Optional[str]):
        key = self.alias2weights.get(alias or "", None)
        if key is None:
            key = self.alias2weights.get("yolo11n")  # 兜底
        if key not in self._models:
            self._models[key] = self._YOLO(key)
        return self._models[key]

    def detect(self, frame, conf: float = 0.25, iou: float = 0.45, alias: Optional[str] = None) -> List[Dict[str, Any]]:
        if frame is None:
            return []
        model = self._get_model(alias)
        res = model.predict(
            frame, imgsz=self.imgsz, conf=conf, iou=iou,
            device=self.device, verbose=False
        )
        outs: List[Dict[str, Any]] = []
        if not res:
            return outs
        r0 = res[0]
        names = r0.names  # id->name
        if getattr(r0, "boxes", None) is None:
            return outs
        for b in r0.boxes:
            cls_id = int(b.cls[0])
            name   = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            score  = float(b.conf[0])
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            outs.append({"bbox":[x1,y1,x2,y2], "cls":cls_id, "name":name, "score":score})
        return outs
# -----------------------------------------------------------------------
class DetectorAPI:
    def detect(self, frame, conf: float = 0.25, iou: float = 0.45): raise NotImplementedError

class DummyDetector(DetectorAPI):
    def detect(self, frame, conf: float = 0.25, iou: float = 0.45): return []

class UltralyticsAPIDetector:
    def __init__(self, api_key: str, model_map: Dict[str, str], default_alias: str = "yolo11n",
                 url: str = "https://predict.ultralytics.com"):
        self.api_key, self.model_map, self.default_alias, self.url = api_key, model_map, default_alias, url

    def detect(self, frame, conf=0.25, iou=0.45, alias: str = None) -> List[Dict]:
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok: return []
        model_url = self.model_map.get(alias or self.default_alias, self.model_map[self.default_alias])
        r = requests.post(self.url,
                          headers={"x-api-key": self.api_key},
                          data={"model": model_url, "imgsz": 640, "conf": conf, "iou": iou},
                          files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                          timeout=30)
        r.raise_for_status()
        js = r.json()
        outs = []
        for img in js.get("images", []):
            for res in img.get("results", []):
                b = res.get("box", {})
                outs.append({"bbox": [b.get("x1",0), b.get("y1",0), b.get("x2",0), b.get("y2",0)],
                             "cls": res.get("class", -1),
                             "score": res.get("confidence", 0.0),
                             "name": res.get("name", "")})
        return outs

'''def build_detector(cfg):
    ultra = cfg["ultra"]
    return UltralyticsAPIDetector(api_key=ultra["api_key"],
                                  model_map=ultra["models"],
                                  default_alias=ultra.get("default","yolo11n"))'''

# adapter/detection_api.py 片段

def build_detector(cfg):
    # 优先走本地权重
    w = cfg.get("weights") or cfg.get("detector", {}).get("local", {}).get("weights")
    if w:
        alias2weights = {
            "yolo11n": w, "yolo11m": w, "yolo11x": w,
            "yolo12m": w,            # 兼容 default_model=yolo12m
            "yolo11n-pose": w,
        }
        device = cfg.get("detector", {}).get("local", {}).get("device", "cpu")
        imgsz  = int(cfg.get("detector", {}).get("local", {}).get("imgsz", 640))
        return LocalYoloDetector(alias2weights=alias2weights, device=device, imgsz=imgsz)

    # 否则回退到云端 API（保持你原来的构造参数名）
    ultra = cfg["ultra"]
    return UltralyticsAPIDetector(
        api_key=ultra["api_key"],
        url=ultra.get("url", "https://predict.ultralytics.com"),
        alias2model=ultra["models"],
    )
# adapter/detection_api.py
from typing import List, Dict, Optional

def yolo_to_boxes_api(frame, detector, yolo_prompt: Optional[str], min_conf: float = 0.5) -> List[Dict]:
    """
    调用云端 YOLO（detector.detect），并按类别提示与阈值过滤，返回统一检测结构。
    detector.detect 返回的每个元素应包含: {"bbox":[x1,y1,x2,y2], "cls":int, "score":float, "name":str}
    """
    raw = detector.detect(frame, conf=max(min_conf, 0.25), iou=0.45)
    outs: List[Dict] = []
    for d in raw or []:
        name  = d.get("name", "")
        score = float(d.get("score", 0.0))
        if yolo_prompt and name != yolo_prompt:
            continue
        if score < min_conf:
            continue
        x1, y1, x2, y2 = d["bbox"]
        outs.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": score,
            "class_id": int(d.get("cls", -1)),
            "class_name": name,
        })
    return outs