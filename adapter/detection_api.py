# detection_ap
import os, cv2, requests
import time
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

class CloudYoloDetector:
    """
    云端 YOLO 推理器：支持多个云端模型和自动回退
    """
    def __init__(self, api_key: str, model_urls: Dict[str, str], fallback_detector=None):
        self.api_key = api_key
        self.model_urls = model_urls
        self.fallback_detector = fallback_detector
        self.last_successful_model = "yolo11n"  # 记录最后成功的模型
        self.failed_models = set()  # 记录失败的模型
        
    def _test_connection(self, model_alias: str) -> bool:
        """测试特定模型的连接"""
        if model_alias in self.failed_models:
            return False
            
        try:
            url = self.model_urls.get(model_alias)
            if not url:
                return False
                
            # 发送健康检查请求
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _detect_with_model(self, frame, conf: float, iou: float, model_alias: str) -> List[Dict]:
        """使用特定模型进行检测"""
        try:
            # 优化：压缩图像以减少传输时间
            height, width = frame.shape[:2]
            if height > 640 or width > 640:
                # 如果图像太大，先缩放到640x640
                scale = min(640/width, 640/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # 优化：使用更高质量的JPEG压缩，但文件更小
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # 降低质量以减小文件大小
            ok, buf = cv2.imencode(".jpg", frame, encode_param)
            if not ok: 
                return []
                
            url = self.model_urls.get(model_alias)
            if not url:
                raise ValueError(f"Model {model_alias} not found")
            
            # 优化：减少超时时间，快速失败
            response = requests.post(
                url,
                headers={"x-api-key": self.api_key},
                data={"imgsz": 640, "conf": conf, "iou": iou},
                files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                timeout=5  # 减少超时时间
            )
            response.raise_for_status()
            
            js = response.json()
            outs = []
            for img in js.get("images", []):
                for res in img.get("results", []):
                    b = res.get("box", {})
                    outs.append({
                        "bbox": [b.get("x1",0), b.get("y1",0), b.get("x2",0), b.get("y2",0)],
                        "cls": res.get("class", -1),
                        "score": res.get("confidence", 0.0),
                        "name": res.get("name", "")
                    })
            
            # 如果成功，更新最后成功的模型
            self.last_successful_model = model_alias
            return outs
            
        except Exception as e:
            print(f"❌ 云端模型 {model_alias} 检测失败: {e}")
            self.failed_models.add(model_alias)
            return None

    def detect(self, frame, conf=0.25, iou=0.45, alias: str = None) -> List[Dict]:
        """检测函数，支持自动回退"""
        if frame is None:
            return []
            
        # 确定要使用的模型
        target_model = alias or self.last_successful_model
        
        # 首先尝试目标模型
        if target_model in self.model_urls and target_model not in self.failed_models:
            result = self._detect_with_model(frame, conf, iou, target_model)
            if result is not None:
                return result
        
        # 如果目标模型失败，尝试其他可用模型
        for model_alias in self.model_urls:
            if model_alias != target_model and model_alias not in self.failed_models:
                print(f"🔄 尝试回退到模型: {model_alias}")
                result = self._detect_with_model(frame, conf, iou, model_alias)
                if result is not None:
                    return result
        
        # 如果所有云端模型都失败，使用本地回退
        if self.fallback_detector:
            print("🔄 所有云端模型失败，回退到本地模型")
            return self.fallback_detector.detect(frame, conf, iou, alias)
        
        print("❌ 所有检测方法都失败")
        return []

class UltralyticsAPIDetector:
    """保持向后兼容的旧API检测器"""
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

class HybridDetector:
    """
    混合检测器：优先使用本地模型，云端作为备选
    """
    def __init__(self, local_detector, cloud_detector):
        self.local_detector = local_detector
        self.cloud_detector = cloud_detector
        self.use_cloud = False  # 默认使用本地
        self.cloud_failures = 0
        self.max_cloud_failures = 3  # 连续失败3次后禁用云端
        
    def detect(self, frame, conf=0.25, iou=0.45, alias: str = None) -> List[Dict]:
        """检测函数，优先本地，失败时尝试云端"""
        if frame is None:
            return []
        
        # 优先使用本地检测器
        if self.local_detector:
            try:
                results = self.local_detector.detect(frame, conf, iou, alias)
                if results:  # 如果本地检测成功
                    self.cloud_failures = 0  # 重置云端失败计数
                    return results
                else:
                    # 本地检测成功但没有结果，也返回空列表
                    return []
            except Exception as e:
                print(f"⚠️ 本地检测失败: {e}")
        
        # 如果本地失败或没有本地检测器，尝试云端
        if self.cloud_detector and self.cloud_failures < self.max_cloud_failures:
            try:
                print("🌐 尝试云端检测...")
                results = self.cloud_detector.detect(frame, conf, iou, alias)
                if results:
                    self.cloud_failures = 0  # 重置失败计数
                    self.use_cloud = True  # 标记使用云端
                    return results
                else:
                    self.cloud_failures += 1
            except Exception as e:
                print(f"❌ 云端检测失败: {e}")
                self.cloud_failures += 1
        
        return []

def build_detector(cfg):
    """
    构建混合检测器：优先本地，云端备选
    """
    # 云端模型URL配置
    cloud_model_urls = {
        "yolo11n": "https://predict-blqihm1lbrzviglu81xc-7nza6zqsha-ew.a.run.app",
        "yolo11m": "https://predict-fgerkqvofxwg6vttanta-7nza6zqsha-ew.a.run.app", 
        "yolo11x": "https://predict-gjreehgdl2pc13wg7puc-7nza6zqsha-ew.a.run.app"
    }
    
    api_key = "aee3b7c66e472496b54163e97b1f4c2583cf75c567"
    
    # 创建本地检测器
    local_detector = None
    
    # 优先使用local_models配置，回退到weights
    local_models = cfg.get("local_models", {})
    if not local_models:
        # 如果没有local_models配置，使用单个weights
        w = cfg.get("weights")
        if w and os.path.exists(w):
            local_models = {
                "yolo11n": w, "yolo11m": w, "yolo11x": w,
                "yolo12m": w,  # 兼容 default_model=yolo12m
            }
    
    if local_models:
        try:
            # 检查哪些模型文件存在
            available_models = {}
            for alias, model_path in local_models.items():
                if os.path.exists(model_path):
                    available_models[alias] = model_path
                    print(f"✅ 找到本地模型: {alias} -> {model_path}")
                else:
                    print(f"⚠️ 本地模型文件不存在: {alias} -> {model_path}")
            
            if available_models:
                device = cfg.get("detector", {}).get("local", {}).get("device", "cpu")
                imgsz  = int(cfg.get("detector", {}).get("local", {}).get("imgsz", 640))
                local_detector = LocalYoloDetector(alias2weights=available_models, device=device, imgsz=imgsz)
                print(f"✅ 本地检测器创建成功，支持 {len(available_models)} 个模型")
            else:
                print("❌ 没有可用的本地模型文件")
        except Exception as e:
            print(f"⚠️ 本地检测器创建失败: {e}")
    
    # 创建云端检测器
    cloud_detector = CloudYoloDetector(
        api_key=api_key,
        model_urls=cloud_model_urls,
        fallback_detector=None
    )
    
    # 创建混合检测器
    hybrid_detector = HybridDetector(local_detector, cloud_detector)
    
    print("🚀 使用混合检测器：优先本地，云端备选")
    if local_detector:
        print("   ✅ 本地检测器: 已配置")
    else:
        print("   ❌ 本地检测器: 未配置")
    print(f"   🌐 云端检测器: 已配置 ({len(cloud_model_urls)} 个模型)")
    
    return hybrid_detector
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