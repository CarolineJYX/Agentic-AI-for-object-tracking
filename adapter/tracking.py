from typing import Dict, Any, List
from typing import List, Dict, Tuple
import numpy as np
import cv2
from tracker.byte_tracker import BYTETracker

class TrackerArgs:
    def __init__(self):
        self.track_thresh = 0.4
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = False
        self.frame_rate = 30

def is_scene_cut(prev_bgr, cur_bgr, hist_thr=0.5, luma_thr=35):
    """返回是否发生镜头切换（True/False）"""
    if prev_bgr is None or cur_bgr is None:
        return False
    # HSV 直方图相似度
    prev_hsv = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2HSV)
    cur_hsv  = cv2.cvtColor(cur_bgr,  cv2.COLOR_BGR2HSV)
    hist_prev = cv2.calcHist([prev_hsv], [0,1], None, [50,60], [0,180,0,256])
    hist_cur  = cv2.calcHist([cur_hsv],  [0,1], None, [50,60], [0,180,0,256])
    cv2.normalize(hist_prev, hist_prev)
    cv2.normalize(hist_cur,  hist_cur)
    corr = cv2.compareHist(hist_prev, hist_cur, cv2.HISTCMP_CORREL)
    # 亮度差
    prev_y = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2YCrCb)[:,:,0]
    cur_y  = cv2.cvtColor(cur_bgr,  cv2.COLOR_BGR2YCrCb)[:,:,0]
    luma_diff = float(np.mean(np.abs(prev_y.astype(np.float32) - cur_y.astype(np.float32))))
    # 规则：相关性低 / 亮度变化大 即认为切
    return (corr < hist_thr) or (luma_diff > luma_thr)

def bytetrack_to_id(
    frame: np.ndarray,
    detections: List[Dict],
    tracker: BYTETracker,
    frame_id: int,
) -> List[Dict]:
    """
    Send YOLO detections to ByteTrack for ID assignment, with scene-cut aware ID offset.
    On scene cut, start a new ID range by adding a persistent offset to raw IDs.
    """
    # ---- static state inside the function (persist across calls) ----
    if not hasattr(bytetrack_to_id, "_prev_frame"):
        bytetrack_to_id._prev_frame = None
    if not hasattr(bytetrack_to_id, "_tid_offset"):
        bytetrack_to_id._tid_offset = 0
    if not hasattr(bytetrack_to_id, "_max_raw_tid_in_segment"):
        bytetrack_to_id._max_raw_tid_in_segment = 0
    if not hasattr(bytetrack_to_id, "_reset_on_cut"):
        # 如果你的 BYTETracker 支持 reset()，保持 True；否则设为 False 也可正常工作（仅做偏移）
        bytetrack_to_id._reset_on_cut = True

    print(f"[BYTE] Entering ByteTrack at frame {frame_id}, {len(detections)} detections in.")

    # ---- scene cut detection & segment switch ----
    if is_scene_cut(bytetrack_to_id._prev_frame, frame):
        print(f"[SCENE] cut at frame {frame_id}. Start a new segment.")
        # 将新的段落编号起点叠加上一个段落中出现过的最大原始ID
        bytetrack_to_id._tid_offset += max(1, bytetrack_to_id._max_raw_tid_in_segment)
        bytetrack_to_id._max_raw_tid_in_segment = 0
        # 可选：如果 tracker 有 reset() 方法，调用以彻底断开轨迹延续
        if bytetrack_to_id._reset_on_cut and hasattr(tracker, "reset") and callable(getattr(tracker, "reset")):
            try:
                tracker.reset()
                print("[SCENE] tracker.reset() called.")
            except Exception as e:
                print(f"[SCENE] tracker.reset() failed: {e}")

    if not detections:
        print("[BYTE] No detections, skipping this frame.")
        bytetrack_to_id._prev_frame = frame
        return []

    # ---- build detections for ByteTrack: [x1, y1, x2, y2, score] ----
    H, W = frame.shape[:2]
    def _clip_xyxy(b):
        x1, y1, x2, y2 = map(float, b)
        x1 = max(0.0, min(x1, W - 1.0))
        y1 = max(0.0, min(y1, H - 1.0))
        x2 = max(0.0, min(x2, W - 1.0))
        y2 = max(0.0, min(y2, H - 1.0))
        return [x1, y1, x2, y2]

    dets_np = np.array([
        [*_clip_xyxy(det["bbox"]), float(det.get("confidence", 1.0))]
        for det in detections
    ], dtype=np.float32)

    # 可选：调试打印
    # print(f"[BYTE] dets_np shape={dets_np.shape}, mean_score={dets_np[:,4].mean():.3f}")

    img_info = (H, W)
    img_size = (H, W)

    # ---- run tracker ----
    online_targets = tracker.update(dets_np, img_info, img_size)
    print(f"[BYTE] tracker.update() returned {len(online_targets)} online targets.")

    results: List[Dict] = []
    for i, t in enumerate(online_targets):
        x, y, w, h = t.tlwh  # tlwh floats
        x1, y1, x2, y2 = x, y, x + w, y + h
        x1, y1, x2, y2 = _clip_xyxy([x1, y1, x2, y2])

        raw_tid = int(t.track_id)  # tracker 内部的本段原始ID
        # 对外暴露的跨段 ID：原始ID + 段偏移
        out_tid = int(bytetrack_to_id._tid_offset + raw_tid)

        # 维护本段内出现的最大原始ID，用于下一次切段时推进偏移
        if raw_tid > bytetrack_to_id._max_raw_tid_in_segment:
            bytetrack_to_id._max_raw_tid_in_segment = raw_tid

        print(f"[BYTE] Tracked {i}: rawID={raw_tid} -> ID={out_tid}, "
              f"bbox=({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")

        results.append({
            "track_id": out_tid,           # 对外使用的ID（含段偏移）
            "bbox": [x1, y1, x2, y2],
            "raw_track_id": raw_tid,       # 保留原始ID便于调试（可选）
        })

    if not results:
        print("[BYTE] No valid tracked objects produced.")

    # 更新上一帧缓存
    bytetrack_to_id._prev_frame = frame
    return results

def _build_tracker(args_dict: Dict[str, Any]) -> BYTETracker:
    """根据 bt_cfg 构建/更新 BYTETracker 实例（带简单缓存）"""
    cfg = TrackerArgs()
    cfg.track_thresh       = float(args_dict.get("track_thresh", 0.4))
    cfg.match_thresh       = float(args_dict.get("match_thresh", 0.8))
    cfg.track_buffer       = int(args_dict.get("track_buffer", 30))
    cfg.aspect_ratio_thresh= float(args_dict.get("aspect_ratio_thresh", 1.6))
    cfg.min_box_area       = int(args_dict.get("min_box_area", 10))
    cfg.mot20              = bool(args_dict.get("mot20", False))
    cfg.frame_rate         = int(args_dict.get("frame_rate", 30))

    # 简单缓存：bt_cfg 变化才重建 tracker
    key = (cfg.track_thresh, cfg.match_thresh, cfg.track_buffer,
           cfg.aspect_ratio_thresh, cfg.min_box_area, cfg.mot20, cfg.frame_rate)
    if getattr(_build_tracker, "_cache_key", None) != key:
        _build_tracker._tracker = BYTETracker(cfg)
        _build_tracker._cache_key = key
    return _build_tracker._tracker  # type: ignore[attr-defined]

def run_bytetrack(frame, detections: List[Dict], bt_cfg: Dict[str, Any], frame_idx: int) -> Dict[str, Any]:
    """
    将检测框送入 ByteTrack，返回统一结构：
      {
        "tracks": [{"track_id": int, "bbox": [x1,y1,x2,y2], "raw_track_id": int}, ...],
        "id_switch": int,   # 简单估计：本帧新出现的 track 数
        "frame_ids": [frame_idx] 或 []
      }
    """
    tracker = _build_tracker(bt_cfg)

    # 空帧/空检测：仅维护 prev_ids 与上一帧缓存，保持接口一致
    if frame is None or frame_idx is None:
        return {"tracks": [], "id_switch": 0, "frame_ids": []}
    if not detections:
        # 更新上一帧缓存（供切镜判定），但不产出轨迹
        if not hasattr(run_bytetrack, "_prev_ids"):
            run_bytetrack._prev_ids = set()  # type: ignore[attr-defined]
        run_bytetrack._prev_ids = set()      # 本帧无对象
        # 仍让 bytetrack_to_id 更新内部 prev_frame
        _ = bytetrack_to_id(frame, [], tracker, frame_idx)
        return {"tracks": [], "id_switch": 0, "frame_ids": []}

    # 正常：调用你已有的场景切换 + ID 偏移逻辑
    tracks = bytetrack_to_id(frame, detections, tracker, frame_idx)

    # 简单的 “id_switch” 估算：与上一帧 ID 集合差的新增数（出生数）
    new_ids = {t["track_id"] for t in tracks}
    prev_ids = getattr(run_bytetrack, "_prev_ids", set())  # type: ignore[attr-defined]
    id_switch = len(new_ids - prev_ids)
    run_bytetrack._prev_ids = new_ids  # type: ignore[attr-defined]

    return {
        "tracks": tracks,
        "id_switch": int(id_switch),
        "frame_ids": [frame_idx] if len(tracks) > 0 else [],
    }