from typing import TypedDict, Any, Dict, List, Tuple

class AgentState(TypedDict, total=False):
    # 输入
    prompt: str
    frame_idx: int
    frame: Any

    # 运行状态
    model_in_use: str
    cooldown_left: int
    missing_gap: int
    id_switch: int
    stable: bool

    # 适配器/工具（★关键：必须声明）
    detector: Any             # 或更严格：DetectorAPI
    # 可选：如果你还注入其它工具，也一并声明，如：tracker、clipper 等

    # 策略（AI Planner 写入）
    policy: Dict[str, Any]    # 统一让各节点从这里读阈值/配置

    # 中间产物
    detections: Any
    det_found: bool
    bt_cfg: Dict[str, float]
    tracks: Any
    frame_ids: List[int]
    avg_conf: float
    top10_conf: float
    margin: float
    clip_thresh: float
    max_bridge: int

    # 结果
    merged_segments: List[Tuple[int, int]]
    gid_frames: Dict[int, List[int]]
    gid_scores: Dict[int, List[float]]
    exported_video_path: str
    export_error: str
    
    # 输入参数
    video_path: str
    output_dir: str
    fps: float
    total_frames: int
    
    # CLIP模型缓存
    clip_model: Any
    clip_preprocess: Any
    clip_tokenizer: Any
    clip_device: str

    # 标志/错误
    need_human_review: bool
    error: str