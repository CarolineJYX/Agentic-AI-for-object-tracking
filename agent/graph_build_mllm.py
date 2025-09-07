from langgraph.graph import StateGraph, START, END
from agent.state_types import AgentState
from adapter.nlp_parser import nlp_json_deepseek
from adapter.detection_api import DetectorAPI, yolo_to_boxes_api 
from adapter.tracking import run_bytetrack
from adapter.clip_matcher import clip_stats
from adapter.editing import run_video, merge_frame_ranges, export_best_match_video
from adapter.global_id import update_stats, is_ready, get_or_assign_gid_with_reuse, get_assigned_gid
from multimodal_advisor import MultiModalAdvisor
import torch
from pathlib import Path

def build_mllm_app():
    """构建集成MLLM参数决策的应用"""
    g = StateGraph(dict)
    
    # ---------- 节点函数 ----------
    def n_parse(s):
        """NLP解析节点"""
        if s.get("parsed"):
            return s
        p = nlp_json_deepseek(s.get("user_input") or s.get("prompt", ""))
        s["parsed"] = p
        s["prompt"] = p.get("clip_prompt", s.get("prompt", ""))
        s["det_class_hint"] = p.get("yolo_class")
        s["need_human_review"] = bool(p.get("parse_error"))
        print("Parsed:", p)
        print(f"clip_prompt={s['prompt']}  yolo_class={s.get('det_class_hint')}")
        return s

    def n_mllm_advisor(s):
        """MLLM参数顾问节点 - 统一决定所有参数"""
        # 只在第一帧执行MLLM分析，避免重复调用
        if s.get("mllm_analyzed", False):
            return s
        
        video_path = s.get("video_path", "")
        prompt = s.get("prompt", "")
        
        print(f"🤖 [MLLM] 开始视频分析和参数决策...")
        print(f"   视频: {video_path}")
        print(f"   目标: {prompt}")
        
        if not Path(video_path).exists():
            print(f"   ⚠️ 视频文件不存在，使用默认参数")
            # 使用默认参数
            s.update({
                "model_in_use": "yolo11n",
                "clip_thresh": 0.25,
                "max_bridge": 30,
                "gap": 2,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "track_thresh": 0.3,
                "mllm_confidence": 0.0,
                "mllm_reasoning": "视频文件不存在，使用默认参数",
                "mllm_analyzed": True
            })
            return s
        
        try:
            # 初始化MLLM顾问
            API_KEY = "sk-ZKv84hrnoNgqB8bXy6qrIWga9ak4zHI1M7RcByXCLdRcoG1o"
            BASE_URL = "https://api.xinyun.ai/v1"
            advisor = MultiModalAdvisor(API_KEY, BASE_URL)
            
            # 执行视频分析
            result = advisor.analyze_video(video_path, prompt)
            llm_analysis = result.get('llm_analysis', {})
            
            if "error" in llm_analysis:
                print(f"   ❌ MLLM分析失败: {llm_analysis['error']}")
                # 使用默认参数
                s.update({
                    "model_in_use": "yolo11n",
                    "clip_thresh": 0.25,
                    "max_bridge": 30,
                    "gap": 2,
                    "track_buffer": 30,
                    "match_thresh": 0.8,
                    "track_thresh": 0.3,
                    "mllm_confidence": 0.0,
                    "mllm_reasoning": f"MLLM分析失败: {llm_analysis.get('error', 'Unknown error')}"
                })
                return s
            
            # 提取MLLM推荐的参数
            if "recommended_params" in llm_analysis:
                params = llm_analysis["recommended_params"]
                analysis = llm_analysis.get("analysis", {})
                confidence = llm_analysis.get("confidence", 0.0)
                
                # YOLO模型选择逻辑（基于场景复杂度）
                video_stats = result.get('video_stats', {})
                complexity = result.get('complexity_metrics', {})
                
                # 根据视频特征选择YOLO模型
                motion_intensity = complexity.get('motion_intensity', 0.0)
                background_complexity = complexity.get('background_complexity', 0.0)
                duration = video_stats.get('duration', 0.0)
                
                # YOLO模型选择策略
                if motion_intensity > 0.3 or background_complexity > 0.4 or duration > 60:
                    yolo_model = "yolo11m"  # 复杂场景使用中等模型
                    if motion_intensity > 0.5 or background_complexity > 0.6:
                        yolo_model = "yolo11l"  # 非常复杂场景使用大模型
                else:
                    yolo_model = "yolo11n"  # 简单场景使用轻量模型
                
                # 更新状态
                s.update({
                    "model_in_use": yolo_model,
                    "clip_thresh": float(params.get("clip_thresh", 0.25)),
                    "max_bridge": int(params.get("max_bridge", 30)),
                    "gap": int(params.get("gap", 2)),
                    "track_buffer": int(params.get("track_buffer", 30)),
                    "match_thresh": float(params.get("match_thresh", 0.8)),
                    "track_thresh": float(params.get("track_thresh", 0.3)),
                    "mllm_confidence": float(confidence),
                    "mllm_reasoning": analysis.get("parameter_reasoning", "MLLM参数推荐"),
                    "mllm_analysis": analysis,
                    "mllm_analyzed": True
                })
                
                print(f"   ✅ MLLM参数决策完成:")
                print(f"      YOLO模型: {yolo_model}")
                print(f"      CLIP阈值: {params.get('clip_thresh', 0.25)}")
                print(f"      桥接参数: {params.get('max_bridge', 30)}")
                print(f"      跟踪缓冲: {params.get('track_buffer', 30)}")
                print(f"      置信度: {confidence}")
                
            else:
                print(f"   ⚠️ MLLM未返回参数建议，使用默认值")
                s.update({
                    "model_in_use": "yolo11n",
                    "clip_thresh": 0.25,
                    "max_bridge": 30,
                    "gap": 2,
                    "track_buffer": 30,
                    "match_thresh": 0.8,
                    "track_thresh": 0.3,
                    "mllm_confidence": 0.0,
                    "mllm_reasoning": "MLLM未返回有效参数建议",
                    "mllm_analyzed": True
                })
                
        except Exception as e:
            print(f"   ❌ MLLM节点异常: {str(e)}")
            # 使用默认参数
            s.update({
                "model_in_use": "yolo11n",
                "clip_thresh": 0.25,
                "max_bridge": 30,
                "gap": 2,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "track_thresh": 0.3,
                "mllm_confidence": 0.0,
                "mllm_reasoning": f"MLLM节点异常: {str(e)}",
                "mllm_analyzed": True
            })
        
        return s

    def n_detect(s):
        """YOLO检测节点"""
        frame_idx = s.get("frame_idx", 0)
        frame = s.get("frame")
        model_name = s.get("model_in_use", "yolo11n")
        det_class_hint = s.get("det_class_hint")
        
        if frame is None:
            s["detections"] = []
            s["det_found"] = False
            return s
        
        # 使用MLLM决定的模型 - 创建本地检测器
        from adapter.detection_api import LocalYoloDetector
        
        # 简化的模型映射
        alias2weights = {
            "yolo11n": "yolo11n.pt",
            "yolo11m": "yolo11m.pt", 
            "yolo11l": "yolo11l.pt",
            "yolo11x": "yolo11x.pt"
        }
        
        detector = LocalYoloDetector(alias2weights)
        raw_dets = detector.detect(frame, alias=model_name)
        
        # 直接处理raw_dets，因为LocalYoloDetector已经返回了正确格式
        detections = []
        for d in raw_dets or []:
            detections.append({
                "bbox": d.get("bbox", [0, 0, 0, 0]),
                "confidence": float(d.get("score", 0.0)),
                "class_id": int(d.get("cls", -1)),
                "class_name": d.get("name", "")
            })
        s["detections"] = detections
        s["det_found"] = len(detections) > 0
        
        if frame_idx % 10 == 0:
            print(f"API returned {len(detections)} objects for {model_name} (hint={det_class_hint})")
        
        return s

    def n_det_check(s):
        """检测结果检查"""
        if not s.get("det_found", False):
            s["missing_gap"] = s.get("missing_gap", 0) + 1
        else:
            s["missing_gap"] = 0
        return s

    def n_bytetrack(s):
        """ByteTrack跟踪节点"""
        frame_idx = s.get("frame_idx", 0)
        detections = s.get("detections", [])
        
        # 使用MLLM决定的ByteTrack参数
        bt_config = {
            "track_thresh": s.get("track_thresh", 0.3),
            "match_thresh": s.get("match_thresh", 0.8),
            "track_buffer": s.get("track_buffer", 30)
        }
        
        frame = s.get("frame")
        tracks_result = run_bytetrack(frame, detections, bt_config, frame_idx)
        tracks = tracks_result.get("tracks", [])
        s["tracks"] = tracks
        
        if frame_idx % 10 == 0:
            print(f"[BYTE] frame={frame_idx} tracks={len(tracks)} config={bt_config}")
        
        return s

    def n_bt_check(s):
        """ByteTrack结果检查"""
        tracks = s.get("tracks", [])
        prev_tracks = s.get("prev_tracks", [])
        
        # 计算ID切换
        id_switches = 0
        if prev_tracks:
            prev_ids = {t.get("track_id") for t in prev_tracks}
            curr_ids = {t.get("track_id") for t in tracks}
            id_switches = len(prev_ids.symmetric_difference(curr_ids))
        
        s["id_switch"] = id_switches
        s["prev_tracks"] = tracks.copy()
        s["stable"] = id_switches == 0
        
        return s

    def n_clip_apply(s):
        """CLIP语义匹配节点"""
        frame = s.get("frame")
        detections = s.get("detections", [])
        prompt = s.get("prompt", "")
        
        # 使用MLLM决定的CLIP阈值
        clip_thresh = s.get("clip_thresh", 0.25)
        
        # 性能优化：缓存CLIP模型组件
        clip_model = s.get("clip_model")
        clip_preprocess = s.get("clip_preprocess") 
        clip_tokenizer = s.get("clip_tokenizer")
        clip_device = s.get("clip_device")
        
        if frame is not None and detections:
            scores = clip_stats(
                detections, prompt, frame,
                model=clip_model, preprocess=clip_preprocess,
                tokenizer=clip_tokenizer, device=clip_device
            )
            
            # 缓存模型组件（首次加载后）
            if clip_model is None:
                from adapter.clip_matcher import load_clip_model
                s["clip_model"], s["clip_preprocess"], s["clip_tokenizer"], s["clip_device"] = load_clip_model()
        
        s["clip_thresh"] = clip_thresh
        return s

    def n_global_id(s):
        """全局ID管理节点"""
        tracks = s.get("tracks", [])
        dets = s.get("detections", [])
        th = float(s.get("clip_thresh", 0.0))

        def _match_det_for_track(track_bbox, detections):
            best_det, best_iou = None, 0.0
            for det in detections:
                det_bbox = det.get("bbox", [0,0,0,0])
                x1 = max(track_bbox[0], det_bbox[0])
                y1 = max(track_bbox[1], det_bbox[1])
                x2 = min(track_bbox[2], det_bbox[2])
                y2 = min(track_bbox[3], det_bbox[3])
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    track_area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
                    det_area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
                    union = track_area + det_area - intersection
                    iou = intersection / max(union, 1e-6)
                    if iou > best_iou:
                        best_iou = iou
                        best_det = det
            return best_det if best_iou > 0.3 else None

        for t in tracks:
            tid = int(t["track_id"])
            tb = t.get("bbox", [0,0,0,0])
            d = _match_det_for_track(tb, dets)

            passed = False
            if d is not None:
                sc = float(d.get("clip_score", d.get("confidence", d.get("score", 0.0))))
                passed = (sc >= th) if th > 0 else True
            
            update_stats(tid, passed)

            # 全局ID分配逻辑
            gid = get_assigned_gid(tid)
            if gid is None and passed and d is not None and isinstance(d.get("img_feat"), torch.Tensor):
                pidx = abs(hash(s.get("prompt",""))) % 10_000_000
                gid = get_or_assign_gid_with_reuse(tid, d["img_feat"], prompt_idx=pidx, sim_threshold=0.10)
            
            t["global_id"] = int(gid) if gid is not None else None

        if tracks:
            gids = [tr.get("global_id") for tr in tracks if tr.get("global_id") is not None]
            if gids:
                frame_idx = s.get('frame_idx', -1)
                print(f"[GID] frame={frame_idx} assigned GIDs={gids}")

        return s

    def n_edit(s):
        """编辑节点 - 使用MLLM决定的参数"""
        idx = int(s.get("frame_idx", 0))
        tracks = s.get("tracks", [])
        detections = s.get("detections", [])
        gid_frames = s.get("gid_frames", {})
        gid_scores = s.get("gid_scores", {})

        # 累计通过CLIP阈值的tracks
        for t in tracks:
            gid = t.get("global_id")
            if gid is None:
                continue
                
            track_bbox = t.get("bbox", [0,0,0,0])
            best_det = None
            best_iou = 0.0
            
            for det in detections:
                det_bbox = det.get("bbox", [0,0,0,0])
                x1, y1, x2, y2 = max(track_bbox[0], det_bbox[0]), max(track_bbox[1], det_bbox[1]), \
                                min(track_bbox[2], det_bbox[2]), min(track_bbox[3], det_bbox[3])
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    track_area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
                    det_area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
                    union = track_area + det_area - intersection
                    iou = intersection / max(union, 1e-6)
                    if iou > best_iou:
                        best_iou = iou
                        best_det = det
            
            if best_det is not None and best_iou > 0.3:
                clip_score = float(best_det.get("clip_score", 0.0))
                clip_thresh = float(s.get("clip_thresh", 0.0))
                passed = (clip_score >= clip_thresh) if clip_thresh > 0 else True
                
                if passed:
                    gid_frames.setdefault(int(gid), []).append(idx)
                    gid_scores.setdefault(int(gid), []).append(clip_score)
        
        s["gid_frames"] = gid_frames
        s["gid_scores"] = gid_scores

        # 使用MLLM决定的gap和bridge参数进行片段合并
        gap = int(s.get("gap", 2))
        bridge = int(s.get("max_bridge", 30))

        merged = {
            gid: merge_frame_ranges(frames, gap=gap, bridge=bridge)
            for gid, frames in gid_frames.items()
        }
        s["merged_segments"] = merged

        if idx % 10 == 0 and gid_frames:
            sample_gid = next(iter(gid_frames.keys()))
            sample_segments = merged.get(sample_gid, [])
            print(f"[EDIT] frame={idx} sample gid={sample_gid} segments={sample_segments}")

        return s

    def n_export(s):
        """导出节点 - 仅在最后一帧执行"""
        frame_idx = int(s.get("frame_idx", 0))
        total_frames = int(s.get("total_frames", 0))
        
        # 只在最后一帧执行导出
        if frame_idx < total_frames - 1:
            return s
        
        print(f"🎬 [EXPORT] 开始导出最终视频...")
        
        gid_frames = s.get("gid_frames", {})
        gid_scores = s.get("gid_scores", {})
        video_path = s.get("video_path", "")
        output_dir = s.get("output_dir", "./output")
        fps = s.get("fps", 30.0)
        
        # 使用MLLM决定的参数
        gap = int(s.get("gap", 2))
        bridge = int(s.get("max_bridge", 30))
        
        try:
            output_path = export_best_match_video(
                video_path=video_path,
                gid_frames=gid_frames,
                gid_scores=gid_scores,
                output_dir=output_dir,
                fps=fps,
                gap=gap,
                bridge=bridge
            )
            s["exported_video_path"] = output_path
            print(f"[SUCCESS] 视频导出完成: {output_path}")
            
            # 显示MLLM决策信息
            mllm_confidence = s.get("mllm_confidence", 0.0)
            mllm_reasoning = s.get("mllm_reasoning", "")
            print(f"[MLLM] 参数决策置信度: {mllm_confidence}")
            print(f"[MLLM] 决策理由: {mllm_reasoning}")
            
        except Exception as e:
            error_msg = f"视频导出失败: {str(e)}"
            s["export_error"] = error_msg
            print(f"[ERROR] {error_msg}")
        
        return s

    def n_human(s):
        """人工审核节点"""
        s["need_human_review"] = False
        return s

    # ---------- 条件函数 ----------
    def c_det(s):
        """检测条件判断"""
        if s.get("det_found"):
            return "ok"
        # 简化逻辑，直接继续处理
        return "miss"

    def c_bt(s):
        """ByteTrack条件判断"""
        return "ok"  # 简化，总是继续

    def c_clip(s):
        """CLIP条件判断"""
        return "accept"  # 简化，总是接受

    # ---------- 注册节点 ----------
    g.add_node("ParseInstruction", n_parse)
    g.add_node("MLLMAdvisor", n_mllm_advisor)  # 新增MLLM节点
    g.add_node("YOLODetect", n_detect)
    g.add_node("DetectResultCheck", n_det_check)
    g.add_node("ByteTrack", n_bytetrack)
    g.add_node("ByteTrackCheck", n_bt_check)
    g.add_node("CLIPApply", n_clip_apply)
    g.add_node("GlobalID", n_global_id)
    g.add_node("Edit", n_edit)
    g.add_node("Export", n_export)
    g.add_node("HumanReview", n_human)

    # ---------- 连边 ----------
    # 新的流程：Parse -> MLLM -> Detect -> ...
    g.add_edge(START, "ParseInstruction")
    g.add_edge("ParseInstruction", "MLLMAdvisor")  # NLP后直接进入MLLM
    g.add_edge("MLLMAdvisor", "YOLODetect")       # MLLM决策后开始检测
    g.add_edge("YOLODetect", "DetectResultCheck")
    
    g.add_conditional_edges("DetectResultCheck", c_det,
                          {"ok": "ByteTrack", "miss": "ByteTrack"})
    
    g.add_edge("ByteTrack", "ByteTrackCheck")
    g.add_conditional_edges("ByteTrackCheck", c_bt,
                          {"ok": "CLIPApply"})
    
    g.add_edge("CLIPApply", "GlobalID")
    g.add_conditional_edges("GlobalID", c_clip,
                          {"accept": "Edit"})
    
    g.add_edge("Edit", "Export")
    g.add_edge("Export", END)

    return g.compile()

# 为了保持兼容性，也提供原有接口
def build_app():
    """兼容性接口，返回MLLM版本"""
    return build_mllm_app()
