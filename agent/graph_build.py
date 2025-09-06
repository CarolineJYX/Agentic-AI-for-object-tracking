from langgraph.graph import StateGraph
from langgraph.graph import StateGraph, START, END
from agent.state_types import AgentState
from adapter.nlp_parser import nlp_json_deepseek
from agent.policies import select_yolo_model, adapt_clip_threshold, decide_bridge, adapt_bt_config
from agent.planner import plan_routing, plan_tracking, plan_clip, plan_bridge
from adapter.detection_api import DetectorAPI, yolo_to_boxes_api 
from adapter.tracking import run_bytetrack
from adapter.clip_matcher import clip_stats
from adapter.editing import run_video, merge_frame_ranges, export_best_match_video
from adapter.global_id import update_stats, is_ready, get_or_assign_gid_with_reuse, get_or_assign_gid_sticky, get_assigned_gid
import torch

def build_app():
    g = StateGraph(dict)
    # ---------- 节点函数 ----------
    def n_parse(s):
        if s.get("parsed"):
            return s
        p = nlp_json_deepseek(s.get("user_input") or s.get("prompt", ""))
        s["parsed"] = p
        s["prompt"] = p.get("clip_prompt", s.get("prompt", ""))
        s["model_in_use"] = p.get("first_yolo_model", "yolo11n")
        s["det_class_hint"] = p.get("yolo_class")
        s["need_human_review"] = bool(p.get("parse_error"))
        print("Parsed:", p)
        print(f"clip_prompt={s['prompt']}  first_yolo_model={s['model_in_use']}  yolo_class={s.get('det_class_hint')}")
        return s

    def n_yolo_select(s):
    # 若上一帧标记了需要升级，这里一次性执行，并清理标志
        if s.pop("request_upgrade", False):
            # 也可调用 select_yolo_model(...) 做更聪明的选择
            s["model_in_use"] = "yolo11x"
            s["cooldown_left"] = max(0, s.get("cooldown_left", 0))
            return s

        # 正常的选择逻辑（含 cooldown 保持）
        r = s.get("policy", {}).get("routing", {})
        override = r.get("model_override")
        if override:
            s["model_in_use"] = override
        else:
            s["model_in_use"] = select_yolo_model(
                s.get("prompt",""),
                s.get("missing_gap",0),
                s.get("stable",True),
                s.get("model_in_use","yolo11n"),
                s.get("cooldown_left",0)
            )
        return s

    def n_detect(s):
        det = s["detector"]
        alias = s.get("model_in_use", "yolo11n")
        # 先调用 API 拿全量框
        # （UltralyticsAPIDetector.detect 已支持 alias 参数的话，可这样传；否则省略 alias）
        _ = det.detect  # 仅为静态检查
        raw = det.detect(s["frame"], conf=0.25, iou=0.45, alias=alias)
        # 按提示与阈值过滤成统一结构
        s["detections"] = yolo_to_boxes_api(s["frame"], det, s.get("det_class_hint"), min_conf=0.5)
        s["det_count"] = len(s["detections"])
        print(f"API returned {s['det_count']} objects for {alias} (hint={s.get('det_class_hint')})")
        return s

    def n_det_check(s):
        cnt = len(s.get("detections", []))
        s["det_count"] = cnt
        s["det_found"] = cnt > 0
        if s["det_found"]:
            s["missing_gap"] = 0
            s["escalated_in_episode"] = False   # ← 本轮缺失结束，清除“已升级”闸门
        else:
            s["missing_gap"] = s.get("missing_gap", 0) + 1
        return s

    def n_plan_tracking(s):
    # 规划：可能返回 {"bt_cfg": {...}}，也可能不返回（留给执行时兜底）
        s["policy"] = {**s.get("policy", {}), **plan_tracking(s)}
        return s

    def n_bytetrack(s):
        # 取最终追踪配置：优先 policy.bt_cfg，其次已有 s["bt_cfg"]，否则按缺失自适应
        bt_cfg = s.get("policy", {}).get("bt_cfg") or s.get("bt_cfg") or adapt_bt_config(s.get("missing_gap", 0))
        s["bt_cfg"] = bt_cfg

        out = run_bytetrack(
            s.get("frame"),                 # ← 传入当前帧
            s.get("detections", []),        # ← 检测结果
            bt_cfg,                         # ← 追踪配置
            s.get("frame_idx", -1),         # ← 帧号
        )
        s["tracks"] = out.get("tracks", [])
        s["id_switch"] = out.get("id_switch", 0)
        s["frame_ids"] = out.get("frame_ids", [])
        return s

    def n_bt_check(s):
        s["missing_gap"] = 0 if s.get("det_found") else s.get("missing_gap",0)+1
        s["stable"] = (s.get("id_switch",0)==0 and s["missing_gap"]==0); return s

    def n_plan_clip(s):
        # 规划：可返回 {"clip_thresh": ... , "review_avg_min": ..., "review_margin_min": ...}
        s["policy"] = {**s.get("policy", {}), **plan_clip(s)}
        return s

    # graph_build.py 中，替换你的 n_clip_apply
    def n_clip_apply(s):
        dets   = s.get("detections", [])
        prompt = s.get("prompt", "")
        frame  = s.get("frame")  # ← 关键：把当前帧传给 clip_stats，走 CLIP 路径

        # ★ 优化：只在第一帧加载CLIP模型
        if s.get("clip_model") is None:
            from adapter.clip_matcher import load_clip_model
            s["clip_model"], s["clip_preprocess"], s["clip_tokenizer"], s["clip_device"] = load_clip_model()
            print("[CLIP] 模型加载完成")

        # 使用已加载的模型
        st = clip_stats(dets, prompt, frame, 
                       s["clip_model"], s["clip_preprocess"], s["clip_tokenizer"], s["clip_device"])

        s["avg_conf"], s["top10_conf"], s["margin"] = st["avg_conf"], st["top10_conf"], st["margin"]
        print(f"[CLIP] frame={s.get('frame_idx',-1)} n={len(dets)} "
            f"avg={st['avg_conf']:.3f} top10={st['top10_conf']:.3f} margin={st['margin']:.3f}")

        # 阈值：优先 Planner 给的；否则按配置自适应（不再落回 0.25）
        pol = s.get("policy", {})
        cfg_clip = (s.get("cfg", {}) or {}).get("clip", {})  # 如果你没把 cfg 放到 state，可留空用默认
        base  = float(cfg_clip.get("base_thresh", 0.70))
        tight = float(cfg_clip.get("tight_thresh", 0.80))
        loose = float(cfg_clip.get("loose_thresh", 0.60))
        delta = float(cfg_clip.get("margin_delta", 0.05))

        auto_th = max(loose, min(tight, max(base, st["top10_conf"] - delta)))
        s["clip_thresh"] = pol.get("clip_thresh") or auto_th
        th = s["clip_thresh"]

        # 逐个检测打印（优先用 CLIP 分数，若没有则回退到 detection confidence）
        def _score(d):
            return float(d.get("clip_score", d.get("confidence", d.get("score", 0.0))))
        for i, d in enumerate(dets):
            sc = _score(d)
            x1, y1, x2, y2 = d.get("bbox", [0,0,0,0])
            name = d.get("name", d.get("class_name", "?"))
            keep = "KEEP" if sc >= th else "DROP"
            print(f"[CLIP]   det#{i}: s={sc:.3f} th={th:.3f} {keep} "
                f"name={name} bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")

        return s
    # --- 小工具：IoU 与 Track↔Det 匹配 ---
    def _iou_xyxy(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = max(1e-6, area_a + area_b - inter)
        return inter / union

    def _match_det_for_track(track_bbox, dets, iou_thr=0.3):
        best, best_iou = None, 0.0
        for d in dets:
            iou = _iou_xyxy(track_bbox, d.get("bbox", [0,0,0,0]))
            if iou > best_iou:
                best, best_iou = d, iou
        return best if best_iou >= iou_thr else None
    
    # --- 节点：为当前帧的 tracks 赋 Global ID ---
    def n_global_id(s):
        tracks = s.get("tracks", [])          # [{track_id, bbox, ...}]
        dets   = s.get("detections", [])      # YOLO/筛选后的检测，若含 img_feat 则更佳
        th     = float(s.get("clip_thresh", 0.0))

        for t in tracks:
            tid  = int(t["track_id"])
            tb   = t.get("bbox", [0,0,0,0])
            d    = _match_det_for_track(tb, dets)  # 取与该轨迹 IoU 最大的检测

            # 1) 统计该轨迹的“通过率”（如果有 clip_score 就用它与阈值比较）
            passed = False
            if d is not None:
                sc = float(d.get("clip_score", d.get("confidence", d.get("score", 0.0))))
                passed = (sc >= th) if th > 0 else True
            ap, tot, rate = update_stats(tid, passed)

            # 2) ★ 修复：按照capstone2的正确逻辑分配Global ID
            gid = get_assigned_gid(tid)  # 先检查是否已经分配过
            if gid is None and passed and d is not None and isinstance(d.get("img_feat"), torch.Tensor):
                # 只有通过CLIP阈值且有视觉特征时才分配新的Global ID
                pidx = abs(hash(s.get("prompt",""))) % 10_000_000
                gid = get_or_assign_gid_with_reuse(tid, d["img_feat"], prompt_idx=pidx, sim_threshold=0.10)
            
            # 3) 设置Global ID（可能为None）
            t["global_id"] = int(gid) if gid is not None else None

        # 可选：统计输出
        if tracks:
            gids = [tr.get("global_id") for tr in tracks if tr.get("global_id") is not None]
            if gids:
                print(f"[GID] frame={s.get('frame_idx',-1)} assigned GIDs={gids}")

        return s

    def n_plan_bridge(s):
        bridge_plan = plan_bridge(s)
        s["policy"] = {**s.get("policy", {}), **bridge_plan}
        # ★ 修复：将max_bridge直接放到状态根级别，供export使用
        if "max_bridge" in bridge_plan:
            s["max_bridge"] = bridge_plan["max_bridge"]
        return s

    def n_human(s): s["need_human_review"]=False; return s

    # ========= 节点：Edit =========
    def n_edit(s):
        """
        1) 将本帧的 tracks[global_id] 写入 gid_frames 映射
        2) 对每个 gid 的帧序列做合并（gap/bridge）
        3) ★ 收集每个GID的CLIP分数统计
        4) 可选：实时预览叠框（s['show_preview']=True 时）
        """
        idx = int(s.get("frame_idx", 0))
        tracks = s.get("tracks", [])
        detections = s.get("detections", [])
        gid_frames = s.get("gid_frames", {})   # Dict[int, List[int]]
        gid_scores = s.get("gid_scores", {})   # Dict[int, List[float]] - 新增

        # 1) 累计 gid -> frames 和 gid -> scores（只有passed的track才记录）
        for t in tracks:
            gid = t.get("global_id")
            if gid is None:
                continue
                
            # ★ 检查是否通过CLIP阈值（只有passed的才记录到gid_frames）
            track_bbox = t.get("bbox", [0,0,0,0])
            best_det = None
            best_iou = 0.0
            
            for det in detections:
                det_bbox = det.get("bbox", [0,0,0,0])
                # 计算IoU
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
                
                # 只有通过CLIP阈值的才记录
                if passed:
                    gid_frames.setdefault(int(gid), []).append(idx)
                    gid_scores.setdefault(int(gid), []).append(clip_score)
        
        s["gid_frames"] = gid_frames
        s["gid_scores"] = gid_scores  # 新增

        # 2) 合并片段（每个 gid 各自合并）
        # gap/bridge 来源：优先 policy / plan，回退默认
        gap = int(s.get("policy", {}).get("gap", 2))
        bridge = int(s.get("max_bridge", s.get("policy", {}).get("bridge", 15)))

        merged = {
            gid: merge_frame_ranges(frames, gap=gap, bridge=bridge)
            for gid, frames in gid_frames.items()
        }
        s["merged_segments"] = merged  # Dict[int, List[(start,end)]]

        # 调试输出（可删）
        if merged:
            sample_gid = next(iter(merged))
            print(f"[EDIT] frame={idx} sample gid={sample_gid} segments={merged[sample_gid]}")
            


        # 3) 预览：画框+GID（可选）
        if s.get("show_preview", True) and s.get("frame") is not None:
            ok = run_video(
                s["frame"].copy(),
                s.get("tracks", []),
                detections=s.get("detections", []),
                clip_thresh=s.get("clip_thresh"),
                yolo_thresh=s.get("yolo_conf_thresh", 0.25),  # 可不使用，仅占位
            )
            if ok is False:
                s["stop"] = True
        return s

    # ========= 节点：Export =========
    def n_export(s):
        """
        导出匹配prompt的最佳对象视频
        """
        # ★ 优化：只在最后一帧导出视频
        frame_idx = s.get("frame_idx", 0)
        total_frames = s.get("total_frames", 0)
        
        # 如果不是最后一帧，跳过导出
        if frame_idx < total_frames - 1:
            return s
        
        gid_frames = s.get("gid_frames", {})
        gid_scores = s.get("gid_scores", {})
        video_path = s.get("video_path", "")
        output_dir = s.get("output_dir", "./output")
        fps = s.get("fps", 30.0)
        
        # 获取配置参数
        cfg = s.get("cfg", {})
        bridge_config = cfg.get("bridge", {})
        gap = int(bridge_config.get("default", 8))
        # ★ 修复：使用PlanBridge设置的max_bridge值，而不是配置文件的默认值
        bridge = int(s.get("max_bridge", bridge_config.get("default", 8)))
        
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
        except Exception as e:
            s["export_error"] = str(e)
            print(f"[ERROR] 视频导出失败: {e}")
        
        return s


    # ---------- 条件 ----------
    '''def c_det(s):
        r = s.get("policy", {}).get("switch", {}) or s.get("policy", {}).get("routing", {})
        short_gap, long_gap = r.get("short_gap",2), r.get("long_gap",15)
        if s.get("det_found"): return "ok"
        gap = s.get("missing_gap",0)
        if gap >= long_gap: return "long_missing"
        if gap <= short_gap: return "short_missing"
        return "short_missing"'''
    
    def c_det(s):
        r = s.get("policy", {}).get("switch", {})
        long_gap = r.get("long_gap", 15)
        if s.get("det_found"):
            return "ok"
        # 记一个“申请升级”的标志位，但不在本帧立刻跳 YOLOSelect
        if s.get("missing_gap", 0) >= long_gap:
            s["request_upgrade"] = True
        return "miss"


    def c_bt(s):  return "retune" if (s.get("id_switch",0)>0 and not s.get("stable",True)) else "ok"
    def c_clip(s): return "review" if s.get("need_human_review",False) else "accept"

    # ---------- 注册节点 ----------
    g.add_node("ParseInstruction", n_parse)
    g.add_node("YOLOSelect", n_yolo_select)
    g.add_node("YOLODetect", n_detect)
    g.add_node("detectResultCheck", n_det_check)
    g.add_node("PlanTracking", n_plan_tracking)
    g.add_node("ByteTrack", n_bytetrack)
    g.add_node("ByteTrackCheck", n_bt_check)
    g.add_node("PlanCLIP", n_plan_clip)
    g.add_node("TuneCLIP", n_clip_apply)
    g.add_node("GlobalID", n_global_id)
    g.add_node("PlanBridge", n_plan_bridge)
    g.add_node("Edit", n_edit)
    g.add_node("Export", n_export)
    g.add_node("HumanReview", n_human)

    # ---------- 连边（多 Planner，顺序清晰） ----------
    # 入口
    g.add_edge(START, "ParseInstruction")
    g.add_edge("ParseInstruction", "YOLOSelect")
    g.add_edge("YOLOSelect", "YOLODetect")
    g.add_edge("YOLODetect", "detectResultCheck")

    g.add_conditional_edges("detectResultCheck", c_det,
    {"ok":"PlanTracking", "miss":"PlanTracking"})

    g.add_edge("PlanTracking", "ByteTrack")
    g.add_edge("ByteTrack", "ByteTrackCheck")
    g.add_edge("ByteTrackCheck", "PlanCLIP")
    g.add_edge("PlanCLIP", "TuneCLIP")
    g.add_edge("TuneCLIP", "GlobalID")
    g.add_edge("GlobalID", "PlanBridge")
    g.add_edge("PlanBridge", "Edit")
    g.add_edge("Edit", "Export")
    # 结束
    g.add_edge("Export", END)           # 流程在 Export 后结束

    return g.compile()