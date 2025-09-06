import cv2, yaml, argparse
from agent.graph_build import build_app
from adapter.detection_api import build_detector

def iter_frames(path):
    cap = cv2.VideoCapture(path); i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        yield i, frame
        i += 1
    cap.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="configs/agent.yaml")
    # ★ 新增两种输入方式
    ap.add_argument("--text", type=str, default="", help="natural language instruction; overrides config prompt")
    ap.add_argument("--ask", action="store_true", help="prompt for instruction interactively")
    # ★ 新增输出目录参数
    ap.add_argument("--output_dir", type=str, default="./output", help="output directory for clipped videos")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # ★ 决定 user_input 来源：--text > --ask > config.prompt
    user_text = (args.text or "").strip()
    if not user_text and args.ask:
        user_text = input("Enter instruction (e.g., 'track the yellow dog'): ").strip()
    if not user_text:
        user_text = cfg.get("prompt", "")

    detector = build_detector(cfg)
    app = build_app()

    state = {
        "user_input": user_text,                    # ★ 给 ParseInstruction 节点用
        "prompt": user_text or cfg.get("prompt", ""),
        "model_in_use": cfg.get("default_model", "yolo11n"),
        "cooldown_left": 0,
        "missing_gap": 0,
        "id_switch": 0,
        "stable": True,
        "detector": detector,
        "cfg": cfg,
        "recursion_limit": 6,
    }

    # 获取视频信息
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 确保输出目录存在
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # 添加导出相关参数到状态
    state.update({
        "video_path": args.video,
        "output_dir": args.output_dir,
        "fps": fps,
        "total_frames": total_frames,
    })

    # 处理视频帧
    for idx, frame in iter_frames(args.video):
        state["frame_idx"] = idx
        state["frame"] = frame
        state = app.invoke(state)

        # ★ 优化：减少日志输出频率
        if idx % 10 == 0 or idx == total_frames - 1:  # 每10帧输出一次，最后一帧必输出
            print(f"[{idx}/{total_frames}] model={state.get('model_in_use')} gap={state.get('missing_gap')}"
                  f" det={state.get('det_count','NA')}"
                  f" bt={state.get('bt_cfg')} clip={state.get('clip_thresh')}"
                  f" review={state.get('need_human_review', False)}")

    # 检查导出结果
    if state.get("export_error"):
        print(f"[ERROR] 视频导出失败: {state.get('export_error')}")
    elif state.get("exported_video_path"):
        print(f"[SUCCESS] 视频导出完成: {state.get('exported_video_path')}")
    else:
        print("[WARNING] 没有导出视频")

if __name__ == "__main__":
    main()