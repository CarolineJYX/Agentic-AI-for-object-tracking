from typing import List, Tuple
# Step 6: Import MoviePy for video clipping
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import cv2
import os

# run_video
def run_video(frame, tracked_objects):
    """
    在当前帧上画出目标框 + Global ID，并显示画面。
    
    参数:
        frame: 当前帧图像 (numpy array)
        tracked_objects: 每个目标应包含字段 ['bbox', 'global_id']
    """
    for obj in tracked_objects:
        x1, y1, x2, y2 = map(int, obj['bbox'])
        gid = obj.get('global_id') 

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID: {gid}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Detection", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        return False  # 终止信号
    return True

from typing import List, Tuple, Dict
from moviepy.editor import VideoFileClip, concatenate_videoclips

def merge_frame_ranges(frames: List[int], gap: int = 2, bridge: int = 15) -> List[Tuple[int, int]]:
    """
    Merge a sorted list of frame indices into continuous segments, and optionally bridge
    small gaps between segments by merging them too.

    Args:
        frames: Frame indices (can be unsorted / have duplicates).
        gap:    Max gap (in frames) to consider frames continuous within a segment.
        bridge: Max gap (in frames) between *segments* to merge them into one segment.
                Example: if bridge=10 and we have [(1, 30), (35, 50)], since 35-30-1=4<=10,
                they will be merged into (1, 50).

    Returns:
        List of (start_frame, end_frame) tuples.
    """
    if not frames:
        return []

    # 1) 去重 + 排序
    frames = sorted(set(frames))

    # 2) 先按 gap 合并成基础片段
    segments: List[Tuple[int, int]] = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f - prev <= gap:
            prev = f
        else:
            segments.append((start, prev))
            start = prev = f
    segments.append((start, prev))

    if bridge <= 0 or len(segments) <= 1:
        return segments

    # 3) 再按 bridge 把相邻片段桥接（把小空档也并进去）
    bridged: List[Tuple[int, int]] = []
    cur_s, cur_e = segments[0]
    for s, e in segments[1:]:
        # 片段间空档 = 下段起点 - 上段终点 - 1
        gap_between = s - cur_e - 1
        # ★ 调试：输出桥接决策过程
        print(f"[BRIDGE] 检查: ({cur_s},{cur_e}) -> ({s},{e}), 间隔={gap_between}, 阈值={bridge}")
        if gap_between <= bridge:
            # 合并：直接延长到后一个片段的末尾
            print(f"[BRIDGE] ✅ 桥接: ({cur_s},{cur_e}) + ({s},{e}) -> ({cur_s},{e})")
            cur_e = e
        else:
            print(f"[BRIDGE] ❌ 跳过: 间隔{gap_between} > 阈值{bridge}")
            bridged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    bridged.append((cur_s, cur_e))
    return bridged


def clip_video_by_gid(
    video_path: str,
    gid_to_frames: Dict[int, List[int]],
    target_gid: int,
    fps: float,
    output_path: str,
    gap: int = 2,
    bridge: int = 15
):
    """
    Clip segments of a video corresponding to a specific global ID, and optionally
    bridge small gaps between segments.

    Args:
        video_path: Path to the source video.
        gid_to_frames: Mapping from global_id to its frame indices.
        target_gid: Global ID to clip.
        fps: Frames per second of the video.
        output_path: Path to save the final clipped video.
        gap:   See merge_frame_ranges().
        bridge:See merge_frame_ranges() for bridging small gaps between segments.
    """
    frames = gid_to_frames.get(target_gid, [])
    if not frames:
        print(f"[EDIT] No frames for gid={target_gid}. Skip export.")
        return

    video = VideoFileClip(video_path)
    segments = merge_frame_ranges(frames, gap=gap, bridge=bridge)
    print(f"[EDIT] Segments after merge+bridge: {segments}")

    clips = []
    for start_f, end_f in segments:
        # 注意 end_t 用 (end_f + 1) / fps，确保最后一帧也包含到
        start_t = start_f / fps
        end_t   = (end_f + 1) / fps
        clips.append(video.subclip(start_t, end_t))

    final_clip = concatenate_videoclips(clips) if len(clips) > 1 else clips[0]
    final_clip.write_videofile(output_path)



def clip_video_with_music(video_path, audio_path, output_path):
    """
    Add background music to an entire video (no cutting; music length is matched to video duration).

    Args:
        video_path (str): Path to the source video.
        audio_path (str): Path to the background music file.
        output_path (str): Path to save the final video with background music.
    """
    print("[INFO] Loading video and audio")
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path).subclip(0, video.duration)  # Trim music to match video length

    final = video.set_audio(audio)
    final.write_videofile(
        output_path,
        fps=30,
        audio_codec="aac",                 # Ensure proper audio codec
        temp_audiofile="temp-audio.m4a",   # Temporary audio file to avoid ffmpeg errors
        remove_temp=True
    )
    print(f"[DONE] Successfully exported video with background music: {output_path}")


# adapter/editing.py
import cv2
from typing import List, Dict, Optional

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

def _match_det_for_track(track_bbox, dets, iou_thr=0.3) -> Optional[Dict]:
    best, best_iou = None, 0.0
    for d in dets or []:
        iou = _iou_xyxy(track_bbox, d.get("bbox", [0,0,0,0]))
        if iou > best_iou:
            best, best_iou = d, iou
    return best if best_iou >= iou_thr else None

def run_video(
    frame,
    tracked_objects: List[Dict],
    detections: Optional[List[Dict]] = None,
    clip_thresh: Optional[float] = None,
    yolo_thresh: Optional[float] = None,
    win_name: str = "Agent Preview"
):
    """
    叠加：bbox + GID + raw track id + YOLO conf + CLIP score（带简易条形条）
    - tracked_objects: 来自 ByteTrack 的每个轨迹，含 bbox / track_id / raw_track_id(可选) / global_id(可选)
    - detections: YOLO/CLIP 的检测，含 bbox / confidence(或score) / class_name(可选) / clip_score(可选)
    """
    H, W = frame.shape[:2]

    for t in tracked_objects or []:
        x1, y1, x2, y2 = map(int, t["bbox"])
        tb = [x1, y1, x2, y2]

        gid    = t.get("global_id", "?")
        raw_id = t.get("raw_track_id", t.get("track_id", "?"))

        d = _match_det_for_track(tb, detections, iou_thr=0.3)
        yolo_conf = None
        clip_score = None
        cls_name = None
        if d is not None:
            yolo_conf = d.get("confidence", d.get("score", None))
            clip_score = d.get("clip_score", None)
            cls_name = d.get("class_name", None)

        # 颜色逻辑：默认灰；CLIP 有分数且过阈值则绿；有分数但不过阈值橙
        passed = (clip_score is not None and clip_thresh is not None and float(clip_score) >= float(clip_thresh))
        color = (200,200,200)
        if clip_score is not None and clip_thresh is not None:
            color = (0,255,0) if passed else (0,165,255)

        # 框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 文本三行
        line1 = f"GID:{gid} RAW:{raw_id}" 
        if cls_name is not None:
            line1 += f" {cls_name}"
        line2 = f"YOLO:{yolo_conf:.2f}" if yolo_conf is not None else "YOLO:N/A"
        line3 = f"CLIP:{clip_score:.2f}" if clip_score is not None else "CLIP:N/A"

        ytxt = max(14, y1 - 8)
        for i, txt in enumerate([line1, line2, line3]):
            cv2.putText(frame, txt, (x1, ytxt + 18*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # 简易进度条（宽度100像素）
        bar_w = 100
        # YOLO bar (灰)
        if yolo_conf is not None:
            v = max(0.0, min(1.0, float(yolo_conf)))
            cv2.rectangle(frame, (x1, min(H-1, y2+6)), (x1+int(bar_w*v), min(H-1, y2+12)), (128,128,128), -1)
        # CLIP bar（绿/红）
        if clip_score is not None:
            v = max(0.0, min(1.0, float(clip_score)))
            col = (0,255,0) if passed else (0,0,255)
            cv2.rectangle(frame, (x1, min(H-1, y2+16)), (x1+int(bar_w*v), min(H-1, y2+22)), col, -1)

    cv2.imshow(win_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        return False
    return True

def export_best_match_video(
    video_path: str,
    gid_frames: Dict[int, List[int]],
    gid_scores: Dict[int, List[float]],
    output_dir: str,
    fps: float,
    gap: int = 8,
    bridge: int = 50  # ★ 优化：增加默认bridge值
) -> str:
    """
    导出匹配prompt的最佳对象视频
    
    Args:
        video_path: 源视频路径
        gid_frames: Global ID到帧列表的映射
        gid_scores: Global ID到CLIP分数列表的映射
        output_dir: 输出目录
        fps: 视频帧率
        gap: 片段内最大间隔
        bridge: 片段间最大间隔
    
    Returns:
        导出的视频文件路径
    """
    if not gid_frames:
        raise ValueError("没有找到任何跟踪对象")
    
    # 找到最佳匹配的Global ID
    best_gid = None
    best_score = -1.0
    
    # ★ 修改策略：优先选择帧数最多的GID，然后考虑CLIP分数
    max_frames = 0
    best_gid_by_frames = None
    
    # 1) 找到帧数最多的GID
    for gid, frames in gid_frames.items():
        if len(frames) > max_frames:
            max_frames = len(frames)
            best_gid_by_frames = gid
    
    # 2) 在帧数最多的几个GID中选择CLIP分数最高的
    top_gids = []
    for gid, frames in gid_frames.items():
        if len(frames) >= max_frames * 0.8:  # 帧数达到最多的80%以上
            top_gids.append(gid)
    
    # 3) 在这些GID中选择CLIP分数最高的
    for gid in top_gids:
        scores = gid_scores.get(gid, [])
        if len(scores) > 0:
            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_gid = gid
    
    # 4) 如果没有CLIP分数，选择帧数最多的
    if best_gid is None:
        best_gid = best_gid_by_frames
    
    if best_gid is None:
        raise ValueError("没有找到匹配prompt的对象")
    
    # 获取最佳GID的帧列表
    frames = gid_frames.get(best_gid, [])
    if not frames:
        raise ValueError(f"GID {best_gid} 没有帧数据")
    
    # 导出视频
    output_path = os.path.join(output_dir, "prompt_match_clip.mp4")
    
    clip_video_by_gid(
        video_path=video_path,
        gid_to_frames={best_gid: frames},
        target_gid=best_gid,
        fps=fps,
        output_path=output_path,
        gap=gap,
        bridge=bridge
    )
    
    return output_path