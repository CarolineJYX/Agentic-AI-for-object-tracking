# here i set the rules for model change
def select_yolo_model(prompt: str, missing_gap: int, stable: bool,
                      last_model: str, cooldown_left: int,
                      short_gap=2, long_gap=15) -> str:
    if cooldown_left > 0:
        return last_model
    elif missing_gap >= long_gap:
        return "yolo11x"
    elif stable:
        return "yolo11n"            
    elif last_model == "yolo11n" and missing_gap >= short_gap:
        return "yolo11m"
    else:
        return last_model

def adapt_bt_config(missing_gap: int):
    if missing_gap <= 2:   return dict(track_thresh=0.30, match_thresh=0.80, track_buffer=30)
    if missing_gap >= 15:  return dict(track_thresh=0.50, match_thresh=0.70, track_buffer=50)
    return dict(track_thresh=0.40, match_thresh=0.75, track_buffer=30)

def adapt_clip_threshold(avg: float, top10: float, base=0.70, tight=0.80, loose=0.60):
    if avg > 0.90 and top10 > 0.85: return tight
    if avg < 0.60: return loose
    return base

def decide_bridge(is_stable: bool, high_motion: bool, default=8, hi=4, lo=12):
    if high_motion: return hi
    return lo if is_stable else default