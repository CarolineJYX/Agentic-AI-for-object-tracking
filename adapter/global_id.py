# global_id.py
from typing import Dict, Optional, Tuple
import torch

gid_counter: int = 0
track_stats: Dict[int, Dict[str, int]] = {}          # track_id -> {"approved": int, "total": int}
track_to_gid: Dict[int, int] = {}                    # track_id -> gid
gid_bank: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
# gid_bank[prompt_idx][gid] = {"centroid": 1xD tensor, "count": int}

@torch.no_grad()
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity for 1xD tensors (handles unnormalized inputs)."""
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-6)
    return float((a @ b.T).item())

def reset_global_state():
    """Reset all global state (start a fresh session)."""
    global gid_counter, track_stats, track_to_gid, gid_bank
    gid_counter = 0
    track_stats.clear()
    track_to_gid.clear()
    gid_bank.clear()

def update_stats(track_id: int, passed_clip: bool) -> Tuple[int, int, float]:
    """
    Update CLIP pass statistics for a track_id.

    Returns:
        approved (int), total (int), rate (float)
    """
    s = track_stats.setdefault(track_id, {"approved": 0, "total": 0})
    s["total"] += 1
    if passed_clip:
        s["approved"] += 1
    rate = s["approved"] / max(1, s["total"])
    return s["approved"], s["total"], rate

def is_ready(track_id: int) -> bool:
    """
    A track becomes 'ready' if:
      - approved >= 10  OR
      - approved / total >= 0.2
    """
    s = track_stats.get(track_id)
    if not s:
        return False
    approved, total = s["approved"], s["total"]
    return (approved >= 10) or (approved / max(1, total) >= 0.2)

@torch.no_grad()
def get_or_assign_gid_with_reuse(
    track_id: int,
    image_feature: torch.Tensor,   # 1xD CLIP image feature
    prompt_idx: int,
    sim_threshold: float = 0.1
) -> Optional[int]:
    """
    When a track is 'ready', try to reuse an existing GID within the same prompt
    by cosine similarity to the prompt's GID centroids. If best_sim >= threshold,
    reuse that GID; otherwise create a new GID.

    Returns:
        gid (int) if assigned, else None if not ready yet.
    """
    global gid_counter

    # Already assigned
    if track_id in track_to_gid:
        return track_to_gid[track_id]

    # Not ready -> keep accumulating stats
    if not is_ready(track_id):
        return None

    # Find best match in this prompt's bank
    bank_p = gid_bank.get(prompt_idx, {})
    best_gid, best_sim = None, -1.0
    for gid, rec in bank_p.items():
        sim = cosine_sim(image_feature, rec["centroid"])
        if sim > best_sim:
            best_sim, best_gid = sim, gid

    if best_gid is not None and best_sim >= sim_threshold:
        # Reuse: update centroid with running mean
        rec = bank_p[best_gid]
        n = int(rec["count"])
        feat = image_feature / (image_feature.norm(dim=-1, keepdim=True) + 1e-6)
        rec["centroid"] = (rec["centroid"] * n + feat) / (n + 1)
        rec["count"] = n + 1
        track_to_gid[track_id] = best_gid
        return best_gid

    # Create a new GID
    new_gid = gid_counter
    gid_counter += 1
    feat = image_feature / (image_feature.norm(dim=-1, keepdim=True) + 1e-6)
    if prompt_idx not in gid_bank:
        gid_bank[prompt_idx] = {}
    gid_bank[prompt_idx][new_gid] = {"centroid": feat.clone(), "count": 1}
    track_to_gid[track_id] = new_gid
    return new_gid

def get_assigned_gid(track_id: int) -> Optional[int]:
    return track_to_gid.get(track_id)

# global_id.py 里新增
def get_assigned_gid(track_id: int) -> Optional[int]:
    """Read-only lookup: return existing gid if this track_id has been assigned before."""
    return track_to_gid.get(track_id)

def get_or_assign_gid_sticky(track_id: int) -> int:
    """
    Sticky mapping: assign a gid to this track_id immediately (no CLIP / no readiness gating).
    Use for stable on-screen IDs. This does NOT touch the reuse/bank logic.
    """
    global gid_counter
    gid = track_to_gid.get(track_id)
    if gid is not None:
        return gid
    gid = gid_counter
    gid_counter += 1
    track_to_gid[track_id] = gid
    return gid