# 规则版（默认），随时可换成 LLM/RL
from typing import Dict, Any

# 每个函数只返回自己负责的策略字段，最终合并进 state["policy"]
def plan_routing(state) -> dict:
    # 模型选择阈值 & 覆写（示例：长缺失强制 12x）
    gap = state.get("missing_gap", 0)
    policy = {"routing": {"short_gap": 2, "long_gap": 15}}
    if gap >= 15:
        policy["routing"]["model_override"] = "yolo12x"
    return policy

def plan_tracking(state) -> dict:
    gap = state.get("missing_gap", 0)
    if gap <= 2:
        bt = dict(track_thresh=0.30, match_thresh=0.80, track_buffer=30)
    elif gap >= 15:
        bt = dict(track_thresh=0.50, match_thresh=0.70, track_buffer=50)
    else:
        bt = dict(track_thresh=0.40, match_thresh=0.75, track_buffer=30)
    return {"bt_cfg": bt, "switch": {"short_gap": 2, "long_gap": 15}}

def plan_clip(state) -> dict:
    avg, top10 = state.get("avg_conf", 0.0), state.get("top10_conf", 0.0)
    if avg > 0.90 and top10 > 0.85: thresh = 0.30
    elif avg < 0.60:               thresh = 0.25
    else:                           thresh = 0.25
    return {"clip_thresh": thresh}

def plan_bridge(state) -> dict:
    stable = state.get("stable", True)
    # ★ 优化：增加bridge值以合并更多相近片段
    # 基于分析，很多间隔在16-41帧之间，应该可以合并
    if stable:
        return {"max_bridge": 50}  # 稳定时允许更大的bridge
    else:
        return {"max_bridge": 30}  # 不稳定时也适当增加

# —— 若要 LLM 决策，示例（可替换上面的规则版）——
"""
from langchain_openai import ChatOpenAI
import json
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
def ai_plan(state):
    prompt = f\"\"\"Given: {json.dumps({k: state.get(k) for k in [
        'missing_gap','avg_conf','top10_conf','stable','model_in_use']})}
Return JSON with keys: switch(short_gap,long_gap), bt_cfg, clip_thresh, max_bridge, model_override(optional).\"\"\"
    text = llm.invoke(prompt).content
    return json.loads(text)
"""