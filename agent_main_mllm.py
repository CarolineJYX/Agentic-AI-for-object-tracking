#!/usr/bin/env python3
"""
MLLM增强的Agentic视频目标跟踪系统
集成多模态LLM进行智能参数决策

作者: Caroline Xia
版本: MLLM Enhanced
"""

import cv2
import argparse
from pathlib import Path
from agent.graph_build_mllm import build_mllm_app

def main():
    parser = argparse.ArgumentParser(description="MLLM增强的Agentic视频目标跟踪系统")
    parser.add_argument("--video", type=str, required=True, help="输入视频文件路径")
    parser.add_argument("--text", type=str, required=True, help="目标描述文本")
    parser.add_argument("--output_dir", type=str, default="./mllm_output", help="输出目录")
    
    args = parser.parse_args()
    
    # 验证输入
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ 视频文件不存在: {args.video}")
        return
    
    print("🚀 启动MLLM增强的Agentic视频跟踪系统")
    print("="*60)
    print(f"📹 输入视频: {args.video}")
    print(f"🎯 跟踪目标: {args.text}")
    print(f"📁 输出目录: {args.output_dir}")
    print("="*60)
    
    # 构建MLLM应用
    app = build_mllm_app()
    
    # 获取视频信息
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 视频信息: {total_frames} 帧, {fps:.1f} FPS")
    
    # 初始化状态
    initial_state = {
        "video_path": args.video,
        "prompt": args.text,
        "user_input": args.text,
        "output_dir": args.output_dir,
        "fps": fps,
        "total_frames": total_frames,
        "frame_idx": 0,
        "gid_frames": {},
        "gid_scores": {},
        "missing_gap": 0,
        "stable": True,
        "need_human_review": False
    }
    
    print(f"\n🤖 开始MLLM参数分析...")
    
    # 首先进行MLLM参数分析（只执行一次）
    mllm_state = initial_state.copy()
    mllm_state.update({
        "frame": None,  # 初始化时不需要frame
        "frame_idx": 0
    })
    
    # 只调用MLLM相关节点进行参数决策
    from multimodal_advisor import MultiModalAdvisor
    
    try:
        API_KEY = "sk-ZKv84hrnoNgqB8bXy6qrIWga9ak4zHI1M7RcByXCLdRcoG1o"
        BASE_URL = "https://api.xinyun.ai/v1"
        advisor = MultiModalAdvisor(API_KEY, BASE_URL)
        
        result = advisor.analyze_video(args.video, args.text)
        llm_analysis = result.get('llm_analysis', {})
        
        if "error" not in llm_analysis:
            # 检查目标是否存在
            target_exists = llm_analysis.get("target_exists", True)
            target_found_frames = llm_analysis.get("target_found_frames", [])
            
            if not target_exists:
                print(f"   ❌ MLLM检测结果: 视频中不存在目标物体")
                analysis = llm_analysis.get("analysis", {})
                reason = analysis.get("target_description_match", "未提供原因")
                print(f"   📝 详细原因: {reason}")
                print(f"   🚫 建议: 请检查目标描述是否准确，或更换视频")
                
                # 询问用户是否继续
                print(f"\n❓ 是否仍要继续处理? (输入 'y' 继续，其他键退出)")
                user_input = input().strip().lower()
                if user_input != 'y':
                    print(f"   ⏹️ 用户选择退出处理")
                    cap.release()
                    return
                else:
                    print(f"   ⚠️ 用户选择继续，使用默认参数")
            
            # 如果目标存在或用户选择继续，设置参数
            if "recommended_params" in llm_analysis:
                params = llm_analysis["recommended_params"]
                analysis = llm_analysis.get("analysis", {})
                confidence = llm_analysis.get("confidence", 0.0)
                
                # 更新全局参数
                initial_state.update({
                    "model_in_use": "yolo11n",  # 暂时使用轻量模型
                    "clip_thresh": float(params.get("clip_thresh", 0.25)),
                    "max_bridge": int(params.get("max_bridge", 30)),
                    "gap": int(params.get("gap", 2)),
                    "track_buffer": int(params.get("track_buffer", 30)),
                    "match_thresh": float(params.get("match_thresh", 0.8)),
                    "track_thresh": float(params.get("track_thresh", 0.3)),
                    "mllm_confidence": float(confidence),
                    "mllm_reasoning": analysis.get("parameter_reasoning", "MLLM参数推荐"),
                    "target_exists": target_exists,
                    "target_found_frames": target_found_frames
                })
                
                print(f"   ✅ MLLM参数决策完成:")
                print(f"      目标存在: {'是' if target_exists else '否'}")
                if target_found_frames:
                    print(f"      发现帧数: {len(target_found_frames)}")
                print(f"      YOLO模型: {initial_state['model_in_use']}")
                print(f"      CLIP阈值: {params.get('clip_thresh', 0.25)}")
                print(f"      桥接参数: {params.get('max_bridge', 30)}")
                print(f"      跟踪缓冲: {params.get('track_buffer', 30)}")
                print(f"      置信度: {confidence}")
            else:
                print(f"   ⚠️ 缺少推荐参数，使用默认参数")
        else:
            print(f"   ⚠️ MLLM分析失败，使用默认参数")
            
    except Exception as e:
        print(f"   ❌ MLLM分析异常: {e}")
        print(f"   使用默认参数继续...")
    
    print(f"\n📹 使用MLLM参数启动原始系统...")
    
    # 创建临时配置文件，使用MLLM参数
    import tempfile
    import yaml
    
    temp_config = {
        "prompt": args.text,
        "default_model": initial_state.get("model_in_use", "yolo11n"),
        "clip": {
            "threshold": initial_state.get("clip_thresh", 0.25)
        },
        "bridge": {
            "default": initial_state.get("max_bridge", 30)
        },
        "detector": {
            "local": {
                "weights": "yolo11n.pt",
                "device": "cpu",
                "imgsz": 640
            }
        }
    }
    
    # 写入临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(temp_config, f, default_flow_style=False)
        temp_config_path = f.name
    
    # 调用原始系统
    import subprocess
    import sys
    
    cmd = [
        sys.executable, "agent_main.py",
        "--video", args.video,
        "--config", temp_config_path,
        "--text", args.text,
        "--output_dir", args.output_dir
    ]
    
    print(f"   执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    # 清理临时文件
    import os
    os.unlink(temp_config_path)
    cap.release()
    
    # 显示最终结果
    print("\n" + "="*60)
    print("📊 MLLM参数决策结果")
    print("="*60)
    
    if "mllm_confidence" in initial_state:
        print(f"🤖 MLLM置信度: {initial_state.get('mllm_confidence', 0.0)}")
        print(f"📝 决策理由: {initial_state.get('mllm_reasoning', 'N/A')}")
        print(f"\n📋 最终使用参数:")
        print(f"   YOLO模型: {initial_state.get('model_in_use', 'N/A')}")
        print(f"   CLIP阈值: {initial_state.get('clip_thresh', 'N/A')}")
        print(f"   桥接参数: {initial_state.get('max_bridge', 'N/A')}")
        print(f"   间隔参数: {initial_state.get('gap', 'N/A')}")
        print(f"   跟踪缓冲: {initial_state.get('track_buffer', 'N/A')}")
        print(f"   匹配阈值: {initial_state.get('match_thresh', 'N/A')}")
        print(f"   跟踪阈值: {initial_state.get('track_thresh', 'N/A')}")
    
    # 显示跟踪统计
    gid_frames = initial_state.get("gid_frames", {})
    if gid_frames:
        print(f"\n🎯 跟踪统计:")
        for gid, frames in gid_frames.items():
            print(f"   Global ID {gid}: {len(frames)} 帧")
    
    # 显示导出结果
    exported_path = initial_state.get("exported_video_path")
    if exported_path:
        print(f"\n✅ 视频导出成功: {exported_path}")
    else:
        export_error = initial_state.get("export_error")
        if export_error:
            print(f"\n❌ 导出失败: {export_error}")
    
    print("\n🎉 MLLM增强处理完成!")

if __name__ == "__main__":
    main()
