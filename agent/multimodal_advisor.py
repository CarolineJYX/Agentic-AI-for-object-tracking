#!/usr/bin/env python3
"""
多模态LLM参数顾问系统
用于分析视频内容并推荐最优的跟踪参数

作者: Caroline Xia
用途: 学术研究 - Agentic AI视频目标跟踪系统
"""

import cv2
import json
import base64
import requests
from typing import List, Dict, Any, Tuple
import numpy as np
from pathlib import Path
from datetime import datetime

class MultiModalAdvisor:
    """多模态LLM参数顾问"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.xinyun.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def extract_key_frames(self, video_path: str, n_frames: int = 5) -> List[np.ndarray]:
        """提取视频关键帧"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 均匀分布提取关键帧
        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def get_video_stats(self, video_path: str) -> Dict[str, Any]:
        """获取视频基础统计信息"""
        cap = cv2.VideoCapture(video_path)
        
        stats = {
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        
        cap.release()
        return stats
    
    def frame_to_base64(self, frame: np.ndarray) -> str:
        """将帧转换为base64编码"""
        # 调整图片大小以减少API调用成本
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = 800
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    
    def analyze_video_complexity(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """分析视频复杂度特征"""
        complexity_metrics = {
            "motion_intensity": 0.0,
            "background_complexity": 0.0,
            "brightness_variation": 0.0,
            "edge_density": 0.0
        }
        
        if len(frames) < 2:
            return complexity_metrics
        
        # 计算帧间差异（运动强度）
        motion_scores = []
        for i in range(1, len(frames)):
            gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            motion_scores.append(np.mean(diff))
        
        complexity_metrics["motion_intensity"] = np.mean(motion_scores) / 255.0
        
        # 计算背景复杂度（边缘密度）
        edge_scores = []
        brightness_scores = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 边缘密度
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            edge_scores.append(edge_density)
            
            # 亮度变化
            brightness_scores.append(np.std(gray) / 255.0)
        
        complexity_metrics["edge_density"] = np.mean(edge_scores)
        complexity_metrics["background_complexity"] = np.mean(edge_scores)
        complexity_metrics["brightness_variation"] = np.mean(brightness_scores)
        
        return complexity_metrics
    
    def call_multimodal_llm(self, frames: List[np.ndarray], prompt: str, video_stats: Dict, complexity: Dict) -> Dict[str, Any]:
        """调用多模态LLM分析视频并推荐参数"""
        
        # 准备图片数据
        image_contents = []
        for i, frame in enumerate(frames[:3]):  # 限制为3帧以控制成本
            base64_image = self.frame_to_base64(frame)
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        # 构建LLM提示
        llm_prompt = f"""
你是一个专业的视频分析专家，专门为视频目标跟踪系统推荐最优参数。

## 任务描述
跟踪目标: {prompt}

## 视频信息
- 时长: {video_stats['duration']:.1f}秒
- 帧率: {video_stats['fps']:.1f} FPS  
- 分辨率: {video_stats['width']}x{video_stats['height']}
- 总帧数: {video_stats['total_frames']}

## 计算机视觉分析
- 运动强度: {complexity['motion_intensity']:.3f}
- 背景复杂度: {complexity['background_complexity']:.3f}
- 亮度变化: {complexity['brightness_variation']:.3f}
- 边缘密度: {complexity['edge_density']:.3f}

## 参数说明
请基于视频内容分析，为以下参数推荐最优值：

1. **clip_thresh** (0.15-0.35): CLIP语义匹配阈值
   - 越低越宽松，容易匹配但可能误匹配
   - 越高越严格，准确但可能漏检

2. **max_bridge** (20-80): 片段桥接最大间隔（帧数）
   - 用于连接被短暂遮挡的目标片段
   - 运动连续性好可以设大一些

3. **gap** (1-5): 帧内连续性间隔
   - 同一片段内允许的最大检测间隔
   - 检测稳定可以设大一些

4. **track_buffer** (15-50): 跟踪缓冲区大小
   - 目标消失后保持多久的跟踪记忆
   - 遮挡频繁需要设大一些

5. **match_thresh** (0.6-0.9): ByteTrack匹配阈值
   - 新检测与已有轨迹的匹配严格程度

6. **track_thresh** (0.2-0.5): 跟踪置信度阈值
   - 低于此值的检测不参与跟踪

## 分析要求
请仔细观察提供的关键帧，按以下步骤进行分析：

**第一步：目标存在性检测**
- 仔细观察每一帧，判断描述的目标是否真实存在于视频中
- 如果存在，标记在哪些帧中发现了目标
- 评估目标描述与实际内容的匹配程度

**第二步：目标特征分析**（仅当目标存在时）
1. **目标特征**: 目标对象的清晰度、颜色、形状特征是否明显？
2. **相似干扰**: 场景中是否有与目标相似的其他对象？
3. **运动模式**: 目标运动是否平滑连续？是否有急剧变化？
4. **遮挡情况**: 是否存在遮挡？遮挡持续时间如何？
5. **背景干扰**: 背景是否复杂？是否影响目标检测？
6. **光照条件**: 光照是否稳定？是否有明暗变化？

请以JSON格式输出分析结果：

```json
{{
    "target_exists": true/false,
    "target_found_frames": [1,2,3],
    "recommended_params": {{
        "clip_thresh": 0.xx,
        "max_bridge": xx,
        "gap": x,
        "track_buffer": xx,
        "match_thresh": 0.xx,
        "track_thresh": 0.xx
    }},
    "analysis": {{
        "target_description_match": "目标描述与视频内容的匹配情况",
        "target_clarity": "目标清晰度分析",
        "scene_complexity": "场景复杂度分析", 
        "motion_pattern": "运动模式分析",
        "occlusion_assessment": "遮挡情况评估",
        "tracking_difficulty": "跟踪难度评估 (easy/medium/hard)",
        "parameter_reasoning": "参数选择详细理由"
    }},
    "confidence": 0.xx
}}
```

**重要提醒**：
- 如果target_exists为false，请在analysis中详细说明未找到目标的原因
- 只有当目标确实存在时，才需要推荐跟踪参数
- 请仔细观察每一帧的细节，不要匆忙下结论
"""

        # 构建API请求
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": llm_prompt}
                ] + image_contents
            }
        ]
        
        payload = {
            "model": "gemini-1.5-flash",  # 使用更稳定的flash版本
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        try:
            print(f"   发送API请求到: {self.base_url}/chat/completions")
            print(f"   使用模型: {payload['model']}")
            print(f"   图片数量: {len(image_contents)}")
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            print(f"   API响应状态: {response.status_code}")
            
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.text[:500]}"
                print(f"   ❌ API错误: {error_msg}")
                return {"error": error_msg}
            
            result = response.json()
            llm_response = result['choices'][0]['message']['content']
            
            print(f"   ✅ API调用成功，响应长度: {len(llm_response)} 字符")
            
            # 尝试解析JSON响应
            try:
                # 提取JSON部分
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = llm_response[json_start:json_end]
                    parsed_json = json.loads(json_str)
                    print(f"   ✅ JSON解析成功")
                    
                    # 验证并添加默认值
                    if "target_exists" not in parsed_json:
                        print(f"   ⚠️ 缺少target_exists字段，默认设为true")
                        parsed_json["target_exists"] = True
                    
                    if "target_found_frames" not in parsed_json:
                        parsed_json["target_found_frames"] = []
                    
                    # 显示目标检测结果
                    target_exists = parsed_json.get("target_exists", True)
                    if target_exists:
                        found_frames = parsed_json.get("target_found_frames", [])
                        print(f"   🎯 目标检测: 存在 (发现于帧: {found_frames})")
                    else:
                        print(f"   ⚠️ 目标检测: 不存在")
                        reason = parsed_json.get("analysis", {}).get("target_description_match", "未提供原因")
                        print(f"   📝 原因: {reason}")
                    
                    return parsed_json
            except Exception as parse_error:
                print(f"   ⚠️ JSON解析失败: {str(parse_error)}")
                print(f"   原始响应: {llm_response[:300]}...")
            
            # 如果JSON解析失败，返回原始响应
            return {"raw_response": llm_response, "error": "JSON parsing failed"}
            
        except requests.exceptions.RequestException as e:
            error_msg = f"网络请求失败: {str(e)}"
            print(f"   ❌ {error_msg}")
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"API调用异常: {str(e)}"
            print(f"   ❌ {error_msg}")
            return {"error": error_msg}
    
    def analyze_video(self, video_path: str, prompt: str) -> Dict[str, Any]:
        """完整的视频分析流程"""
        print(f"🎬 开始分析视频: {video_path}")
        print(f"🎯 跟踪目标: {prompt}")
        
        # 1. 提取关键帧
        print("📸 提取关键帧...")
        frames = self.extract_key_frames(video_path)
        print(f"   提取了 {len(frames)} 个关键帧")
        
        # 2. 获取视频统计
        print("📊 分析视频统计信息...")
        video_stats = self.get_video_stats(video_path)
        print(f"   时长: {video_stats['duration']:.1f}s, 帧率: {video_stats['fps']:.1f} FPS")
        
        # 3. 计算复杂度指标
        print("🧮 计算复杂度指标...")
        complexity = self.analyze_video_complexity(frames)
        
        # 4. 调用LLM分析
        print("🤖 调用多模态LLM分析...")
        llm_result = self.call_multimodal_llm(frames, prompt, video_stats, complexity)
        
        # 5. 整合结果
        result = {
            "video_path": video_path,
            "prompt": prompt,
            "video_stats": video_stats,
            "complexity_metrics": complexity,
            "llm_analysis": llm_result,
            "timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else "unknown"
        }
        
        return result

def test_multiple_videos():
    """测试多个demo视频"""
    # API配置 - 直接写入代码用于测试
    API_KEY = "sk-ZKv84hrnoNgqB8bXy6qrIWga9ak4zHI1M7RcByXCLdRcoG1o"
    BASE_URL = "https://api.xinyun.ai/v1"
    
    print(f"🔑 使用API Key: {API_KEY[:20]}...")
    print(f"🌐 API地址: {BASE_URL}")
    
    # 初始化顾问
    advisor = MultiModalAdvisor(API_KEY, BASE_URL)
    
    # 定义测试场景
    test_scenarios = [
        {
            "video_path": "demo.mp4",
            "prompt": "track the brown teddy dog",
            "scenario_name": "Demo1: 棕色泰迪犬"
        },
        {
            "video_path": "demo1.mp4", 
            "prompt": "track the person in dark clothing",
            "scenario_name": "Demo1: 深色衣服的人"
        },
        {
            "video_path": "demo2.mp4",
            "prompt": "track the woman in white clothes dancing",  # 根据您的描述更新
            "scenario_name": "Demo2: 白衣跳舞女性"
        },
        {
            "video_path": "demo3.mp4",
            "prompt": "track the man in white shirt",  # 根据您的描述更新
            "scenario_name": "Demo3: 白衣男性"
        }
    ]
    
    results = []
    
    print(f"\n🚀 开始测试 {len(test_scenarios)} 个视频场景")
    print("="*80)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🔄 进度: {i}/{len(test_scenarios)}")
        print(f"🎬 场景: {scenario['scenario_name']}")
        print(f"📹 视频: {scenario['video_path']}")
        print(f"🎯 目标: {scenario['prompt']}")
        
        if not Path(scenario['video_path']).exists():
            print(f"❌ 视频文件不存在，跳过: {scenario['video_path']}")
            continue
        
        # 执行分析
        result = advisor.analyze_video(scenario['video_path'], scenario['prompt'])
        result['scenario_name'] = scenario['scenario_name']
        results.append(result)
        
        # 显示简要结果
        display_brief_result(result, scenario['scenario_name'])
        
        print("-" * 60)
    
    # 生成对比报告
    if results:
        generate_comparison_report(results)
    
    return results

def display_brief_result(result: dict, scenario_name: str):
    """显示简要结果"""
    llm_analysis = result.get('llm_analysis', {})
    
    if "error" in llm_analysis:
        print(f"❌ 分析失败: {llm_analysis['error']}")
        return
    
    # 显示推荐参数
    if "recommended_params" in llm_analysis:
        params = llm_analysis["recommended_params"]
        print(f"📋 推荐参数:")
        print(f"   clip_thresh: {params.get('clip_thresh', 'N/A')}")
        print(f"   max_bridge: {params.get('max_bridge', 'N/A')}")
        print(f"   track_buffer: {params.get('track_buffer', 'N/A')}")
    
    # 显示置信度
    confidence = llm_analysis.get('confidence', 0)
    print(f"📊 置信度: {confidence}")

def generate_comparison_report(results: list):
    """生成多视频对比报告"""
    print(f"\n{'='*80}")
    print("📊 多视频参数对比报告")
    print('='*80)
    
    successful_results = [r for r in results if "error" not in r.get('llm_analysis', {})]
    
    if not successful_results:
        print("❌ 没有成功的分析结果")
        return
    
    print(f"✅ 成功分析: {len(successful_results)}/{len(results)} 个视频")
    
    # 创建对比表格
    print(f"\n📋 参数对比表:")
    print(f"{'场景':<20} {'clip_thresh':<12} {'max_bridge':<12} {'track_buffer':<13} {'置信度':<8}")
    print("-" * 70)
    
    for result in successful_results:
        scenario = result.get('scenario_name', 'Unknown')[:18]
        llm_analysis = result.get('llm_analysis', {})
        
        if "recommended_params" in llm_analysis:
            params = llm_analysis["recommended_params"]
            confidence = llm_analysis.get('confidence', 0)
            
            clip_thresh = params.get('clip_thresh', 'N/A')
            max_bridge = params.get('max_bridge', 'N/A')
            track_buffer = params.get('track_buffer', 'N/A')
            
            print(f"{scenario:<20} {clip_thresh:<12} {max_bridge:<12} {track_buffer:<13} {confidence:<8}")
    
    # 分析趋势
    print(f"\n🔍 参数趋势分析:")
    
    param_values = {'clip_thresh': [], 'max_bridge': [], 'track_buffer': []}
    
    for result in successful_results:
        llm_analysis = result.get('llm_analysis', {})
        if "recommended_params" in llm_analysis:
            params = llm_analysis["recommended_params"]
            for param in param_values:
                value = params.get(param)
                if value is not None and isinstance(value, (int, float)):
                    param_values[param].append(value)
    
    for param, values in param_values.items():
        if values:
            avg_val = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            print(f"   {param}: 平均={avg_val:.3f}, 范围=[{min_val:.3f}, {max_val:.3f}]")

def main():
    """主函数 - 选择单视频测试还是多视频测试"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--multi":
        # 多视频测试
        results = test_multiple_videos()
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if 'datetime' in globals() else "test"
        output_file = f"multi_video_analysis_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 完整结果已保存到: {output_file}")
        
    else:
        # 单视频测试（保持原有功能）
        single_video_test()

def single_video_test():
    """单视频测试（原有功能）"""
    # API配置 - 直接写入代码用于测试
    API_KEY = "sk-ZKv84hrnoNgqB8bXy6qrIWga9ak4zHI1M7RcByXCLdRcoG1o"
    BASE_URL = "https://api.xinyun.ai/v1"
    
    # 初始化顾问
    advisor = MultiModalAdvisor(API_KEY, BASE_URL)
    
    # 分析demo视频
    video_path = "demo.mp4"  # 您的视频路径
    prompt = "track the brown teddy dog"  # 您的跟踪目标
    
    if not Path(video_path).exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    # 执行分析
    result = advisor.analyze_video(video_path, prompt)
    
    # 输出结果
    print("\n" + "="*60)
    print("🎯 多模态LLM参数建议")
    print("="*60)
    
    if "error" in result["llm_analysis"]:
        print(f"❌ 分析失败: {result['llm_analysis']['error']}")
        return
    
    # 显示推荐参数
    if "recommended_params" in result["llm_analysis"]:
        params = result["llm_analysis"]["recommended_params"]
        print("\n📋 推荐参数:")
        for key, value in params.items():
            print(f"   {key}: {value}")
    
    # 显示分析结果
    if "analysis" in result["llm_analysis"]:
        analysis = result["llm_analysis"]["analysis"]
        print("\n🔍 分析结果:")
        for key, value in analysis.items():
            print(f"   {key}: {value}")
    
    # 显示置信度
    if "confidence" in result["llm_analysis"]:
        confidence = result["llm_analysis"]["confidence"]
        print(f"\n📊 分析置信度: {confidence}")
    
    # 对比当前使用的参数
    print("\n" + "="*60)
    print("📊 参数对比")
    print("="*60)
    
    current_params = {
        "clip_thresh": 0.25,
        "max_bridge": 50,  # 您当前优化后的值
        "gap": 2,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "track_thresh": 0.3
    }
    
    if "recommended_params" in result["llm_analysis"]:
        recommended = result["llm_analysis"]["recommended_params"]
        print(f"{'参数':<15} {'当前值':<10} {'建议值':<10} {'差异':<10}")
        print("-" * 50)
        
        for param in current_params:
            current = current_params[param]
            suggested = recommended.get(param, "N/A")
            
            if suggested != "N/A":
                diff = f"{suggested - current:+.3f}" if isinstance(suggested, (int, float)) else "N/A"
            else:
                diff = "N/A"
                
            print(f"{param:<15} {current:<10} {suggested:<10} {diff:<10}")
    
    # 保存完整结果
    output_file = "multimodal_analysis_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 完整分析结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
