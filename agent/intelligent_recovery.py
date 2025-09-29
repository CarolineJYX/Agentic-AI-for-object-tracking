#!/usr/bin/env python3
"""
🧠 Intelligent Recovery Agent
智能错误恢复和自适应决策模块
"""

from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

class TrackingState(Enum):
    """跟踪状态枚举"""
    EXCELLENT = "excellent"      # 跟踪质量优秀
    GOOD = "good"               # 跟踪质量良好  
    DEGRADED = "degraded"       # 跟踪质量下降
    LOST = "lost"               # 目标丢失
    RECOVERED = "recovered"     # 目标重新找回

@dataclass
class PerformanceMetrics:
    """性能指标"""
    detection_rate: float       # 检测率
    tracking_continuity: float  # 跟踪连续性
    clip_confidence: float      # CLIP置信度
    stability_score: float      # 稳定性得分
    
class IntelligentRecoveryAgent:
    """智能恢复代理"""
    
    def __init__(self):
        self.performance_history: List[PerformanceMetrics] = []
        self.strategy_success_rate: Dict[str, float] = {}
        self.current_state = TrackingState.GOOD
        
    def analyze_performance(self, state: Dict[str, Any]) -> PerformanceMetrics:
        """分析当前性能"""
        
        # 计算检测率
        recent_detections = state.get("recent_detection_count", 0)
        detection_rate = min(recent_detections / 10.0, 1.0)  # 假设10为满分
        
        # 计算跟踪连续性
        missing_gap = state.get("missing_gap", 0)
        tracking_continuity = max(0, 1 - missing_gap / 20.0)  # 20帧内认为连续
        
        # CLIP置信度
        clip_confidence = state.get("avg_conf", 0.0)
        
        # 稳定性得分 (基于ID切换频率)
        id_switches = state.get("id_switch", 0)
        stability_score = max(0, 1 - id_switches / 5.0)  # 5次切换内认为稳定
        
        metrics = PerformanceMetrics(
            detection_rate=detection_rate,
            tracking_continuity=tracking_continuity,
            clip_confidence=clip_confidence,
            stability_score=stability_score
        )
        
        self.performance_history.append(metrics)
        return metrics
    
    def determine_tracking_state(self, metrics: PerformanceMetrics) -> TrackingState:
        """确定跟踪状态（更保守的判断）"""
        
        overall_score = (
            metrics.detection_rate * 0.3 +
            metrics.tracking_continuity * 0.3 +
            metrics.clip_confidence * 0.2 +
            metrics.stability_score * 0.2
        )
        
        # 更保守的阈值，避免频繁触发恢复
        if overall_score >= 0.7:
            return TrackingState.EXCELLENT
        elif overall_score >= 0.4:  # 提高GOOD的下限
            return TrackingState.GOOD
        elif overall_score >= 0.2:  # 提高DEGRADED的下限
            return TrackingState.DEGRADED
        else:
            return TrackingState.LOST
    
    def suggest_recovery_strategy(self, current_state: TrackingState, 
                                 metrics: PerformanceMetrics) -> Dict[str, Any]:
        """建议恢复策略"""
        
        strategy = {
            "action": "maintain",
            "parameter_adjustments": {},
            "reasoning": "Performance is acceptable"
        }
        
        if current_state == TrackingState.LOST:
            strategy = {
                "action": "aggressive_recovery",
                "parameter_adjustments": {
                    "clip_thresh": max(0.15, metrics.clip_confidence - 0.1),
                    "track_buffer": 50,  # 增加缓冲
                    "match_thresh": 0.6,  # 降低匹配要求
                    "max_bridge": 80  # 增加桥接范围
                },
                "reasoning": "Target lost, using aggressive recovery parameters"
            }
            
        elif current_state == TrackingState.DEGRADED:
            if metrics.detection_rate < 0.5:
                # 检测问题
                strategy = {
                    "action": "improve_detection", 
                    "parameter_adjustments": {
                        "track_thresh": max(0.2, metrics.clip_confidence - 0.1),
                        "model_upgrade": True
                    },
                    "reasoning": "Low detection rate, adjusting detection threshold"
                }
            elif metrics.tracking_continuity < 0.5:
                # 跟踪连续性问题
                strategy = {
                    "action": "improve_tracking",
                    "parameter_adjustments": {
                        "track_buffer": 40,
                        "max_bridge": 60,
                        "gap": 3
                    },
                    "reasoning": "Poor tracking continuity, increasing tracking persistence"
                }
                
        return strategy
    
    def learn_from_success(self, strategy_name: str, success: bool):
        """从成功/失败中学习"""
        if strategy_name not in self.strategy_success_rate:
            self.strategy_success_rate[strategy_name] = 0.5  # 初始成功率
        
        # 简单的指数移动平均
        current_rate = self.strategy_success_rate[strategy_name]
        alpha = 0.3  # 学习率
        
        if success:
            self.strategy_success_rate[strategy_name] = current_rate * (1 - alpha) + alpha
        else:
            self.strategy_success_rate[strategy_name] = current_rate * (1 - alpha)
    
    def get_adaptive_parameters(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """获取自适应参数"""
        
        # 1. 分析当前性能
        metrics = self.analyze_performance(state)
        
        # 2. 确定跟踪状态
        tracking_state = self.determine_tracking_state(metrics)
        
        # 3. 更新状态
        previous_state = self.current_state
        self.current_state = tracking_state
        
        # 4. 获取恢复策略
        strategy = self.suggest_recovery_strategy(tracking_state, metrics)
        
        # 5. 基于历史成功率调整
        if len(self.performance_history) > 5:
            recent_trend = self._analyze_trend()
            if recent_trend < 0:  # 性能下降趋势
                strategy["parameter_adjustments"]["track_buffer"] = \
                    strategy["parameter_adjustments"].get("track_buffer", 30) + 10
        
        return {
            "current_state": tracking_state.value,
            "previous_state": previous_state.value,
            "strategy": strategy,
            "confidence": self._calculate_confidence(metrics),
            "adaptive_params": strategy["parameter_adjustments"]
        }
    
    def _analyze_trend(self) -> float:
        """分析性能趋势"""
        if len(self.performance_history) < 3:
            return 0
        
        recent = self.performance_history[-3:]
        scores = [
            (m.detection_rate + m.tracking_continuity + m.clip_confidence + m.stability_score) / 4
            for m in recent
        ]
        
        # 简单线性趋势
        return scores[-1] - scores[0]
    
    def _calculate_confidence(self, metrics: PerformanceMetrics) -> float:
        """计算决策置信度"""
        base_confidence = (
            metrics.detection_rate * 0.3 +
            metrics.tracking_continuity * 0.3 +
            metrics.clip_confidence * 0.2 +
            metrics.stability_score * 0.2
        )
        
        # 基于历史数据调整置信度
        if len(self.performance_history) > 5:
            recent_variance = np.var([
                m.tracking_continuity for m in self.performance_history[-5:]
            ])
            # 方差越小，置信度越高
            confidence_adjustment = max(0, 1 - recent_variance * 2)
            base_confidence *= confidence_adjustment
        
        return min(1.0, base_confidence)

# 集成到现有系统的接口函数
def create_intelligent_recovery_node():
    """创建智能恢复节点"""
    recovery_agent = IntelligentRecoveryAgent()
    
    def intelligent_recovery_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """智能恢复节点函数"""
        
        # 获取自适应参数
        recovery_result = recovery_agent.get_adaptive_parameters(state)
        
        # 更新状态
        state.update({
            "tracking_state": recovery_result["current_state"],
            "recovery_strategy": recovery_result["strategy"],
            "decision_confidence": recovery_result["confidence"],
            "adaptive_adjustments": recovery_result["adaptive_params"]
        })
        
        # 如果有参数调整建议，应用它们
        if recovery_result["adaptive_params"]:
            for param, value in recovery_result["adaptive_params"].items():
                if param in state:
                    state[param] = value
                    
        return state
    
    return intelligent_recovery_node


