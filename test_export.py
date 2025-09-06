#!/usr/bin/env python3
"""
测试视频导出功能
"""
import os
import sys

def test_export():
    """测试视频导出功能"""
    print("=== 测试视频导出功能 ===")
    
    # 检查demo.mp4是否存在
    if not os.path.exists("demo.mp4"):
        print("[ERROR] demo.mp4 不存在，请确保测试视频文件存在")
        return False
    
    # 运行主程序
    cmd = f"python agent_main.py --video demo.mp4 --text 'track the person' --output_dir ./test_output"
    print(f"执行命令: {cmd}")
    
    import subprocess
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print("=== 命令输出 ===")
        print(result.stdout)
        if result.stderr:
            print("=== 错误输出 ===")
            print(result.stderr)
        
        # 检查输出目录
        if os.path.exists("./test_output"):
            files = os.listdir("./test_output")
            print(f"\n=== 输出文件 ===")
            for f in files:
                print(f"  {f}")
            return len(files) > 0
        else:
            print("[ERROR] 输出目录不存在")
            return False
            
    except Exception as e:
        print(f"[ERROR] 执行失败: {e}")
        return False

if __name__ == "__main__":
    success = test_export()
    if success:
        print("\n✅ 测试成功！视频导出功能正常工作")
    else:
        print("\n❌ 测试失败！请检查错误信息")

