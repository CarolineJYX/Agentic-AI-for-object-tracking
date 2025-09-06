#!/usr/bin/env python3
"""
测试单个对象视频导出功能
"""
import os
import sys

def test_single_export():
    """测试只导出匹配prompt的单个对象视频"""
    print("=== 测试单个对象视频导出功能 ===")
    
    # 检查demo.mp4是否存在
    if not os.path.exists("demo.mp4"):
        print("[ERROR] demo.mp4 不存在，请确保测试视频文件存在")
        return False
    
    # 测试不同的prompt
    test_cases = [
        ("track the person", "person"),
        ("follow the red car", "car"),
        ("find the yellow dog", "dog"),
    ]
    
    for prompt, expected_type in test_cases:
        print(f"\n--- 测试: {prompt} ---")
        
        # 清理之前的输出
        output_dir = f"./test_output_{expected_type}"
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        
        # 运行主程序
        cmd = f"python agent_main.py --video demo.mp4 --text '{prompt}' --output_dir {output_dir}"
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
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                print(f"\n=== 输出文件 ===")
                for f in files:
                    print(f"  {f}")
                
                # 检查是否只有一个视频文件
                video_files = [f for f in files if f.endswith('.mp4')]
                if len(video_files) == 1:
                    print(f"✅ 成功：只导出了一个视频文件 {video_files[0]}")
                else:
                    print(f"❌ 失败：导出了 {len(video_files)} 个视频文件，期望1个")
                    return False
            else:
                print("[ERROR] 输出目录不存在")
                return False
                
        except Exception as e:
            print(f"[ERROR] 执行失败: {e}")
            return False
    
    print("\n✅ 所有测试通过！单个对象视频导出功能正常工作")
    return True

if __name__ == "__main__":
    success = test_single_export()
    if success:
        print("\n✅ 测试成功！只导出匹配prompt的单个对象视频")
    else:
        print("\n❌ 测试失败！请检查错误信息")

