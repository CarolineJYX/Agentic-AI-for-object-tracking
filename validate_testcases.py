#!/usr/bin/env python3
"""
Validation test case script - run all video and prompt combinations in testcase
"""

import json
import subprocess
import os
import time
from pathlib import Path

def load_ground_truth():
    """Load ground truth data"""
    try:
        with open("testcase/groundtruth.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load ground truth: {e}")
        return []

def run_single_test(video, prompt, output_dir):
    """Run a single test case"""
    print(f"\n🎬 Running test: {video} + '{prompt}'")
    
    # Create output directory
    case_dir = Path(output_dir) / f"{Path(video).stem}_{prompt.replace(' ', '_')}"
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # Run main.py
    cmd = [
        "python", "agent_main_post_recovery.py",
        "--video", f"testcase/{video}",
        "--text", prompt,
        "--output_dir", str(case_dir)
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
            cwd=os.getcwd()
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Test completed: {duration:.1f} seconds")
            
            # Parse output, display retained frame information
            tracking_info = parse_tracking_output(result.stdout)
            retained_segments = tracking_info.get("retained_segments", [])
            final_frames = tracking_info.get("final_frames", 0)
            
            if retained_segments:
                segments_str = ", ".join([f"{start}-{end}" for start, end in retained_segments])
                print(f"📍 Retained frame segments: {segments_str} (total {final_frames} frames)")
            else:
                print(f"📍 Retained frame segments: none (total {final_frames} frames)")
            
            return {
                "success": True,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            print(f"❌ Test failed: {duration:.1f} seconds")
            return {
                "success": False,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ Execution timeout: {duration:.1f} seconds")
        return {
            "success": False,
            "duration": duration,
            "error": "timeout"
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 Execution exception: {duration:.1f} seconds - {e}")
        return {
            "success": False,
            "duration": duration,
            "error": str(e)
        }

def parse_tracking_output(stdout):
    """Parse tracking output, extract key information"""
    lines = stdout.split('\n')
    
    # Find final results
    final_frames = 0
    coverage = 0.0
    gid_frames = {}
    retained_segments = []
    
    for line in lines:
        if "Detection=" in line and "GID_count=" in line:
            # Extract frame count and GID count
            parts = line.split()
            for part in parts:
                if "Detection=" in part:
                    try:
                        final_frames = int(part.split("=")[1])
                    except:
                        pass
        elif "coverage" in line and "%" in line:
            # Extract coverage
            try:
                coverage_str = line.split("coverage")[1].split("%")[0].strip()
                coverage = float(coverage_str)
            except:
                pass
        elif "[EDIT] frame=" in line and "sample gid=" in line:
            # Extract GID information
            try:
                if "segments=" in line:
                    segments_str = line.split("segments=")[1]
                    # Parse segments format: [(320, 1164)]
                    if "(" in segments_str and ")" in segments_str:
                        # Simple parsing, extract numbers
                        import re
                        numbers = re.findall(r'\d+', segments_str)
                        if len(numbers) >= 2:
                            start_frame = int(numbers[0])
                            end_frame = int(numbers[1])
                            gid_frames[0] = list(range(start_frame, end_frame + 1))
                            retained_segments.append((start_frame, end_frame))
            except:
                pass
        elif "Segments after merge+bridge:" in line:
            # Extract final merged segments
            try:
                segments_str = line.split("Segments after merge+bridge:")[1].strip()
                # Parse format: [(317, 1164)]
                import re
                matches = re.findall(r'\((\d+),\s*(\d+)\)', segments_str)
                retained_segments = [(int(start), int(end)) for start, end in matches]
            except:
                pass
    
    return {
        "final_frames": final_frames,
        "coverage": coverage,
        "gid_frames": gid_frames,
        "retained_segments": retained_segments
    }

def generate_results_document(results, output_dir):
    """Generate result document in ground truth format"""
    
    # Generate JSON format result document
    results_data = []
    
    for case in results:
        video = case["video"]
        prompt = case["prompt"]
        result = case["result"]
        
        if result["success"]:
            retained_segments = result.get("retained_segments", [])
            final_frames = result.get("final_frames", 0)
            coverage = result.get("coverage", 0.0)
            
            # Convert to ground truth format
            case_data = {
                "video": video,
                "prompt": prompt,
                "segments": retained_segments,
                "total_frames": final_frames,
                "coverage_percent": coverage,
                "status": "success"
            }
        else:
            # Failed test cases
            case_data = {
                "video": video,
                "prompt": prompt,
                "segments": [],
                "total_frames": 0,
                "coverage_percent": 0.0,
                "status": "failed",
                "error": result.get("error", "unknown")
            }
        
        results_data.append(case_data)
    
    # Save JSON format
    json_path = output_dir / "tracking_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # Generate Markdown format summary report
    md_path = output_dir / "tracking_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Tracking Test Results Summary\n\n")
        f.write(f"Test time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Statistics
        success_count = sum(1 for case in results_data if case["status"] == "success")
        total_count = len(results_data)
        f.write(f"## Test Statistics\n\n")
        f.write(f"- Total tests: {total_count}\n")
        f.write(f"- Successful: {success_count}\n")
        f.write(f"- Failed: {total_count - success_count}\n")
        f.write(f"- Success rate: {success_count/total_count*100:.1f}%\n\n")
        
        # Detailed results table
        f.write("## Detailed Results\n\n")
        f.write("| Video | Prompt | Retained Segments | Total Frames | Coverage | Status |\n")
        f.write("|------|--------|----------|--------|--------|------|\n")
        
        for case in results_data:
            video = case["video"]
            prompt = case["prompt"]
            segments = case["segments"]
            total_frames = case["total_frames"]
            coverage = case["coverage_percent"]
            status = case["status"]
            
            # Format frame segments
            if segments:
                segments_str = ", ".join([f"[{start}, {end}]" for start, end in segments])
            else:
                segments_str = "None"
            
            # Status icon
            status_icon = "✅" if status == "success" else "❌"
            
            f.write(f"| {video} | {prompt} | {segments_str} | {total_frames} | {coverage:.1f}% | {status_icon} |\n")
        
        f.write("\n")
        
        # Results sorted by coverage
        f.write("## Sorted by Coverage\n\n")
        f.write("| Rank | Video | Prompt | Coverage | Total Frames |\n")
        f.write("|------|------|--------|--------|--------|\n")
        
        sorted_results = sorted([case for case in results_data if case["status"] == "success"], 
                              key=lambda x: x["coverage_percent"], reverse=True)
        
        for i, case in enumerate(sorted_results, 1):
            video = case["video"]
            prompt = case["prompt"]
            coverage = case["coverage_percent"]
            total_frames = case["total_frames"]
            f.write(f"| {i} | {video} | {prompt} | {coverage:.1f}% | {total_frames} |\n")
    
    print(f"📄 Result documents generated:")
    print(f"   📋 JSON format: {json_path}")
    print(f"   📊 Markdown report: {md_path}")

def main():
    """Main function"""
    print("🚀 Starting to run all test cases...")
    
    # Load ground truth
    test_cases = load_ground_truth()
    if not test_cases:
        print("❌ No test cases found")
        return
    
    print(f"📋 Found {len(test_cases)} test cases")
    
    # Create total output directory
    output_dir = Path("./testcase_results")
    output_dir.mkdir(exist_ok=True)
    
    results = []
    success_count = 0
    
    for i, case in enumerate(test_cases, 1):
        video = case["video"]
        prompt = case["prompt"]
        
        print(f"\n{'='*60}")
        print(f"📊 Test {i}/{len(test_cases)}: {video} + '{prompt}'")
        print(f"{'='*60}")
        
        # Run test
        result = run_single_test(video, prompt, output_dir)
        
        # Parse results
        if result["success"]:
            tracking_info = parse_tracking_output(result["stdout"])
            result.update(tracking_info)
            success_count += 1
        
        results.append({
            "video": video,
            "prompt": prompt,
            "result": result
        })
        
        # Save intermediate results
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Output summary
    print(f"\n{'='*60}")
    print(f"📊 Test completion summary")
    print(f"{'='*60}")
    print(f"✅ Successful: {success_count}/{len(test_cases)}")
    print(f"❌ Failed: {len(test_cases) - success_count}/{len(test_cases)}")
    
    # Display results for each test
    for i, case in enumerate(results, 1):
        video = case["video"]
        prompt = case["prompt"]
        result = case["result"]
        
        status = "✅" if result["success"] else "❌"
        duration = result.get("duration", 0)
        final_frames = result.get("final_frames", 0)
        coverage = result.get("coverage", 0.0)
        retained_segments = result.get("retained_segments", [])
        
        print(f"{status} {i:2d}. {video:15s} + '{prompt:20s}' | {duration:6.1f}s | {final_frames:4d} frames | {coverage:5.1f}%")
        
        # Display retained frame segments
        if retained_segments:
            segments_str = ", ".join([f"{start}-{end}" for start, end in retained_segments])
            print(f"    📍 Retained frame segments: {segments_str}")
        else:
            print(f"    📍 Retained frame segments: None")
    
    # Generate result document in ground truth format
    generate_results_document(results, output_dir)
    
    print(f"\n📁 Detailed results saved in: {output_dir}")
    print("🎉 All tests completed!")

if __name__ == "__main__":
    main()