import requests
import json

# Step 1: Receive the user's natural language instruction from DeepSeek and return a structured JSON command
def nlp_json_deepseek(user_input: str) -> dict:
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": "YOUR API KEY".encode("utf-8").decode("utf-8"),
        "Content-Type": "application/json; charset=utf-8"
    }

    example_prompt = f"""
You are an intelligent video instruction parser. Convert the user's natural language video instruction into a structured JSON format.

[Example Input]
Please track the yellow dog in the video, use a medium shot, and add upbeat music.

[Output Format] Do not include markdown syntax
{{
  "clip_prompt": "yellow dog",
  "first_yolo_model": "yolo11n",
  "yolo_class": "dog"
}}

  "clip_prompt": "...",             // Object to track
  "first_yolo_model": "yolo11n", // Initial YOLO model to use: yolo11n / yolo11m / yolo11x      
  "yolo_class": "..."         // Optional: one of the following categories
    person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

[User Input]
{user_input}

[Output]
Please return only the JSON format without any additional explanation.
"""

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": example_prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    response_text = result["choices"][0]["message"]["content"]
    
    try:
        parsed_result = json.loads(response_text)
        return parsed_result
    except Exception as e:
        return f"[Error] Failed to parse JSON: {e}, Content: {response_text}"
