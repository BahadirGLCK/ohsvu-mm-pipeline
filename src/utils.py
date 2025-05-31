import pathlib
import random
import cv2
from PIL import Image
from . import config # Import the config module
import json # Add json for config dumping

# Use constants from the config module
# VIDEO_DIR = pathlib.Path("data/inputs/vlm_videos") # Replaced by config.VIDEO_DIR
# NUM_FRAMES_PER_VIDEO = 25 # Replaced by config.NUM_FRAMES_PER_VIDEO
# SAMPLING_STRATEGY = "uniform" # Replaced by config.SAMPLING_STRATEGY

def sample_frames_from_video(video_path: pathlib.Path, 
                               k: int = config.NUM_FRAMES_PER_VIDEO, 
                               strategy: str = config.SAMPLING_STRATEGY):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        raise ValueError(f"{video_path} has no frames")

    if k <= 0:
        raise ValueError(f"Number of frames to sample (k) must be positive, got {k}")
    if k > frame_count: # If k is larger than available, sample all frames
        k = frame_count

    if strategy == "uniform":
        if k == 0: indices = [] # Should be caught by k<=0 check, but good for safety
        else: step = frame_count / k
        indices = [int(i * step) for i in range(k)]
    elif strategy == "index":
        if k == 1:
            indices = [frame_count // 2] # Middle frame
        elif k == 2:
            indices = [0, frame_count -1] # First and Last
        elif k == 3:
            indices = [0, frame_count // 2, frame_count - 1]
        else: # Fallback to uniform-like for other k with "index"
            indices = [int(frame_count * (i + 1) / (k + 1)) for i in range(k)]
    elif strategy == "random":
        if k > frame_count: k = frame_count # Ensure k is not greater than frame_count for random.sample
        indices = sorted(random.sample(range(frame_count), k))
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_data = cap.read()
        if not ret:
            # print(f"Warning: Could not read frame at index {idx} from {video_path}")
            continue
        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_data))
    cap.release()
    return frames

def row_to_example(row, ohs_prompt):
    # VIDEO_DIR is now sourced from config
    video_path = config.VIDEO_DIR / row["video_name"]
    
    # sample_frames_from_video will use NUM_FRAMES_PER_VIDEO and SAMPLING_STRATEGY from config by default
    frames = sample_frames_from_video(video_path)
    
    content = [{"type": "text", "text": ohs_prompt}]
    content += [{"type": "image", "image": img} for img in frames]
    
    #Make dataframe column names more generalizable.
    messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": row["gemini_answer"]}]},
    ]
    return {"messages": messages, "video_path": str(video_path)} # Store video_path as string for easier serialization if needed 

# Removed save_config_snapshot as it's now in ExperimentManager 