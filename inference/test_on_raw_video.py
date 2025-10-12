
import torch
from torch.nn import functional as F
import os
import numpy as np
from test_tools.common import detect_all, grab_all_frames
from test_tools.utils import get_crop_box
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.supply_writer import SupplyWriter
import argparse
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor, Normalize
from model.framework import get_model
import cv2
from PIL import Image

mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255,]).cuda().view(1, 3, 1, 1, 1)
std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255,]).cuda().view(1, 3, 1, 1, 1)

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="FTCN Face Depth Estimation Inference")
    parser.add_argument("--video", type=str, help="Input video path", default="./examples/shining.mp4")
    parser.add_argument("--out_dir", type=str, help="Output directory", default="./examples")
    parser.add_argument("--model_path", type=str, help="Model checkpoint path", default="./model/for_deploy_model3.pth")
    parser.add_argument("--max_frame", type=int, help="Maximum number of frames to process", default=768)
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = get_model()
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    model.cuda()
    print("Model loaded successfully!")

    # Initialize preprocessing functions
    crop_align_func = FasterCropAlignXRay(224)
    
    # Setup input/output paths
    input_file = args.video
    os.makedirs(args.out_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_file))[0] +"_detect.mp4"
    out_file = os.path.join(args.out_dir, basename)

    # Process video frames
    max_frame = args.max_frame
    cache_file = f"{input_file}_{str(max_frame)}.pth"

    if os.path.exists(cache_file):
        print("Loading cached detection results...")
        detect_res, all_lm68 = torch.load(cache_file)
        frames = grab_all_frames(input_file, max_size=max_frame, cvt=True)
        print("Detection results loaded from cache")
    else:
        print("Performing face detection...")
        detect_res, all_lm68, frames = detect_all(
            input_file, return_frames=True, max_size=max_frame
        )
        torch.save((detect_res, all_lm68), cache_file)
        print("Face detection completed")

    print(f"Processing {len(frames)} frames")

    
    # Process detection results
    shape = frames[0].shape[:2]
    all_detect_res = []

    assert len(all_lm68) == len(detect_res)

    for faces, faces_lm68 in zip(detect_res, all_lm68):
        new_faces = []
        for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
            new_face = (box, lm5, face_lm68, score)
            new_faces.append(new_face)
        all_detect_res.append(new_faces)

    detect_res = all_detect_res

    # Track faces across frames
    print("Tracking faces across frames...")
    tracks = multiple_tracking(detect_res)
    tuples = [(0, len(detect_res))] * len(tracks)

    print(f"Found {len(tracks)} face tracks")

    if len(tracks) == 0:
        print("No tracks found, using longest sequence...")
        tuples, tracks = find_longest(detect_res)

    # Extract face crops and landmarks
    data_storage = {}
    frame_boxes = {}
    super_clips = []

    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
        print(f"Processing track {track_i}: frames {start}-{end}")
        assert len(detect_res[start:end]) == len(track)

        super_clips.append(len(track))

        for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
            box, lm5, lm68 = face[:3]
            big_box = get_crop_box(shape, box, scale=0.5)

            top_left = big_box[:2][None, :]

            new_lm5 = lm5 - top_left
            new_lm68 = lm68 - top_left
            new_box = (box.reshape(2, 2) - top_left).reshape(-1)

            info = (new_box, new_lm5, new_lm68, big_box)

            x1, y1, x2, y2 = big_box
            cropped = frames[frame_idx][y1:y2, x1:x2]

            base_key = f"{track_i}_{j}_"
            data_storage[base_key + "img"] = cropped
            data_storage[base_key + "ldm"] = info
            data_storage[base_key + "idx"] = frame_idx

            frame_boxes[frame_idx] = np.rint(box).astype(np.int32)

    print(f"Sampling clips from super clips: {super_clips}")

    # Generate clips for temporal analysis
    clips_for_video = []
    clip_size = 32
    pad_length = clip_size - 1

    for super_clip_idx, super_clip_size in enumerate(super_clips):
        inner_index = list(range(super_clip_size))
        
        if super_clip_size < clip_size:  # Need padding for short sequences
            post_module = inner_index[1:-1][::-1] + inner_index
            l_post = len(post_module)
            post_module = post_module * (pad_length // l_post + 1)
            post_module = post_module[:pad_length]
            
            if len(post_module) != pad_length:
                continue  # Skip sequences that are too short
            
            pre_module = inner_index + inner_index[1:-1][::-1]
            l_pre = len(post_module)
            pre_module = pre_module * (pad_length // l_pre + 1)
            pre_module = pre_module[-pad_length:]
            
            if len(pre_module) != pad_length:
                continue  # Skip sequences that are too short

            inner_index = pre_module + inner_index + post_module

        super_clip_size = len(inner_index)

        # Generate sliding window clips
        frame_range = [
            inner_index[i : i + clip_size] 
            for i in range(super_clip_size) 
            if i + clip_size <= super_clip_size
        ]
        
        for indices in frame_range:
            clip = [(super_clip_idx, t) for t in indices]
            clips_for_video.append(clip)
    
    # Run inference on clips
    preds = []
    frame_res = {}
    test_transform = Compose([
        ToTensor(), 
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Running inference on {len(clips_for_video)} clips...")
    for clip in tqdm(clips_for_video, desc="Processing clips"):
        # Prepare data for this clip
        images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
        landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
        frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]
        
        # Align and crop faces
        landmarks, images = crop_align_func(landmarks, images)
        
        # Convert to temporal frequency domain
        images_tensor = []
        ft_images = []
        for image in images:
            image = np.array(image)
            img_pil = Image.fromarray(image)
            img_pil = test_transform(img_pil)
            images_tensor.append(img_pil)
            
            # Apply median filter and compute frequency domain
            img_filtered = cv2.medianBlur(image.copy(), 5)
            ft_images.append(cv2.cvtColor((image - img_filtered), cv2.COLOR_RGB2GRAY)) 
            
        # Compute FFT for temporal analysis
        ft_images = np.array(ft_images)
        ft_images = np.absolute(np.fft.fft(ft_images, axis=0)[:clip_size//2] * 1/clip_size)
        ft_images = torch.from_numpy(ft_images).cuda()
        ft_images = ft_images.unsqueeze(0)
        
        # Prepare image tensor
        images = torch.stack(images_tensor, dim=1)
        images = images.unsqueeze(0)
        images = images.cuda()
        
        # Run model inference
        with torch.no_grad():
            output = model(images, ft_images)
            output = F.sigmoid(output)
            output = output.squeeze(0)
            
        pred = float(output.item())
        
        # Store predictions for each frame
        for f_id in frame_ids:
            if f_id not in frame_res:
                frame_res[f_id] = []
            frame_res[f_id].append(pred)
        preds.append(pred)


    # Aggregate results
    mean_pred = np.mean(preds)
    print(f"Average prediction score: {mean_pred:.4f}")
    
    # Prepare final results
    boxes = []
    scores = []

    for frame_idx in range(len(frames)):
        if frame_idx in frame_res:
            pred_prob = np.mean(frame_res[frame_idx])
            rect = frame_boxes[frame_idx]
        else:
            pred_prob = None
            rect = None
        scores.append(pred_prob)
        boxes.append(rect)

    # Save results to video
    print(f"Saving results to {out_file}")
    SupplyWriter(args.video, out_file, 0.002584857167676091).run(frames, scores, boxes)
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
