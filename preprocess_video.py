#!/usr/bin/env python3
"""
Video preprocessing script - converts video to H5 format with minimal memory usage.

Usage:
    python preprocess_video.py --video_path videos/video.mp4 --target_fps 12
    
Output will be automatically saved as videos/video.h5
"""

import os
import cv2
import h5py
import numpy as np
import argparse
import hashlib
from tqdm import tqdm


def analyze_video(video_path, target_fps=12, skip_seconds=30.0):
    """Analyze video and determine frame sampling parameters"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / original_fps
    
    # Calculate frame sampling
    target_frame_interval = 1.0 / target_fps
    original_frame_interval = 1.0 / original_fps
    skip_frames = int(skip_seconds * original_fps)
    
    print(f"Video Analysis:")
    print(f"  Path: {video_path}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Original FPS: {original_fps:.2f}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Skip first: {skip_seconds}s ({skip_frames:,} frames)")
    print(f"  Target FPS: {target_fps}")
    
    # Calculate which frame indices to extract
    frame_indices = []
    current_time = 0.0
    next_target_time = 0.0
    frame_index = skip_frames
    
    while frame_index < total_frames:
        if current_time >= next_target_time:
            frame_indices.append(frame_index)
            next_target_time += target_frame_interval
        
        current_time += original_frame_interval
        frame_index += 1
    
    cap.release()
    
    print(f"  Frames to extract: {len(frame_indices):,}")
    print(f"  Effective duration: {len(frame_indices) / target_fps:.2f}s")
    
    return frame_indices, original_fps


def process_frame(frame, target_size=(320, 180)):
    """Process a single frame: resize and normalize"""
    # Resize with nearest neighbor (no antialiasing)
    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Convert to float and normalize to [-1, 1]
    frame_array = rgb_frame.astype(np.float32)
    frame_array = frame_array / 127.5 - 1.0
    
    # Transpose to (C, H, W) format
    frame_array = frame_array.transpose(2, 0, 1)
    
    return frame_array


def preprocess_video(video_path, output_path, target_fps=12, target_size=(320, 180), 
                    skip_seconds=30.0, chunk_size=100):
    """
    Preprocess video to H5 format with minimal memory usage
    
    Args:
        video_path: Path to input video
        output_path: Path to output H5 file
        target_fps: Target frames per second
        target_size: Target frame size (width, height)
        skip_seconds: Seconds to skip at start
        chunk_size: Number of frames to process in each chunk
    """
    
    # Analyze video first
    frame_indices, original_fps = analyze_video(video_path, target_fps, skip_seconds)
    
    if not frame_indices:
        print("No frames to extract!")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize H5 file
    print(f"\nCreating H5 file: {output_path}")
    h, w = target_size[1], target_size[0]  # Height, Width
    
    with h5py.File(output_path, 'w') as f:
        # Create resizable dataset
        frames_dataset = f.create_dataset(
            'frames',
            shape=(0, 3, h, w),
            maxshape=(len(frame_indices), 3, h, w),
            dtype=np.float32,
            compression='gzip',
            compression_opts=1,
            shuffle=True,
            chunks=(min(50, len(frame_indices)), 3, h, w)
        )
        
        # Save metadata
        f.attrs['video_path'] = str(video_path).encode('utf-8')
        f.attrs['target_width'] = w
        f.attrs['target_height'] = h
        f.attrs['target_fps'] = float(target_fps)
        f.attrs['original_fps'] = float(original_fps)
        f.attrs['skip_seconds'] = float(skip_seconds)
        f.attrs['total_frames'] = len(frame_indices)
        
        # Process video in chunks
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        frames_written = 0
        current_chunk = []
        
        with tqdm(total=len(frame_indices), desc="Processing frames", unit="frames") as pbar:
            for frame_idx in frame_indices:
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Process frame
                    processed_frame = process_frame(frame, target_size)
                    current_chunk.append(processed_frame)
                    
                    # Write chunk when full
                    if len(current_chunk) >= chunk_size:
                        # Resize dataset
                        current_size = frames_dataset.shape[0]
                        frames_dataset.resize((current_size + len(current_chunk), 3, h, w))
                        
                        # Write chunk
                        frames_dataset[current_size:current_size + len(current_chunk)] = np.array(current_chunk)
                        frames_written += len(current_chunk)
                        current_chunk = []
                        
                        pbar.update(chunk_size)
                else:
                    print(f"Warning: Could not read frame {frame_idx}")
                    pbar.update(1)
            
            # Write remaining frames
            if current_chunk:
                current_size = frames_dataset.shape[0]
                frames_dataset.resize((current_size + len(current_chunk), 3, h, w))
                frames_dataset[current_size:current_size + len(current_chunk)] = np.array(current_chunk)
                frames_written += len(current_chunk)
                pbar.update(len(current_chunk))
        
        cap.release()
        
        # Update final metadata
        f.attrs['frames_written'] = frames_written
    
    # Verify file
    file_size = os.path.getsize(output_path)
    print(f"\n✅ Preprocessing complete!")
    print(f"  Output: {output_path}")
    print(f"  Frames written: {frames_written:,}")
    print(f"  File size: {file_size / 1024**2:.1f} MB")
    print(f"  Compression ratio: {frames_written * 3 * h * w * 4 / file_size:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Preprocess video to H5 format")
    parser.add_argument("--video_path", required=True, help="Path to input video")
    parser.add_argument("--target_fps", type=float, default=12, help="Target FPS (default: 12)")
    parser.add_argument("--width", type=int, default=320, help="Target width (default: 320)")
    parser.add_argument("--height", type=int, default=180, help="Target height (default: 180)")
    parser.add_argument("--skip_seconds", type=float, default=30.0, help="Seconds to skip at start (default: 30)")
    parser.add_argument("--chunk_size", type=int, default=100, help="Frames per chunk (default: 100)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    # Generate output path by replacing extension with .h5
    video_name = os.path.splitext(args.video_path)[0]
    output_path = video_name + ".h5"
    
    target_size = (args.width, args.height)
    
    preprocess_video(
        video_path=args.video_path,
        output_path=output_path,
        target_fps=args.target_fps,
        target_size=target_size,
        skip_seconds=args.skip_seconds,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()
