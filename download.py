#!/usr/bin/env python3
import os
import sys
import re
import yt_dlp
from urllib.parse import urlparse, parse_qs


def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    parsed_url = urlparse(url)
    
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
    elif parsed_url.hostname in ['youtu.be']:
        return parsed_url.path[1:]  # Remove leading slash
    
    # Try to extract from various YouTube URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:v\/)([0-9A-Za-z_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def download_youtube_video(url, output_dir='./videos'):
    """Download YouTube video with specified settings"""
    
    # Extract video ID for filename
    video_id = extract_video_id(url)
    if not video_id:
        print(f"Error: Could not extract video ID from URL: {url}")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output filename
    output_path = os.path.join(output_dir, f"{video_id}.%(ext)s")
    
    # yt-dlp options
    ydl_opts = {
        'format': 'best[height<=720][ext=mp4]/best[height<=720]/worst[ext=mp4]/worst',
        'outtmpl': output_path,
        'noplaylist': True,
        'no_warnings': False,
        'extractaudio': False,  # No audio
        'writesubtitles': False,
        'writeautomaticsub': False,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        # Remove audio stream
        'postprocessor_args': ['-an'],  # -an removes audio
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading video ID: {video_id}")
            print(f"URL: {url}")
            print(f"Output: {output_dir}/{video_id}.mp4")
            
            ydl.download([url])
            print(f"Successfully downloaded: {video_id}.mp4")
            return True
            
    except Exception as e:
        print(f"Error downloading video: {e}")
        return False


def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) != 2:
        print("Usage: python download.py <youtube_url>")
        print("Example: python download.py 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'")
        sys.exit(1)
    
    url = sys.argv[1]
    success = download_youtube_video(url)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
