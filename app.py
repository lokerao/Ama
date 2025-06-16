
import os
import re
import cv2
import numpy as np
import pytesseract
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import yt_dlp
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@dataclass
class GiftCardResult:
    code: str
    timestamp: float
    confidence: float
    frame_number: int

class GiftCardScanner:
    def __init__(self):
        self.gift_card_pattern = re.compile(r'[A-Z0-9]{4}-[A-Z0-9]{6}-[A-Z0-9]{4}')
        
    def preprocess_frame(self, frame):
        """Advanced frame preprocessing for better OCR accuracy"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple preprocessing techniques
        processed_frames = []
        
        # Original grayscale
        processed_frames.append(gray)
        
        # Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        processed_frames.append(enhanced)
        
        # Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        processed_frames.append(sharpened)
        
        # Thresholding
        _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_frames.append(thresh)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed_frames.append(morph)
        
        return processed_frames
    
    def extract_text_from_frame(self, frame):
        """Extract text from a frame using Tesseract with multiple configurations"""
        results = []
        
        # Multiple Tesseract configurations
        configs = [
            '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
            '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
            '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
            '--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        ]
        
        processed_frames = self.preprocess_frame(frame)
        
        for proc_frame in processed_frames:
            for config in configs:
                try:
                    # Scale up frame for better OCR
                    scale = 2
                    height, width = proc_frame.shape
                    scaled = cv2.resize(proc_frame, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
                    
                    text = pytesseract.image_to_string(scaled, config=config)
                    if text.strip():
                        results.append(text.strip())
                except Exception as e:
                    logger.warning(f"OCR error with config {config}: {e}")
                    continue
        
        return results
    
    def find_gift_cards_in_text(self, text_results):
        """Find gift card codes in extracted text"""
        codes = set()
        for text in text_results:
            # Clean text and find matches
            cleaned_text = re.sub(r'[^A-Z0-9-]', '', text.upper())
            matches = self.gift_card_pattern.findall(cleaned_text)
            codes.update(matches)
            
            # Also try with spaces removed and dashes added
            no_spaces = text.replace(' ', '').upper()
            if len(no_spaces) >= 15:  # Minimum length for gift card
                for i in range(len(no_spaces) - 14):
                    candidate = no_spaces[i:i+15]
                    if re.match(r'^[A-Z0-9]{15}$', candidate):
                        formatted = f"{candidate[:4]}-{candidate[4:10]}-{candidate[10:]}"
                        codes.add(formatted)
        
        return list(codes)
    
    def process_frame(self, frame_data):
        """Process a single frame for gift card detection"""
        frame, frame_number, timestamp = frame_data
        
        try:
            text_results = self.extract_text_from_frame(frame)
            gift_cards = self.find_gift_cards_in_text(text_results)
            
            results = []
            for code in gift_cards:
                results.append(GiftCardResult(
                    code=code,
                    timestamp=timestamp,
                    confidence=0.9,  # Placeholder confidence
                    frame_number=frame_number
                ))
            
            return results
        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {e}")
            return []

def get_video_info(url):
    """Get video information using yt-dlp"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'url': url
            }
        except Exception as e:
            raise Exception(f"Could not extract video info: {e}")

def extract_frames_from_video(url, max_frames=3000):
    """Extract frames from YouTube video without downloading"""
    ydl_opts = {
        'format': 'best[height<=720]',  # Limit quality for faster processing
        'quiet': True,
        'no_warnings': True,
    }
    
    frames = []
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            video_url = info['url']
            
            # Use OpenCV to capture frames
            cap = cv2.VideoCapture(video_url)
            
            if not cap.isOpened():
                raise Exception("Could not open video stream")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps * 0.1))  # Sample every 0.1 seconds
            
            frame_count = 0
            extracted_count = 0
            
            while extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    frames.append((frame.copy(), extracted_count, timestamp))
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            
        except Exception as e:
            raise Exception(f"Error extracting frames: {e}")
    
    return frames

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan_video():
    try:
        data = request.get_json()
        video_url = data.get('url', '').strip()
        
        if not video_url:
            return jsonify({'error': 'No video URL provided'}), 400
        
        # Validate YouTube URL
        if 'youtube.com' not in video_url and 'youtu.be' not in video_url:
            return jsonify({'error': 'Please provide a valid YouTube URL'}), 400
        
        logger.info(f"Starting scan for: {video_url}")
        
        # Get video info
        video_info = get_video_info(video_url)
        
        # Check duration (15-25 minutes = 900-1500 seconds)
        duration = video_info.get('duration', 0)
        if duration > 1800:  # 30 minutes max for performance
            return jsonify({'error': 'Video too long. Please use videos under 30 minutes.'}), 400
        
        start_time = time.time()
        
        # Extract frames
        logger.info("Extracting frames...")
        frames = extract_frames_from_video(video_url)
        
        if not frames:
            return jsonify({'error': 'Could not extract frames from video'}), 400
        
        logger.info(f"Extracted {len(frames)} frames")
        
        # Process frames in parallel
        scanner = GiftCardScanner()
        all_results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_frame = {executor.submit(scanner.process_frame, frame_data): i 
                             for i, frame_data in enumerate(frames)}
            
            for future in as_completed(future_to_frame):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Error in parallel processing: {e}")
        
        # Remove duplicates and sort by timestamp
        unique_codes = {}
        for result in all_results:
            if result.code not in unique_codes:
                unique_codes[result.code] = result
        
        final_results = sorted(unique_codes.values(), key=lambda x: x.timestamp)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Scan completed in {processing_time:.2f}s, found {len(final_results)} codes")
        
        return jsonify({
            'success': True,
            'video_info': video_info,
            'processing_time': round(processing_time, 2),
            'codes_found': len(final_results),
            'results': [
                {
                    'code': r.code,
                    'timestamp': round(r.timestamp, 2),
                    'frame_number': r.frame_number,
                    'confidence': r.confidence
                }
                for r in final_results
            ]
        })
        
    except Exception as e:
        logger.error(f"Scan error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
