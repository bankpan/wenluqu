"""Simple OCR approach using basic PaddleOCR"""
from pathlib import Path
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from collections import defaultdict

def simple_table_ocr(image_path: Path):
    """Use basic OCR to extract table data"""
    # Initialize basic OCR
    ocr = PaddleOCR(lang='ch')
    
    # Read image  
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Run OCR
    result = ocr.predict(img_array)
    
    if not result or len(result) == 0:
        print("No OCR results")
        return []
    
    # Handle new PaddleOCR format (dict with 'rec_texts' and 'rec_polys')
    data = result[0] if isinstance(result, list) else result
    
    if isinstance(data, dict):
        # New format: {'rec_texts': [...], 'rec_polys': [...]}
        texts = data.get('rec_texts', [])
        polys = data.get('rec_polys', [])
        scores = data.get('rec_scores', [1.0] * len(texts))
        
        print(f"Found {len(texts)} text items")
        
        # Combine texts with coordinates
        boxes = []
        for i, (text, poly) in enumerate(zip(texts, polys)):
            # Calculate center from polygon
            x_coords = poly[:, 0]
            y_coords = poly[:, 1]
            x_center = float(np.mean(x_coords))
            y_center = float(np.mean(y_coords))
            
            boxes.append({
                'text': text,
                'x': x_center,
                'y': y_center,
                'conf': scores[i] if i < len(scores) else 1.0
            })
    else:
        print(f"Unexpected data format: {type(data)}")
        return []
    
    print(f"\nProcessed {len(boxes)} text boxes")
    
    # Group by Y coordinate (rows)
    rows = defaultdict(list)
    y_threshold = 50  # increased threshold for table rows
    
    for box in boxes:
        # Find matching row
        matched_y = None
        for existing_y in rows.keys():
            if abs(box['y'] - existing_y) < y_threshold:
                matched_y = existing_y
                break
        
        if matched_y is None:
            matched_y = box['y']
        
        rows[matched_y].append(box)
    
    # Sort rows by Y coordinate
    sorted_rows = sorted(rows.items(), key=lambda x: x[0])
    
    # For each row, sort boxes by X coordinate
    table_rows = []
    for y_pos, row_boxes in sorted_rows:
        sorted_boxes = sorted(row_boxes, key=lambda x: x['x'])
        row_text = [box['text'] for box in sorted_boxes]
        table_rows.append(row_text)
        print(f"Row at y={y_pos:.0f}: {row_text}")
    
    return table_rows

# Test
image_path = Path("test_images/微信图片_20251112160618_717_2.png")
if image_path.exists():
    print(f"Testing simple OCR on: {image_path}\n")
    table_rows = simple_table_ocr(image_path)
    print(f"\n=== Extracted {len(table_rows)} rows ===")
else:
    print(f"Image not found: {image_path}")
