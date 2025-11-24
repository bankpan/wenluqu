import sys
from pathlib import Path
from steady_score.config import OCRConfig
from steady_score.logging_utils import setup_file_logger
from steady_score.ocr_pipeline import OCRProcessor
import numpy as np
from PIL import Image

# Setup
image_path = Path("test_images/微信图片_20251112160618_717_2.png")
output_dir = Path("debug_output")
output_dir.mkdir(exist_ok=True)

config = OCRConfig(
    image_dir=Path("test_images"),
    output_dir=output_dir,
    raw_dir=output_dir / "raw",
    log_dir=output_dir / "logs",
    error_file=output_dir / "errors.txt",
    lang="ch",
)

logger = setup_file_logger(config.log_dir)
processor = OCRProcessor(config, logger)

# Process image
with Image.open(image_path) as img:
    rgb = img.convert("RGB")
    table_region = processor._crop_table_region(rgb)
    cropped = processor._auto_crop(table_region)
    prepared_image = np.array(cropped)

# Get raw table rows from all crops
crops = processor._generate_crops(prepared_image)
print(f"Number of crops: {len(crops)}")

for i, crop in enumerate(crops):
    print(f"\n=== Crop {i+1} ===")
    results = processor._run_engine(crop)
    
    # Show raw HTML
    for block in results:
        if block.get("type") == "table":
            html = block.get("res", {}).get("html", "")
            if html:
                print(f"HTML preview (first 500 chars):\n{html[:500]}\n")
    
    # Show extracted rows
    rows = processor._extract_rows(results)
    print(f"Extracted {len(rows)} rows:")
    for row in rows:
        print(f"  {row}")
