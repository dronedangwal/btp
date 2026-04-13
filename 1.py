from pathlib import Path
import re

root_dir = Path('model_data')

for rec_dir in sorted(root_dir.iterdir()):

    if not rec_dir.is_dir():
        continue

    sec2img = {}

    for img_path in rec_dir.glob("*.png"):
        m = re.search(r"\d+", img_path.name)
        if m:
            sec2img[int(m.group())] = img_path
    
    print(sec2img)