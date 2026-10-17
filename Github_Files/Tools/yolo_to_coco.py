import os
import json
import shutil
from PIL import Image
from tqdm import tqdm

YOLO_ROOT = "/home/hluma/penguin_dataset_final"
COCO_ROOT = "/home/hluma/penguin_coco_final"

categories = [
    {"id": 1, "name": "penguin"}
]

os.makedirs(COCO_ROOT, exist_ok=True)

for split in ["train", "valid", "test"]:

    print(f"\nConverting {split}...")

    img_dir = os.path.join(YOLO_ROOT, split, "images")
    lbl_dir = os.path.join(YOLO_ROOT, split, "labels")

    out_img_dir = os.path.join(COCO_ROOT, split, "images")
    os.makedirs(out_img_dir, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    image_id = 1
    annotation_id = 1
    skipped = 0

    image_files = sorted(os.listdir(img_dir))

    for image_file in tqdm(image_files):

        if not image_file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(img_dir, image_file)

        width, height = Image.open(image_path).size

        coco["images"].append({
            "id": image_id,
            "file_name": image_file,
            "width": width,
            "height": height,
        })

        label_path = os.path.join(
            lbl_dir,
            os.path.splitext(image_file)[0] + ".txt"
        )

        if os.path.exists(label_path):

            with open(label_path) as f:

                for line in f:

                    values = list(map(float, line.strip().split()))

                    if len(values) < 5:
                        skipped += 1
                        continue

                    cls = int(values[0])

                    # ---------------------------
                    # Standard YOLO detection
                    # class xc yc w h
                    # ---------------------------
                    if len(values) == 5:

                        _, xc, yc, bw, bh = values

                    # ---------------------------
                    # YOLO segmentation
                    # class x1 y1 x2 y2 ...
                    # Convert polygon to bbox
                    # ---------------------------
                    else:

                        coords = values[1:]

                        xs = coords[0::2]
                        ys = coords[1::2]

                        xmin = min(xs)
                        xmax = max(xs)

                        ymin = min(ys)
                        ymax = max(ys)

                        xc = (xmin + xmax) / 2
                        yc = (ymin + ymax) / 2

                        bw = xmax - xmin
                        bh = ymax - ymin

                    # Convert YOLO normalized bbox
                    x = (xc - bw / 2) * width
                    y = (yc - bh / 2) * height

                    bw_pixels = bw * width
                    bh_pixels = bh * height

                    coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [
                            x,
                            y,
                            bw_pixels,
                            bh_pixels
                        ],
                        "area": bw_pixels * bh_pixels,
                        "iscrowd": 0
                    })

                    annotation_id += 1

        shutil.copy2(
            image_path,
            os.path.join(out_img_dir, image_file)
        )

        image_id += 1

    annotation_dir = os.path.join(
        COCO_ROOT,
        split,
        "annotations"
    )

    os.makedirs(annotation_dir, exist_ok=True)

    annotation_file = os.path.join(
        annotation_dir,
        f"instances_{split}.json"
    )

    with open(annotation_file, "w") as f:
        json.dump(coco, f)

    print(
        f"{split}: "
        f"{len(coco['images'])} images, "
        f"{len(coco['annotations'])} annotations"
    )

print("\nConversion complete.")
print("Saved to:", COCO_ROOT)
print("Any malformed labels skipped automatically.")

