import os
from io import BytesIO

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pydantic import BaseModel
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor


class BoundingBox(BaseModel):
    score: float
    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class Inference:
    def __init__(self, dir: str):
        local_model_weights = os.path.join(dir, "pytorch_model.bin")
        local_config_file = os.path.join(dir, "config.json")

        config = DetrConfig.from_pretrained(local_config_file)
        state_dict = torch.load(local_model_weights, map_location="cpu")

        model = DetrForObjectDetection(config)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        self.device: torch.device = device
        self.processor: DetrImageProcessor = DetrImageProcessor.from_pretrained(dir)
        self.model: DetrForObjectDetection = model

    def run(self, image_bytes: bytes) -> list[BoundingBox]:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9
        )[0]

        bounding_boxes: list[BoundingBox] = []

        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box = [round(i, 2) for i in box.tolist()]
            xmin, ymin, xmax, ymax = box
            bounding_boxes.append(
                BoundingBox(
                    score=round(score.item(), 3),
                    label=self.model.config.id2label[label.item()],
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                )
            )

        return bounding_boxes


def plot_inference_results(image: Image.Image, results: list[BoundingBox]) -> None:
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box in results:
        xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax
        width, height = xmax - xmin, ymax - ymin
        ax.add_patch(
            patches.Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
        )
        ax.text(
            xmin,
            ymin,
            f"{box.label}: {box.score}",
            bbox=dict(facecolor="yellow", alpha=0.5),
        )

    plt.axis("off")
    plt.show()
