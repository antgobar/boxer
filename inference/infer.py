import os
from io import BytesIO
from typing import TypedDict

import torch
from PIL import Image
from pydantic import BaseModel
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
from models import BoundingBox


class InferenceResult(TypedDict):
    scores: torch.Tensor
    labels: torch.Tensor
    boxes: torch.Tensor


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
        results: InferenceResult = self.processor.post_process_object_detection(
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
