import os
from io import BytesIO
from typing import TypedDict

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
