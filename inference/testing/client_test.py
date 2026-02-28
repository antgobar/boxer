import time

import grpc
from PIL import Image
import os
from io import BytesIO
from typing import TypedDict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from pydantic import BaseModel
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

import gen.prediction_pb2 as request_pb2
import gen.prediction_pb2_grpc as prediction_pb2_grpc


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


def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        filenames = [
            "../data/cat.jpg",
            "../data/cats.jpg",
            "../data/face.png",
        ]
        responses = []
        stub = prediction_pb2_grpc.ModelStub(channel)

        for filename in filenames:
            with open(filename, "rb") as f:
                image_bytes = f.read()
            print(f"Sending payload: {(len(image_bytes))} bytes from {filename}")

            request = request_pb2.PredictRequest(
                payload=image_bytes, source=f"file-{filename}"
            )
            try:
                response_list = stub.Predict(
                    request, timeout=10.0, metadata=[("key", "value")]
                )
            except grpc.RpcError as e:
                print(f"RPC failed: {e.code()} - {e.details()}")
                time.sleep(1)
                continue

            boxes = []

            for detection in response_list.items:
                boxes.append(
                    BoundingBox(
                        xmin=round(detection.x1, 3),
                        ymin=round(detection.y1, 3),
                        xmax=round(detection.x2, 3),
                        ymax=round(detection.y2, 3),
                        score=round(detection.score, 3),
                        label=detection.label,
                    )
                )
                print(
                    f"Detection type: {type(detection)}, "
                    f"Detection: label={detection.label}, score={detection.score}, "
                    f"bbox=({detection.x1},{detection.y1},{detection.x2},{detection.y2})"
                )
            responses.append((filename, boxes))
            time.sleep(1)

    for filename, boxes in responses:
        img = Image.open(filename)
        plot_inference_results(img, boxes)


if __name__ == "__main__":
    run()
