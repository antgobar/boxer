import traceback
from concurrent import futures
from typing import Protocol

import genproto.prediction_pb2
import genproto.prediction_pb2_grpc
import grpc
from infer import Inference
from models import BoundingBox


class RunInference(Protocol):
    def __call__(self, image_bytes: bytes) -> list[BoundingBox]: ...


class ModelServicer(genproto.prediction_pb2_grpc.ModelServicer):
    def __init__(self, run_inference: RunInference):
        self.run_inference = run_inference

    def Predict(self, request, context):
        try:
            # print(f"Metadata = {context.invocation_metadata()}")
            results = self.run_inference(request.payload)
            distinct_labels = set(box.label for box in results)
            print(
                f"Objects: {len(results)}, source: {request.source}, containing {distinct_labels}."
            )
            detections: list[genproto.prediction_pb2.PredictResponse] = [
                genproto.prediction_pb2.PredictResponse(
                    x1=box.xmin,
                    y1=box.ymin,
                    x2=box.xmax,
                    y2=box.ymax,
                    score=box.score,
                    label=box.label,
                )
                for box in results
            ]
        except Exception as e:
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return genproto.prediction_pb2.PredictResponseList()

        return genproto.prediction_pb2.PredictResponseList(items=detections)


def main():
    inference_app = Inference("./model_store")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    genproto.prediction_pb2_grpc.add_ModelServicer_to_server(
        ModelServicer(inference_app.run), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
