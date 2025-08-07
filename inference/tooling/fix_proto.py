filepath = "genproto/prediction_pb2_grpc.py"

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "from . import prediction_pb2 as prediction__pb2",
    "import prediction_pb2 as prediction__pb2",
)

content = content.replace(
    "import prediction_pb2 as prediction__pb2",
    "from . import prediction_pb2 as prediction__pb2",
)

with open(filepath, "w") as f:
    f.write(content)
