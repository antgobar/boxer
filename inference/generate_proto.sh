source .venv/bin/activate
python -m grpc_tools.protoc \
  -I ../proto \
  --python_out=./genproto \
  --grpc_python_out=./genproto \
  ../proto/prediction.proto

python tooling/fix_proto.py