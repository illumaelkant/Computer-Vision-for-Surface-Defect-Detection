#!/bin/bash
# run_setup.sh - chạy 1 lần để tạo cấu trúc

mkdir -p data/{raw/{NEU-DET/{images,annotations},custom},processed/{images/{train,val,test},labels/{train,val,test}},augmented}
mkdir -p configs
mkdir -p src/{data,training,evaluation,inference,utils}
mkdir -p api/{routes,middleware}
mkdir -p models/{weights,onnx}
mkdir -p notebooks tests runs

# Tạo __init__.py
touch src/__init__.py src/data/__init__.py src/training/__init__.py
touch src/evaluation/__init__.py src/inference/__init__.py src/utils/__init__.py
touch api/__init__.py api/routes/__init__.py api/middleware/__init__.py

echo "✅ Structure created"