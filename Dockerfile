## Build & run OCR plate (C++/CMake) with OpenCV + bundled ONNX Runtime
## Usage:
##   docker build -t ocr-plate .
##   docker run --rm -v "$PWD/img:/app/img" ocr-plate --image /app/img/51V4579.jpg
##   docker run --rm -v "$PWD/img:/app/img" --entrypoint /app/benchmark ocr-plate --image /app/img/1.jpeg --warmup 3 --runs 10

FROM ubuntu:22.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_BUILD_TYPE=Release

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY . .

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
 && cmake --build build -j"$(nproc)" --target main benchmark


FROM ubuntu:22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libopencv-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Binaries
COPY --from=build /src/out/build/bin/main /app/main
COPY --from=build /src/out/build/bin/benchmark /app/benchmark

# Model + ONNX Runtime shared libraries
COPY --from=build /src/model /app/model
COPY --from=build /src/third_party/onnxruntime/lib /app/third_party/onnxruntime/lib

# Optional sample images (can be overridden with -v $PWD/img:/app/img)
COPY img /app/img

ENV LD_LIBRARY_PATH=/app/third_party/onnxruntime/lib

ENTRYPOINT ["/app/main"]
