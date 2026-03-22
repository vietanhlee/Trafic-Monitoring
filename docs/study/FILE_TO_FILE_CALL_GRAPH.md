# So Do Call Graph File-to-File

So do nay o muc file, tap trung vao luong goi/chay chinh (runtime) va phu thuoc noi bo quan trong.

```mermaid
flowchart LR
    A[src/app/main.cpp] --> B[src/app/cli_args.cpp]
    A --> C[src/pipeline/frame_annotator.cpp]

    C --> D[src/services/yolo_detector.cpp]
    C --> E[src/services/brand_classifier.cpp]
    C --> F[src/services/ocr_batch.cpp]
    C --> G[src/utils/plate_parallel.cpp]
    C --> H[src/pipeline/track_trace.cpp]
    C --> I[src/tracking/byte_track_tracker.cpp]
    C --> J[src/tracking/vehicle_identity_store.cpp]

    G --> D
    G --> K[src/utils/image_preprocess.cpp]
    G --> L[src/utils/parallel_utils.cpp]

    F --> M[src/services/post_process_out_string.cpp]
    F --> N[src/utils/onnx_decode_utils.cpp]
    F --> L

    D --> O[src/services/yolo_preprocess.cpp]
    D --> P[src/services/yolo_postprocess.cpp]
    D --> Q[src/services/yolo_nms.cpp]

    R[src/app/benchmark.cpp] --> C
    R --> D
    R --> E
    R --> F
    R --> G

    S[src/services/onnx_runner.cpp] --> N
```

## Ghi chu pham vi
- Day la call/dependency graph muc file trong luong chinh app/pipeline/services.
- Khong ve tung ham chi tiet de tranh roi.
- Hai file legacy o src/ goc khong dua vao graph chinh vi flow hien tai su dung src/app + src/pipeline.

## Layering nhanh
- App layer: src/app/*.cpp
- Pipeline layer: src/pipeline/*.cpp
- Service layer: src/services/*.cpp
- Tracking layer: src/tracking/*.cpp
- Utility layer: src/utils/*.cpp
