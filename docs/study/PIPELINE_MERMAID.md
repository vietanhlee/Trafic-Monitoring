# So Do Mermaid Toan Bo Pipeline

```mermaid
flowchart TD
    A[CLI Args<br/>src/app/cli_args.cpp] --> B[App Main Loop<br/>src/app/main.cpp]
    B --> C{Input Type}

    C -->|Image| D[ProcessOneImage]
    C -->|Video| E[ProcessVideoStream]

    E --> F[Optional ROI Polygon + Gate Line]
    F --> G[Frame Loop]
    G --> H{Infer This Frame?}
    H -->|No| I[Reuse Cached Overlay]
    H -->|Yes| J[InferFrameOverlay<br/>src/pipeline/frame_annotator.cpp]
    I --> K[Draw + Display/Write]
    J --> K

    D --> J

    J --> L[Vehicle Detect<br/>services/yolo_detector.cpp]
    L --> M[Vehicle Crops]

    M --> N[Brand Classify<br/>services/brand_classifier.cpp]
    M --> O[Plate Detect Per Vehicle<br/>utils/plate_parallel.cpp + services/yolo_detector.cpp]

    O --> P[Build Plate Candidates]
    P --> Q[Preprocess Plate For OCR]
    Q --> R[OCR Batch Infer<br/>services/ocr_batch.cpp]
    R --> S[Text Post Process<br/>services/post_process_out_string.cpp]

    L --> T[ByteTrack Update<br/>tracking/byte_track_tracker.cpp]
    T --> U[Vehicle Identity Store Merge<br/>tracking/vehicle_identity_store.cpp]

    N --> U
    S --> U

    U --> V[Build Overlay Objects]
    V --> W[Track Trace Update<br/>src/pipeline/track_trace.cpp]
    W --> X[DrawFrameOverlay]
    X --> K
```

## Ghi chu nhanh
- Infer frame duoc throttle theo cau hinh N frame de giu FPS.
- Brand va plate/OCR la 2 nhanh phu tro cho tung vehicle.
- Identity store dung de on dinh ket qua brand/plate qua nhieu frame.
