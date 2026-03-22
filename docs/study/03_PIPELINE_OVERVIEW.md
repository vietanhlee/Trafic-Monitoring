# 03 - Tong Quan Pipeline OCR Plate

## Luong image mode
1. Parse args
2. Load ONNX sessions
3. Chay AnnotateFrame / InferFrameOverlay
4. Draw overlay
5. Save/show output

## Luong video mode
1. Open video stream
2. (Tuy chon) chon ROI polygon va gate line
3. Frame loop
4. Moi N frame chay infer, frame giua dung cached overlay
5. Tracking + identity update theo frame
6. Render + writer/display queue

## Luong infer trong pipeline
- Vehicle detector tim xe
- Cat crop theo tung xe
- Brand classifier cho crop xe (chu yeu class car)
- Plate detector tim bien trong crop xe
- OCR batch doc text bien
- Post-process text + confidence
- Tracking va identity store gop ket qua nhieu frame
- Draw overlays

## Why identity store?
- OCR 1 frame co the sai.
- Tich luy ket qua theo track id de on dinh hon.

## Files trung tam
- src/app/main.cpp
- src/pipeline/frame_annotator.cpp
- src/services/yolo_detector.cpp
- src/services/ocr_batch.cpp
- src/services/brand_classifier.cpp
- src/tracking/byte_track_tracker.cpp
- src/tracking/vehicle_identity_store.cpp
