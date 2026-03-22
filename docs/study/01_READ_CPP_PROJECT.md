# 01 - Cach Doc Mot Du An C++ Co He Thong

Muc tieu: khong bi lo khi vao codebase lon.

## Thu tu doc nhanh (ngoai vao trong)
1. Build system
- Mo CMakeLists.txt de biet target nao duoc build that.
- Phan biet executable va library.

2. Entrypoint
- Mo src/app/main.cpp (va src/app/benchmark.cpp neu can).
- Tim flow tong: parse args -> load session -> loop xu ly.

3. Public API truoc implementation
- Doc include/ocrplate truoc src/.
- Header cho biet input/output va hop dong ham.

4. Module boundaries
- app: orchestration + I/O
- pipeline: ghep cac nhanh xu ly
- services: infer/preprocess/postprocess
- tracking: track id + identity state
- utils: helper dung chung

5. Sau cung moi xuong .cpp
- Khi da nam hop dong, implementation se de doc hon.

## Map nhanh theo project nay
- include/ocrplate: public API
- src/app: main loop + CLI + benchmark
- src/pipeline: frame level orchestration
- src/services: model execution and decode
- src/tracking: ByteTrack + identity store
- src/utils: processing helpers

## Thu tu hoc 5 buoi de de vao dau
1. CMakeLists.txt -> src/app/main.cpp -> include/ocrplate/pipeline/frame_annotator.h
2. src/pipeline/frame_annotator.cpp
3. src/services/ocr_batch.cpp + src/services/yolo_*.cpp + src/services/brand_classifier.cpp
4. src/tracking/byte_track_tracker.cpp + src/tracking/vehicle_identity_store.cpp
5. src/app/benchmark.cpp va toi uu threshold theo du lieu that

## Rule vang
Khong doc dong nao ma chua tra loi duoc 2 cau hoi:
- Dong nay nam trong module nao?
- Dong nay phuc vu use-case nao trong pipeline?
