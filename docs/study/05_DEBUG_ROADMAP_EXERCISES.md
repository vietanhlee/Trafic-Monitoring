# 05 - Debug, Roadmap Hoc, Bai Tap

## Checklist debug khi khong hieu code
1. Tim ham public trong header.
2. Tim call site cua ham do.
3. In log shape/type/size trung gian.
4. So assumption voi assert/check.
5. Xac minh threshold/condition co qua chat hoac qua long.

## Pattern gap nhieu
- if (ptr != nullptr): optional feature, tranh crash.
- reserve truoc push_back: giam realloc.
- std::move: giam copy object lon.
- clone trong OpenCV: deep copy khi can giu du lieu doc lap.
- ROI view: nhanh hon clone nhung phu thuoc vong doi frame goc.

## Roadmap hoc C++ cho repo nay
A. Nen tang
- pointer/reference/const correctness
- vector/string/map
- RAII

B. Thuc chien
- move semantics
- smart pointers
- thread/atomic/mutex/condvar
- template + if constexpr

C. Nang cao
- CMake target-based
- boundary C++ <-> C API (ONNX/OpenCV)
- toi uu copy/allocation/scheduling

## Bai tap tren chinh project
1. In log shape/type input-output OCR model.
2. Thu thay kVideoInferEveryNFrames va do FPS/ID switch.
3. Them 1 threshold moi vao app_config va noi vao 1 nhanh xu ly.
4. Ve thong ke so vehicle/plate len frame.

## FAQ ngan
Q: Vi sao dung pointer thay vi bien thuong?
A: Optional parameter, C API, hoac can truyen tham chieu doi tuong co the khong ton tai.

Q: Vi sao dung static_cast nhieu?
A: Convert ro rang, tranh warning va bug convert ngam.

Q: Vi sao cv:: thay vi cv.?
A: cv la namespace, con . la member access cua object.
