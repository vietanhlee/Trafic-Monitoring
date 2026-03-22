# Huong Dan Hoc C++ Va Doc Du An OCR Plate

> Ban tach nho tai lieu: su dung bo file moi trong `docs/study/` de de tra cuu.
>
> Muc luc nhanh:
> - `docs/study/00_INDEX.md`
> - `docs/study/01_READ_CPP_PROJECT.md`
> - `docs/study/02_CMAKE_AND_BUILD.md`
> - `docs/study/03_PIPELINE_OVERVIEW.md`
> - `docs/study/04_CPP_CORE_CONCEPTS.md`
> - `docs/study/05_DEBUG_ROADMAP_EXERCISES.md`
> - `docs/study/PIPELINE_MERMAID.md`
> - `docs/study/FILE_TO_FILE_CALL_GRAPH.md`
> - `docs/study/CHECKLIST_CPP_FILE_10_MIN.md`

Tai lieu nay duoc viet de ban co the:
- Doc duoc mot du an C++ production theo cach co he thong.
- Hieu CMake va cach build project.
- Hieu cac ky thuat C++ trung binh -> nang cao dang xuat hien trong code hien tai.
- Hieu vi sao code dung con tro, tham chieu, namespace, template, uint8_t, static_cast, v.v.

Muc tieu: doc duoc code, debug duoc, va tu tin sua duoc du an nay.

---

## 1. Cach Doc Mot Du An C++ (Tong Quan)

Neu nhay vao ham ngay, ban rat de bi lo. Cach dung la di tu ngoai vao trong:

1. Build system
- Xem CMakeLists de biet target nao duoc build that su.
- Xem target nao la executable (main, benchmark), target nao la library (utils, services, tracking, pipeline).

2. Entrypoint
- Mo main cua executable (src/app/main.cpp).
- Tim luong dieu khien cao nhat: parse args -> load model -> xu ly image/video.

3. Public API
- Doc cac file header trong include/ocrplate truoc.
- Header cho ban hieu hop dong ham (input/output), truoc khi xem implementation.

4. Module boundaries
- app: orchestration + I/O
- pipeline: ghép cac nhanh xu ly
- services: model inference + postprocess
- tracking: theo doi doi tuong qua frame
- utils: tien ich phu tro

5. Cuoi cung moi xuong file .cpp
- Khi da hieu hop dong, implementation se de doc hon rat nhieu.

Quy tac vang: 
- Khong doc dong nao ma khong biet no nam trong module nao va phuc vu use-case nao.

---

## 2. CMake Tu Co Ban Den Du An Nay

## 2.1 CMake la gi?
CMake khong phai compiler. CMake tao ra he build (Ninja/Makefiles), roi he build moi goi compiler.

So do:
- CMakeLists.txt -> generate build files
- Ninja/Make -> compile/link -> binary

## 2.2 Nhung lenh CMake can biet

1. Khai bao project
- cmake_minimum_required(...)
- project(... LANGUAGES C CXX)

2. Cau hinh chuan C++
- set(CMAKE_CXX_STANDARD 23)
- set(CMAKE_CXX_STANDARD_REQUIRED ON)

3. Tao library
- add_library(ten STATIC ...cpp)

4. Tao executable
- add_executable(main ...cpp)

5. Include paths
- target_include_directories(...)

6. Link dependency
- target_link_libraries(...)

7. Tim package ngoai
- find_package(OpenCV REQUIRED)
- find_library(ONNXRUNTIME_LIB ...)

## 2.3 CMake cua project nay

File CMake chinh:
- CMakeLists tao 4 static libs:
  - ocrplate_utils
  - ocrplate_tracking
  - ocrplate_services
  - ocrplate_pipeline
- Va 2 executable:
  - main
  - benchmark

Y nghia:
- Tach module ro rang, giam coupling.
- pipeline link vao services+utils+tracking.
- main/benchmark chi can link pipeline + lib he thong.

## 2.4 CMakePresets la gi?

CMakePresets.json cung cap profile configure/build co san.
Vi du:
- linux-gcc-debug
- linux-gcc-release

Loi ich:
- Team build dong nhat.
- Giam loi do dung sai compiler hoac sai build dir.

## 2.5 build.sh va run.sh trong project

- build.sh:
  - Parse option (build type, jobs, clean, target)
  - Co logic tu sua cache compiler bi sai moi truong
  - Goi cmake -S -B va cmake --build

- run.sh:
  - Chon chay main hoac benchmark
  - Forward argument xuong binary

Day la kieu production tot: script hoa thao tac lap di lap lai.

---

## 3. Cach Nhin Cau Truc Thu Muc Du An Nay

- include/ocrplate: public headers (API hop dong)
- src/app: entrypoint + CLI + benchmark
- src/pipeline: logic ket hop nhanh xu ly
- src/services: infer model + preprocess/postprocess
- src/tracking: ByteTrack + identity store
- src/utils: helper dung chung
- third_party/onnxruntime: runtime vendor
- model: file onnx
- out/build/bin: binary output

Mot nguyen tac quan trong:
- Header = "what"
- CPP = "how"

---

## 4. Lo Trinh Chay Cua Project OCR Plate

## 4.1 Luong image
1. Parse args
2. Load ONNX sessions
3. AnnotateFrame:
- vehicle detect
- crop vehicle
- brand classify (cho car)
- plate detect + OCR
- tracking/identity update
- draw overlay
4. Save/show

## 4.2 Luong video
Khac image o cho:
- Co loop frame
- Co tracker (giu ID qua frame)
- Co infer moi N frame (kVideoInferEveryNFrames)
- Cac frame giua dung cached overlay + prediction

## 4.3 Tracking + Identity store

- Tracker gan track_id theo bbox qua thoi gian.
- Identity store de tich luy ket qua brand/plate theo track_id.

Vi sao can identity store?
- 1 frame OCR co the sai.
- Tich luy qua nhieu frame se on dinh hon.

---

## 5. C++ Thuc Chien: Con Tro, Tham Chieu, Mang, Vector

## 5.1 Con tro (*), tham chieu (&), gia tri

1. Truyen gia tri (copy)
- Ham nhan ban sao.
- An toan, nhung ton copy.

2. Truyen tham chieu (T& / const T&)
- Khong copy.
- const T& de doc-only, rat pho bien trong project.

3. Con tro (T*)
- Co the null.
- Dung khi tham so tuy chon (optional) hoac tuong tac C API.

Trong du an nay:
- TrackingRuntimeContext* tracking_ctx = nullptr
  - Co nghia: co the bat tracking hoac tat tracking.
  - Neu nullptr -> bo qua logic tracking.

Khi nao dung con tro thay vi bien thuong?
- Khi doi tuong co the khong ton tai (optional).
- Khi can ownership linh hoat.
- Khi goi API C/ONNX/OpenCV can pointer.

## 5.2 Tai sao luc mang, luc pointer, luc vector?

- Mang tinh (int a[10])
  - Kich thuoc co dinh compile-time.
  - It dung trong code production hien dai.

- Pointer (T*)
  - Tro den vung nho lien tuc.
  - Thuong xuat hien khi bridge voi C API (ONNX tensor data).

- std::vector<T>
  - Quan ly bo nho an toan, resize linh hoat.
  - Lua chon mac dinh trong project nay.

Quy tac thuc te:
- Uu tien vector.
- Pointer chi dung o boundary voi API yeu cau pointer.

## 5.3 Dang nguy hiem can tranh
- Dangling pointer
- Double free
- Use-after-free

Project nay giam rui ro bang cach:
- Uu tien vector/string.
- Dung RAII object cua C++/ONNX/OpenCV.

---

## 6. Smart Pointer va Ownership

Ban se gap:
- std::shared_ptr
- std::unique_ptr

Trong project:
- Co cache metadata session dung shared_ptr.
- Muc tieu: tai su dung metadata ma khong copy du lieu lon.

Nguyen tac:
- unique_ptr: 1 chu so huu duy nhat.
- shared_ptr: nhieu noi cung so huu.

Dung shared_ptr qua da se ton overhead (atomic ref count), nen chi dung khi can chia se ownership that su.

---

## 7. static_cast, uint8_t, int64_t, size_t la gi?

## 7.1 static_cast

Dung de ep kieu ro rang, an toan hon C-style cast.

Vi du:
- static_cast<int>(std::floor(x))

Tai sao can?
- Tranh canh bao compiler.
- The hien y do convert.
- Giam bug do ep kieu ngam.

## 7.2 uint8_t

- So nguyen 8-bit khong dau (0..255).
- Rat hop voi pixel image.

Tai sao OCR input dung uint8?
- Model OCR trong project ky vong NHWC uint8.
- Giu nguyen dynamic range pixel va giam bo nho.

## 7.3 int64_t

- So nguyen 64-bit co dau, do rong ro rang tren moi nen tang.
- ONNX shape/index thuong dung int64_t.

## 7.4 size_t

- Kieu dung cho kich thuoc/chi so container.
- Khop voi vector::size().

Quy tac:
- Index vector -> size_t
- Shape ONNX -> int64_t
- Pixel raw byte -> uint8_t

---

## 8. Tai sao cv::... ma khong phai cv....?

cv la namespace, khong phai object.

- cv::Mat
- cv::resize
- cv::cvtColor

Toan tu :: la scope resolution:
- "lay ten ben trong namespace/class"

Toan tu . la truy cap member cua object instance.

Vi du:
- cv::Mat img;           // dung namespace
- img.cols;              // dung object member

---

## 9. Template Trong C++ (Trung Binh -> Nang Cao)

## 9.1 Template la gi?
Template cho phep viet 1 ham/class cho nhieu kieu du lieu.

Vi du trong project:
- FillTensorFromRGB_NCHW<T>
- FillTensorFromRGB_NHWC<T>

Y nghia:
- Tai su dung chung logic cho float va uint8.
- Giam duplicate code.

## 9.2 Tai sao khong viet 2 ham rieng?
- Co the viet, nhung trung lap.
- Template + if constexpr giu code gon, de maintain.

## 9.3 Explicit instantiation
Ban se thay explicit instantiation trong yolo_preprocess.cpp.
Muc dich:
- Tranh link error do template.
- Chot cac kieu can dung (float, uint8).

---

## 10. Concurrency Trong Project Nay

Ban se gap:
- std::thread
- std::atomic
- std::mutex
- std::condition_variable
- std::async/std::future

Muc tieu:
- Tach writer/display khoi infer loop.
- Chay brand branch va plate branch song song.
- Parallel preprocessing/decode khi can.

Khau quyet:
- Parallelism dung cho CPU-bound stage.
- Queue + condvar dung cho producer-consumer.
- Atomic dung cho stop flag va scheduler.

---

## 11. ONNX Runtime Va Tensor Mapping

Nhung diem can hieu:
1. Model co shape va type input/output rieng.
2. Code phai map dung:
- Layout: NCHW hay NHWC
- Type: float32, float16, uint8
3. Sai 1 trong 2 diem tren -> output sai/vo nghia.

Vi du thuc te trong project:
- OCR: (N,64,128,3), uint8, NHWC
- Brand: float32, NCHW

---

## 12. Cach Debug Khi Khong Hieu Doan Code

1. Tim ham public tu header.
2. Doc comment input/output cua ham.
3. Tim noi ham duoc goi (call site).
4. In log bien trung gian (shape, score, size).
5. Xac minh assumption bang assert/check.

Checklist nhanh:
- Du lieu vao co dung type/shape?
- Index map subset -> global co dung?
- Threshold co qua chat/qua long?
- Tracking co bi reset ngoai y muon?

---

## 13. Gioi Thieu Nhanh Tung Module Cua Project Nay

## 13.1 app
- src/app/main.cpp: luong chay image/video, queue writer/display
- src/app/cli_args.cpp: parse CLI
- src/app/benchmark.cpp: benchmark theo stage

## 13.2 pipeline
- src/pipeline/frame_annotator.cpp: trung tam ket hop detect/brand/plate/ocr/tracking
- src/pipeline/track_trace.cpp: ve trace theo track_id

## 13.3 services
- yolo_detector.cpp + yolo_preprocess.cpp + yolo_postprocess.cpp + yolo_nms.cpp
- ocr_batch.cpp
- brand_classifier.cpp
- onnx_runner.cpp
- post_process_out_string.cpp

## 13.4 tracking
- byte_track_tracker.cpp
- vehicle_identity_store.cpp

## 13.5 utils
- image_preprocess.cpp
- plate_parallel.cpp
- parallel_utils.cpp
- onnx_decode_utils.cpp
- ocr_report.cpp

---

## 14. Giai Thich Nhanh Cac Pattern Ban Se Gap Nhieu

1. if (ptr != nullptr)
- Feature optional.
- Tranh crash khi ptr null.

2. reserve() truoc push_back()
- Giam realloc.
- Tang hieu nang.

3. std::move(...)
- Chuyen ownership, tranh copy lon.

4. clone() trong OpenCV
- Tao deep copy.
- Can khi object goc se thay doi hoac qua thread khac.

5. ROI view (bgr(r))
- Khong copy data.
- Nhanh hon clone, nhung can de y vong doi frame goc.

---

## 15. Lo Trinh Hoc C++ De Lam Chu Du An Nay

Giai doan A (nen tang)
- References, pointers, const correctness
- Vector/string/map
- RAII

Giai doan B (thuc chien)
- Move semantics
- Smart pointers
- Thread/atomic/mutex/condvar
- Template co if constexpr

Giai doan C (nang cao)
- Build system (CMake target-based)
- API boundary C++ <-> C (ONNX/OpenCV)
- Performance tuning (copy vs view, allocation, thread scheduling)

Cach hoc nhanh nhat:
- Chon 1 flow nho (image mode)
- Dat breakpoint/log
- Theo du lieu tu input -> output
- Ghi lai assumption sai/dung sau moi buoi

---

## 16. Bai Tap De Tu Tin Hon (Tren Chinh Project Nay)

1. Bai tap 1
- In them log shape/type input-output cua OCR model.
- Muc tieu: hieu NHWC uint8.

2. Bai tap 2
- Thu doi kVideoInferEveryNFrames va xem anh huong FPS/ID switch.

3. Bai tap 3
- Them 1 threshold moi vao app_config va dung no trong 1 nhanh xu ly.
- Build + test lai image/video.

4. Bai tap 4
- Voi frame_annotator, ve so luong vehicle/plate len goc man hinh.

---

## 17. FAQ Nhanh

Q: Tai sao co noi dung pointer ma khong dung bien thuong?
A: Vi co parameter optional, C API can pointer, hoac can chia se du lieu khong copy.

Q: Tai sao phai static_cast lien tuc?
A: De convert ro rang, tranh warning, tranh bug convert ngam.

Q: Tai sao namespace cv:: ma khong cv.?
A: cv la namespace (phai dung ::), con object moi dung .

Q: Tai sao co file legacy trong src/ goc?
A: Duong moi la src/app + src/pipeline + include/ocrplate; file legacy giu de tuong thich va tham chieu.

---

## 18. Cach Ban Nen Doc Project Nay Tu Hom Nay

Buoi 1:
- CMakeLists -> src/app/main.cpp -> include/ocrplate/pipeline/frame_annotator.h

Buoi 2:
- src/pipeline/frame_annotator.cpp (flow tong)

Buoi 3:
- src/services/ocr_batch.cpp + src/services/yolo_* + src/services/brand_classifier.cpp

Buoi 4:
- src/tracking/byte_track_tracker.cpp + vehicle_identity_store.cpp

Buoi 5:
- benchmark + toi uu threshold theo du lieu that

Neu doc dung thu tu nay, ban se thay "de tho" hon rat nhieu.

---

## 19. Ghi Chu Cuoi

Ban khong can hoc het C++ truoc roi moi sua du an.
Huong nhanh nhat la:
- Hoc den dau, apply vao chinh code den do.
- Moi khong hieu de ra 1 ghi chu "vi sao", sau do doi chieu voi code.

Tai lieu nay co chu y viet du de ban "thua con hon thieu".
Neu ban muon, co the tao tiep phien ban 2:
- Kem so do flow bang mermaid
- Kem bai tap co dap an mau ngay tren project nay.
