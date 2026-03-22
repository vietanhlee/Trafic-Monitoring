# 04 - C++ Concepts Can Nam De Lam Duoc Project Nay

## Pointer, reference, value
- Value: copy data, de hieu, ton cost copy.
- Reference (const T&): khong copy, phu hop tham so read-only.
- Pointer (T*): co the null, hop voi optional/C API.

Trong repo nay, pointer thuong xuat hien khi:
- Truyen optional context (vd tracking_ctx co the null)
- Lam viec voi ONNX/OpenCV low-level APIs

## Mang, pointer, vector
- Mang tinh: size co dinh compile-time.
- Pointer: vung nho lien tuc, boundary voi C API.
- std::vector: lua chon mac dinh vi an toan + linh hoat.

## Smart pointers va ownership
- unique_ptr: 1 owner duy nhat.
- shared_ptr: nhieu owner cung quan ly doi tuong.
- Dung shared_ptr khi that su can chia se ownership.

## Kieu so quan trong
- uint8_t: byte pixel 0..255, hop input image.
- int64_t: shape/index ONNX thuong dung.
- size_t: size/index cua container C++.

## static_cast
Dung de ep kieu ro rang, tranh cast ngam gay bug va warning.

## Namespace va scope resolution
- cv::Mat, cv::resize: cv la namespace nen dung ::
- obj.member: dung . tren object instance

## Template
Template giup tai su dung logic cho nhieu type (vd float/uint8) ma khong duplicate code.
Trong project co cac ham preprocess tensor theo template va explicit instantiation.

## Concurrency co ban trong repo
Ban se gap:
- std::thread
- std::atomic
- std::mutex
- std::condition_variable
- std::async/std::future

Muc tieu: tang throughput, tach producer-consumer, va giam block tren loop chinh.
