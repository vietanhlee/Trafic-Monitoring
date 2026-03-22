# Checklist In Hang Ngay: Doc 1 File C++ Trong 10 Phut

In file nay va danh dau moi ngay.

## 0:00 - 1:00 | Xac dinh vai tro file
- File nay thuoc module nao: app/pipeline/services/tracking/utils?
- File nay la header (.h) hay implementation (.cpp)?
- File nay co duoc build that trong CMake target nao?

## 1:00 - 2:30 | Nhin include va dependency
- Include nao la internal (ocrplate/...) ?
- Include nao la external (OpenCV/ONNX/std)?
- Doan nao chi la helper local namespace?

## 2:30 - 4:30 | Tim API chinh
- Ham/class public nao la entrypoint cua file?
- Input/output cua ham chinh la gi?
- Co invariants nao bat buoc dung (shape/type/null)?

## 4:30 - 6:30 | Lan theo data flow
- Du lieu vao den tu dau?
- Bien trung gian nao quan trong nhat?
- Du lieu ra se duoc file nao dung tiep?

## 6:30 - 8:00 | Tim risk/bug hotspots
- Boundary checks co day du khong (index/shape/rect)?
- Co ep kieu nao de loi (cast, sign, overflow)?
- Co race/thread-shared state nao can mutex/atomic?

## 8:00 - 9:00 | Performance quick scan
- Co copy lon khong can thiet khong?
- reserve/move da dung chua?
- Co branch nao co the skip som (early return)?

## 9:00 - 10:00 | Chot 3 dong ghi chu
- 1 dong: file nay lam gi.
- 1 dong: assumption quan trong nhat.
- 1 dong: 1 viec can verify them neu sua file.

## Ban in nhanh (tickbox)
- [ ] Biet file thuoc module nao
- [ ] Biet entrypoint ham/class
- [ ] Biet input/output format
- [ ] Biet dependency chinh
- [ ] Biet 2 risk hotspots
- [ ] Biet 1 diem performance
- [ ] Viet xong 3 dong tom tat
