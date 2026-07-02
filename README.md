# 🏦 Dự báo Rủi ro Tín dụng theo Mô hình 5C

Ứng dụng Streamlit chuyển thể từ notebook huấn luyện mô hình **Logistic Regression**
để dự báo rủi ro tín dụng (biến `PD` — Probability of Default) dựa trên 24 biến khảo
sát theo mô hình 5C:

| Nhóm | Ý nghĩa | Các cột |
|---|---|---|
| TC | Tính cách (Character) | TC1–TC5 |
| NL | Năng lực (Capacity) | NL1–NL4 |
| DK | Điều kiện (Conditions) | DK1–DK5 |
| V | Vốn (Capital) | V1–V6 |
| TS | Tài sản đảm bảo (Collateral) | TS1–TS4 |

Biến mục tiêu: **PD** (0 = không rủi ro, 1 = có rủi ro).

## 1. Cài đặt

```bash
pip install -r requirements.txt
```

## 2. Chạy ứng dụng

```bash
streamlit run app.py
```

## 3. Cấu trúc dữ liệu đầu vào

File CSV cần có tối thiểu các cột sau (giống file mẫu `5c.csv`):

- 24 biến đầu vào: `TC1, TC2, TC3, TC4, TC5, NL1, NL2, NL3, NL4, DK1, DK2, DK3, DK4, DK5, V1, V2, V3, V4, V5, V6, TS1, TS2, TS3, TS4` (thang điểm Likert 1–5).
- Biến mục tiêu: `PD` (0 hoặc 1).

Các cột khác trong file mẫu (ví dụ `Dấu thời gian`, `NN`) không được notebook sử dụng
làm biến đầu vào nên ứng dụng bỏ qua khi huấn luyện/dự báo.

## 4. Mô tả các tab

1. **📋 Tổng quan dữ liệu** — kích thước dữ liệu, xem nhanh dữ liệu thô, thống kê mô
   tả của các biến trong mô hình.
2. **📊 Trực quan hóa dữ liệu** — phân phối biến mục tiêu PD và các biến đầu vào (có
   thể chọn tối đa 3 biến để xem cùng lúc, mặc định TC1, NL1, V1 — mỗi biến đại diện
   một nhóm trong 5C).
3. **🧪 Kết quả huấn luyện & kiểm định mô hình** — Accuracy, Precision, Recall,
   F1-score, ROC-AUC, ma trận nhầm lẫn, đường cong ROC và báo cáo phân loại chi tiết
   (đúng theo các chỉ tiêu notebook đã tính: `confusion_matrix`, `model.score`).
4. **🔮 Sử dụng mô hình** — dự báo cho một khách hàng (nhập trực tiếp từng biến) hoặc
   dự báo hàng loạt (tải file CSV có đúng 24 cột đầu vào, tải kết quả về dạng CSV).

## 5. Ghi chú quan trọng

- Mô hình duy nhất trong notebook là `LogisticRegression()` không kèm scaler/encoder
  (dữ liệu đầu vào đã là số nguyên 1–5), nên ứng dụng không thêm bước chuẩn hóa nào
  ngoài việc chọn đúng thứ tự cột.
- `test_size=0.2` và `random_state=23` được lấy đúng theo notebook và dùng làm giá trị
  mặc định trên sidebar.
- Notebook khởi tạo `LogisticRegression()` **không chỉ định** `C`, `max_iter`, `solver`
  — ứng dụng bổ sung các tham số này trong mục "Tham số nâng cao" với giá trị mặc định
  của scikit-learn (`C=1.0`, `max_iter=100`, `solver="lbfgs"`) để người dùng có thể tinh
  chỉnh nếu cần; đây là lựa chọn hợp lý bổ sung, không có trong notebook gốc.
- Khuyến nghị dùng **Streamlit phiên bản mới** (≥ 1.38) để hỗ trợ tốt các tính năng
  bố cục dùng trong app như `st.multiselect(max_selections=...)`; nếu môi trường có
  Streamlit ≥ 1.55, có thể tận dụng thêm `st.container(horizontal=True)`, `st.space`
  hoặc dynamic container để tối ưu bố cục/tải biểu đồ nặng theo yêu cầu.
