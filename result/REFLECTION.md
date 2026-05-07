# Báo Cáo Lab MLOps - Day 21

**Họ tên:** Bùi Minh Đức 

---

## 1. Bộ siêu tham số tốt nhất và lý do

Sau khi thử nghiệm 3 bộ tham số trên MLflow:

| n_estimators | max_depth | min_samples_split | Accuracy |
|---|---|---|---|
| 50 | 3 | 2 | 0.5580 |
| 200 | 10 | 5 | 0.6440 |
| 300 | 15 | 2 | 0.6700 |

Bộ tham số được chọn: `n_estimators=300, max_depth=15, min_samples_split=2`.

Lý do: độ sâu cây lớn hơn giúp mô hình học được các pattern phức tạp hơn trong dữ liệu Wine Quality, số cây nhiều hơn giúp giảm variance và tăng độ ổn định của dự đoán.

---

## 2. Kết quả so sánh Bước 2 vs Bước 3

| Chỉ số | Bước 2 (2998 mẫu) | Bước 3 (5996 mẫu) |
|---|---|---|
| Accuracy | 0.6700 | 0.7480 |
| F1-score | 0.6417 | 0.7468 |

Thêm dữ liệu từ train_phase2 giúp accuracy tăng từ 0.67 lên 0.748, vượt ngưỡng eval gate 0.70 và cho phép pipeline tự động deploy model mới.

---

## 3. Khó khăn và cách giải quyết

- **Python 3.14 không tương thích scikit-learn:** pip cố build từ source và kéo numpy RC version không tồn tại. Giải quyết bằng cách tạo lại venv với Python 3.12.
- **MLflow UI không hiển thị runs:** tracking URI không được set nên MLflow ghi vào thư mục mặc định thay vì sqlite file. Giải quyết bằng cách hardcode `sqlite:///mlflow.db` trong `train.py`.
- **AWS credentials mất sau mỗi PowerShell session:** lệnh `set` không hoạt động trong PowerShell. Giải quyết bằng cách dùng `$env:AWS_ACCESS_KEY_ID`.
- **VM không có credentials để download model từ S3:** systemd service không kế thừa env vars của user. Giải quyết bằng cách thêm `AWS_ACCESS_KEY_ID` và `AWS_SECRET_ACCESS_KEY` trực tiếp vào file `/etc/systemd/system/mlops-serve.service`.
