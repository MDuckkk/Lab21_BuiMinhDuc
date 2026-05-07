# Checklist Lab MLOps - Day 21

Trạng thái: ✅ Đã xong | ⬜ Chưa làm

---

## Bước 0 - Setup môi trường

- ⬜ Tạo venv Python 3.12: `py -3.12 -m venv .venv`
- ⬜ Activate venv: `.venv\Scripts\activate`
- ⬜ Cài thư viện: `pip install -r requirements.txt`
- ✅ `requirements.txt` đã được cập nhật (numpy 2.2.6, scikit-learn 1.6.1, ...)
- ✅ `.gitignore` đã có đầy đủ
- ⬜ Tải dữ liệu: `python generate_data.py` → tạo ra 3 file CSV trong `data/`

---

## Bước 1 - Thực nghiệm cục bộ với MLflow

### Viết `src/train.py`

- ✅ TODO 1: Đọc `train_phase1.csv` và `eval.csv` bằng `pd.read_csv`
- ✅ TODO 2: Tách X (drop cột "target") và y cho cả train lẫn eval
- ✅ TODO 3: Ghi siêu tham số vào MLflow (`mlflow.log_params`)
- ✅ TODO 4: Khởi tạo và train `RandomForestClassifier(**params, random_state=42)`
- ✅ TODO 5: Tính `accuracy_score` và `f1_score(average="weighted")` trên eval
- ✅ TODO 6: Ghi metrics vào MLflow (`log_metric`) + log model (`log_model`)
- ✅ TODO 7: In kết quả ra màn hình
- ✅ TODO 8: Lưu `outputs/metrics.json` với `{"accuracy": ..., "f1_score": ...}`
- ✅ TODO 9: Lưu `models/model.pkl` bằng `joblib.dump`
- ✅ TODO 10: `return acc`

### Chạy thí nghiệm

- ✅ Chạy lần 1: n_estimators=200, max_depth=10, min_samples_split=5 → accuracy=0.6440
- ✅ Chạy lần 2: n_estimators=50, max_depth=3, min_samples_split=2 → accuracy=0.5580
- ✅ Chạy lần 3: n_estimators=200, max_depth=10, min_samples_split=5 → accuracy=0.6440
- ✅ MLflow UI hiển thị đủ 3 runs (bouncy-deer, skittish-smelt, lyrical-steed)
- ✅ Bộ params tốt nhất: n_estimators=200, max_depth=10, min_samples_split=5 (accuracy=0.6440)
- ✅ `params.yaml` đã cập nhật với bộ params tốt nhất
- ⬜ Chụp màn hình MLflow UI (≥ 3 runs) để nộp bài

---

## Bước 2 - CI/CD tự động

### Cloud Setup

- ✅ Tạo S3 bucket: `lab21-139929687520-ap-southeast-1-an`
- ✅ Tạo IAM credentials (access key + secret key)
- ✅ Mở port 8000 trên Security Group EC2

### DVC

- ✅ `dvc init`
- ✅ `dvc remote add -d myremote s3://<BUCKET>/dvc`
- ✅ `dvc add data/train_phase1.csv data/eval.csv data/train_phase2.csv`
- ✅ `git commit -m "feat: track datasets with DVC"`
- ✅ `dvc push` → data xuất hiện trên S3

### VM trên Cloud

- ✅ Tạo EC2 t3.micro
- ✅ Mở port 8000
- ✅ IP công khai: 13.229.76.131
- ✅ Cài dependencies: fastapi uvicorn scikit-learn joblib boto3
- ✅ Tạo thư mục `~/models ~/src`

### Viết `src/serve.py`

- ✅ Hàm `download_model()` dùng boto3 download từ S3
- ✅ Endpoint `GET /health` → `{"status": "ok"}`
- ✅ Endpoint `POST /predict` → `{"prediction": int, "label": str}`

### Cấu hình systemd trên VM

- ✅ Tạo `/etc/systemd/system/mlops-serve.service`
- ✅ Thêm AWS credentials vào service file
- ✅ `sudo systemctl enable mlops-serve`

### SSH Key cho GitHub Actions

- ✅ Tạo SSH key `~/.ssh/mlops_deploy`
- ✅ Thêm public key vào VM

### GitHub Secrets (5 secrets)

- ✅ `CLOUD_CREDENTIALS`
- ✅ `CLOUD_BUCKET`
- ✅ `VM_HOST`
- ✅ `VM_USER`
- ✅ `VM_SSH_KEY`

### Viết `tests/test_train.py`

- ✅ `_make_temp_data` — tạo data ngẫu nhiên
- ✅ `test_train_returns_float` — pass
- ✅ `test_metrics_file_created` — pass
- ✅ `test_model_file_created` — pass

### Viết `.github/workflows/mlops.yml`

- ✅ 4 jobs: Unit Test → Train → Eval → Deploy
- ✅ Eval gate accuracy >= 0.70
- ✅ SSH deploy + health check

### Kết quả

- ✅ Cả 4 jobs xanh (Pipeline #8)
- ✅ `curl /health` → `{"status":"ok"}`
- ✅ `curl /predict` → `{"prediction":0,"label":"thap"}`
- ✅ Chụp màn hình Actions Bước 2

---

## Bước 3 - Huấn luyện liên tục

- ✅ `python add_new_data.py` → train_phase1.csv tăng từ 2998 lên 5996 mẫu
- ✅ `dvc add data/train_phase1.csv`
- ✅ `git commit -m "data: bổ sung 2998 mẫu dữ liệu mới (train_phase2)"`
- ✅ `dvc push` + `git push origin master`
- ✅ Pipeline #9 tự kích hoạt bởi commit data
- ✅ Cả 4 jobs xanh, accuracy 0.748 > 0.70
- ✅ Chụp màn hình Actions Bước 3

---

## Nộp bài

- ⬜ URL repo GitHub public
- ✅ Ảnh MLflow UI (3 runs)
- ✅ Ảnh GitHub Actions 4 jobs xanh (Bước 2)
- ✅ Ảnh GitHub Actions 4 jobs xanh (Bước 3)
- ✅ Ảnh `curl /health` và `curl /predict`
- ✅ Ảnh S3 Console
- ✅ Báo cáo: `result/baocao.md`
