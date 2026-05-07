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

- ⬜ Tạo bucket trên cloud (GCP/AWS/Azure)
- ⬜ Tạo credentials (GCP: service account + sa-key.json | AWS: access key | Azure: connection string)
- ⬜ Cấp quyền `objectAdmin` cho service account trên bucket

### DVC

- ⬜ `dvc init`
- ⬜ `dvc remote add -d myremote gs://<BUCKET>/dvc` (hoặc s3/azure tương đương)
- ⬜ `dvc remote modify myremote credentialpath sa-key.json` (GCP)
- ⬜ `dvc add data/train_phase1.csv data/eval.csv data/train_phase2.csv`
- ⬜ `git add data/*.dvc .dvc/config .gitignore`
- ⬜ `git commit -m "feat: track datasets with DVC"`
- ⬜ `dvc push` → xác nhận file xuất hiện trên cloud storage console

### VM trên Cloud

- ⬜ Tạo VM (GCE e2-small / EC2 t2.micro / Azure B1s)
- ⬜ Mở firewall port 8000
- ⬜ Lấy IP công khai của VM
- ⬜ SSH vào VM, cài: `pip3 install fastapi uvicorn scikit-learn joblib google-cloud-storage`
- ⬜ Tạo thư mục `~/models ~/src` trên VM
- ⬜ Copy `sa-key.json` lên VM

### Viết `src/serve.py` (tất cả TODO đều chưa làm)

- ⬜ TODO 1-4: Hàm `download_model()` — tạo `storage.Client()`, lấy bucket/blob, download về `~/models/model.pkl`
- ⬜ TODO 5: Endpoint `GET /health` → trả về `{"status": "ok"}`
- ⬜ TODO 6: Kiểm tra `len(req.features) == 12`, raise 400 nếu sai
- ⬜ TODO 7: Gọi `model.predict([req.features])`
- ⬜ TODO 8: Trả về `{"prediction": int, "label": "thấp"|"trung_bình"|"cao"}`

### Cấu hình systemd trên VM

- ⬜ Tạo file `/etc/systemd/system/mlops-serve.service` với đúng `GCS_BUCKET` và `GOOGLE_APPLICATION_CREDENTIALS`
- ⬜ `sudo systemctl daemon-reload && sudo systemctl enable mlops-serve`

### SSH Key cho GitHub Actions

- ⬜ Tạo SSH key: `ssh-keygen -t ed25519 -f ~/.ssh/mlops_deploy -N ""`
- ⬜ Thêm public key vào VM (`~/.ssh/authorized_keys`)

### GitHub Secrets (5 secrets)

- ⬜ `CLOUD_CREDENTIALS` — nội dung `sa-key.json` (JSON)
- ⬜ `CLOUD_BUCKET` — tên bucket
- ⬜ `VM_HOST` — IP công khai của VM
- ⬜ `VM_USER` — username trên VM
- ⬜ `VM_SSH_KEY` — nội dung private key `~/.ssh/mlops_deploy`

### Viết `tests/test_train.py` (tất cả TODO đều chưa làm)

- ⬜ TODO 1-5: Hàm `_make_temp_data` — tạo DataFrame ngẫu nhiên, lưu train.csv (160 dòng) và eval.csv (40 dòng)
- ⬜ TODO 6-7: `test_train_returns_float` — gọi `train()`, assert kết quả là float trong [0, 1]
- ⬜ TODO 8: `test_metrics_file_created` — assert `outputs/metrics.json` tồn tại và có đủ 2 keys
- ⬜ TODO 9: `test_model_file_created` — assert `models/model.pkl` tồn tại
- ⬜ Chạy thử local: `pytest tests/ -v` → cả 3 test phải xanh

### Viết `.github/workflows/mlops.yml` (tất cả TODO đều chưa làm)

- ⬜ TODO 1: Lệnh chạy pytest (`pytest tests/ -v`)
- ⬜ TODO 2: Ghi credentials ra file tạm + set env var xác thực
- ⬜ TODO 3: Lệnh `dvc pull` cho train_phase1.csv và eval.csv
- ⬜ TODO 4: Đọc accuracy từ `metrics.json`, set `$GITHUB_OUTPUT`
- ⬜ TODO 5: Upload `models/model.pkl` lên `gs://<bucket>/models/latest/model.pkl`
- ⬜ TODO 6: Eval gate — nếu accuracy < 0.70 thì `raise SystemExit`
- ⬜ TODO 7-8: SSH deploy — restart service, sleep 5, curl `/health`, exit 1 nếu fail

### Chạy pipeline lần đầu

- ⬜ `git add . && git commit -m "feat: add CI/CD pipeline, tests, and serving API"`
- ⬜ `git push origin main`
- ⬜ Theo dõi tab Actions — cả 4 jobs phải xanh
- ⬜ Sau khi pipeline xong: `sudo systemctl start mlops-serve` trên VM
- ⬜ Test endpoint:
  ```
  curl http://<VM_IP>:8000/health
  curl -X POST http://<VM_IP>:8000/predict -H "Content-Type: application/json" \
    -d '{"features": [7.4,0.70,0.00,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,0]}'
  ```
- ⬜ Chụp màn hình Actions (4 jobs xanh) để nộp bài

---

## Bước 3 - Huấn luyện liên tục

- ⬜ `python add_new_data.py` → train_phase1.csv tăng từ 2998 lên 5996 mẫu
- ⬜ `dvc add data/train_phase1.csv`
- ⬜ `git add data/train_phase1.csv.dvc`
- ⬜ `git commit -m "data: bổ sung 2998 mẫu dữ liệu mới (train_phase2)"`
- ⬜ `dvc push` (phải push DVC trước git push!)
- ⬜ `git push origin main`
- ⬜ Xác nhận pipeline tự kích hoạt bởi commit dữ liệu (không phải commit code)
- ⬜ Cả 4 jobs xanh, VM phục vụ model mới
- ⬜ Điền bảng so sánh accuracy Bước 2 vs Bước 3
- ⬜ Chụp màn hình Actions để nộp bài

---

## Nộp bài

- ⬜ URL repo GitHub public
- ⬜ Ảnh MLflow UI (≥ 3 runs)
- ⬜ Ảnh GitHub Actions (4 jobs xanh — Bước 2)
- ⬜ Ảnh GitHub Actions (4 jobs xanh — Bước 3, triggered bởi commit data)
- ⬜ Ảnh kết quả `curl /health` và `curl /predict`
- ⬜ Ảnh Cloud Storage Console (file dữ liệu + model)
- ⬜ Báo cáo ngắn ≤ 1 trang A4: params tốt nhất + lý do + khó khăn gặp phải
