# Phân loại bình luận tiếng Việt theo Toxicity và Constructiveness

Dự án này xây dựng hệ thống xử lý ngôn ngữ tự nhiên tiếng Việt nhằm phân loại bình luận theo hai tiêu chí chính:

- **Toxicity**: xác định bình luận có mang tính độc hại, công kích, xúc phạm hoặc tiêu cực hay không.
- **Constructiveness**: xác định bình luận có mang tính xây dựng, góp ý hoặc đóng góp ý kiến có giá trị hay không.

Notebook `program.ipynb` triển khai quy trình tương đối đầy đủ cho bài toán phân loại văn bản tiếng Việt, bao gồm: đọc dữ liệu, phân tích dữ liệu khám phá, tiền xử lý văn bản, vector hóa TF-IDF, huấn luyện các mô hình học máy truyền thống, mở rộng với PhoBERT và phân tích lỗi dự đoán.

---

## Mục lục

1. [Mục tiêu dự án](#1-mục-tiêu-dự-án)
2. [Bộ dữ liệu sử dụng](#2-bộ-dữ-liệu-sử-dụng)
3. [Công nghệ và thư viện sử dụng](#3-công-nghệ-và-thư-viện-sử-dụng)
4. [Cài đặt môi trường](#4-cài-đặt-môi-trường)
5. [Cấu trúc dự án đề xuất](#5-cấu-trúc-dự-án-đề-xuất)
6. [Quy trình xử lý chính](#6-quy-trình-xử-lý-chính)
7. [Các mô hình được huấn luyện](#7-các-mô-hình-được-huấn-luyện)
8. [Lý do lựa chọn mô hình](#8-lý-do-lựa-chọn-mô-hình)
9. [Kết quả thực nghiệm](#9-kết-quả-thực-nghiệm)
10. [Phân tích lỗi mô hình](#10-phân-tích-lỗi-mô-hình)
11. [Cách chạy notebook](#11-cách-chạy-notebook)
12. [Lưu ý khi chạy PhoBERT](#12-lưu-ý-khi-chạy-phobert)
13. [Nhận xét kết quả](#13-nhận-xét-kết-quả)
14. [Hạn chế hiện tại](#14-hạn-chế-hiện-tại)
15. [Hướng phát triển](#15-hướng-phát-triển)
16. [Ví dụ dự đoán](#16-ví-dụ-dự-đoán)
17. [Trích dẫn dataset](#17-trích-dẫn-dataset)
18. [Tác giả](#18-tác-giả)
19. [Kết luận](#19-kết-luận)

---

## 1. Mục tiêu dự án

Mục tiêu chính của dự án là xây dựng và đánh giá các mô hình phân loại bình luận tiếng Việt phục vụ cho các bài toán kiểm duyệt và đánh giá chất lượng nội dung.

Cụ thể, dự án tập trung vào các mục tiêu sau:

- Phát hiện bình luận độc hại trên các nền tảng mạng xã hội, diễn đàn hoặc báo điện tử.
- Nhận diện bình luận mang tính xây dựng nhằm hỗ trợ lọc và ưu tiên nội dung có giá trị.
- So sánh hiệu quả giữa mô hình học máy truyền thống và mô hình ngôn ngữ hiện đại.
- Phân tích ảnh hưởng của mất cân bằng dữ liệu đến kết quả phân loại.
- Giải thích nguyên nhân mô hình dự đoán sai thông qua trọng số TF-IDF và đóng góp của từng từ hoặc cụm từ.

---

## 2. Bộ dữ liệu sử dụng

Dự án sử dụng bộ dữ liệu **ViCTSD - Vietnamese Constructive and Toxic Speech Detection dataset**.

Nguồn dữ liệu:

```text
https://huggingface.co/datasets/tarudesu/ViCTSD
```

ViCTSD là bộ dữ liệu gồm **10.000 bình luận tiếng Việt được gán nhãn thủ công**, phục vụ cho hai nhiệm vụ chính:

- Phân loại bình luận độc hại.
- Phân loại bình luận mang tính xây dựng.

Bộ dữ liệu được xây dựng từ bình luận của người dùng tiếng Việt trên nhiều chủ đề khác nhau. Theo thông tin từ dataset card, dữ liệu gồm các bình luận thuộc **10 lĩnh vực/chủ đề** và được chia sẵn thành ba tập train, validation và test.

### 2.1. Phân chia dữ liệu

| Tập dữ liệu | Số mẫu | File sử dụng trong notebook |
|---|---:|---|
| Train | 7.000 | `ViCTSD_train.csv` |
| Validation | 2.000 | `ViCTSD_valid.csv` |
| Test | 1.000 | `ViCTSD_test.csv` |
| Tổng cộng | 10.000 | - |

### 2.2. Các cột trong dữ liệu

| Cột | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| `Unnamed: 0` | Số nguyên | ID hoặc chỉ mục gốc của bình luận |
| `Comment` | Chuỗi | Nội dung bình luận tiếng Việt |
| `Constructiveness` | Số nguyên | Nhãn cho biết bình luận có mang tính xây dựng hay không |
| `Toxicity` | Số nguyên | Nhãn cho biết bình luận có độc hại hay không |
| `Title` | Chuỗi | Tiêu đề bài viết liên quan đến bình luận |
| `Topic` | Chuỗi | Chủ đề của bài viết |

### 2.3. Ý nghĩa nhãn

| Nhãn | Giá trị `0` | Giá trị `1` |
|---|---|---|
| `Toxicity` | Không độc hại | Độc hại |
| `Constructiveness` | Không mang tính xây dựng | Có tính xây dựng |

### 2.4. Phân bố nhãn trong tập train

| Nhãn | Lớp 0 | Lớp 1 | Nhận xét |
|---|---:|---:|---|
| `Toxicity` | 6.241 | 759 | Mất cân bằng mạnh, lớp độc hại chiếm tỷ lệ thấp |
| `Constructiveness` | 4.497 | 2.503 | Mất cân bằng nhẹ hơn so với Toxicity |

Từ phân bố trên, có thể thấy bài toán `Toxicity` khó hơn do số lượng mẫu độc hại thấp hơn nhiều so với mẫu không độc hại. Vì vậy, các mô hình học máy truyền thống trong notebook sử dụng tham số:

```python
class_weight='balanced'
```

Cách làm này giúp giảm ảnh hưởng của mất cân bằng dữ liệu bằng cách gán trọng số cao hơn cho lớp thiểu số trong quá trình huấn luyện.

### 2.5. Cách tải dữ liệu từ Hugging Face

Có thể tải dữ liệu trực tiếp bằng thư viện `datasets`:

```python
from datasets import load_dataset

raw_dataset = load_dataset("tarudesu/ViCTSD")

train_df = raw_dataset["train"].to_pandas()
valid_df = raw_dataset["validation"].to_pandas()
test_df = raw_dataset["test"].to_pandas()
```

Nếu notebook đang đọc dữ liệu từ file CSV, có thể lưu lại các tập dữ liệu như sau:

```python
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(DATA_DIR / "ViCTSD_train.csv", index=False)
valid_df.to_csv(DATA_DIR / "ViCTSD_valid.csv", index=False)
test_df.to_csv(DATA_DIR / "ViCTSD_test.csv", index=False)
```

Sau đó có thể đọc lại bằng `pandas`:

```python
import pandas as pd

train_df = pd.read_csv("data/ViCTSD_train.csv")
valid_df = pd.read_csv("data/ViCTSD_valid.csv")
test_df = pd.read_csv("data/ViCTSD_test.csv")
```

> **Lưu ý:** Dataset ViCTSD được công bố phục vụ mục đích nghiên cứu. Khi sử dụng trong báo cáo, bài nộp hoặc repository công khai, nên ghi rõ nguồn dữ liệu và trích dẫn bài báo gốc.

---

## 3. Công nghệ và thư viện sử dụng

Dự án được triển khai bằng Python trong môi trường Jupyter Notebook hoặc Google Colab.

| Nhóm chức năng | Thư viện / Công cụ |
|---|---|
| Xử lý dữ liệu | `pandas`, `numpy` |
| Trực quan hóa | `matplotlib`, `seaborn`, `wordcloud` |
| Xử lý tiếng Việt | `underthesea` |
| Vector hóa văn bản | `TfidfVectorizer` |
| Mô hình học máy | `LinearSVC`, `LogisticRegression` |
| Pipeline huấn luyện | `sklearn.pipeline.Pipeline` |
| Đánh giá mô hình | `classification_report`, `confusion_matrix`, `accuracy_score`, `f1_score` |
| Deep Learning | `torch` |
| Mô hình Transformer | `transformers` |
| Dataset từ Hugging Face | `datasets` |
| Môi trường chạy | Jupyter Notebook, Google Colab |

---

## 4. Cài đặt môi trường

### 4.1. Chạy trên Google Colab

Cài đặt các thư viện cần thiết:

```bash
pip install underthesea scikit-learn pandas numpy matplotlib seaborn wordcloud transformers torch tqdm datasets
```

Nếu notebook dùng file CSV, cần tải ba file dữ liệu vào thư mục `/content/`:

```text
/content/ViCTSD_train.csv
/content/ViCTSD_valid.csv
/content/ViCTSD_test.csv
```

Nếu muốn lấy dữ liệu trực tiếp từ Hugging Face, có thể dùng:

```python
from datasets import load_dataset

raw_dataset = load_dataset("tarudesu/ViCTSD")
```

### 4.2. Chạy trên máy cá nhân

Clone hoặc tải project về máy:

```bash
git clone <repository-url>
cd <repository-name>
```

Tạo môi trường ảo:

```bash
python -m venv venv
```

Kích hoạt môi trường:

```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

Cài đặt thư viện:

```bash
pip install underthesea scikit-learn pandas numpy matplotlib seaborn wordcloud transformers torch tqdm datasets notebook
```

Chạy Jupyter Notebook:

```bash
jupyter notebook
```

Mở file:

```text
program.ipynb
```

### 4.3. Gợi ý tạo file `requirements.txt`

Để repository dễ cài đặt hơn, có thể tạo file `requirements.txt`:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
wordcloud
underthesea
torch
transformers
tqdm
datasets
notebook
```

Sau đó cài đặt bằng lệnh:

```bash
pip install -r requirements.txt
```

---

## 5. Cấu trúc dự án đề xuất

```text
.
├── program.ipynb
├── README.md
├── requirements.txt
├── data/
│   ├── ViCTSD_train.csv
│   ├── ViCTSD_valid.csv
│   └── ViCTSD_test.csv
├── models/
│   ├── best_model_Toxicity.pt
│   ├── best_model_Constructiveness.pt
│   ├── svm_toxicity.pkl
│   ├── svm_constructiveness.pkl
│   ├── logistic_toxicity.pkl
│   └── logistic_constructiveness.pkl
└── outputs/
    ├── confusion_matrix/
    ├── figures/
    ├── reports/
    └── error_analysis/
```

Notebook hiện tại có thể chạy với `program.ipynb` và ba file CSV. Tuy nhiên, cấu trúc trên được đề xuất để project dễ quản lý hơn khi đưa lên GitHub hoặc phát triển thành ứng dụng thực tế.

---

## 6. Quy trình xử lý chính

### 6.1. Đọc dữ liệu

Notebook đọc ba tập dữ liệu:

```python
train_df = pd.read_csv("/content/ViCTSD_train.csv")
valid_df = pd.read_csv("/content/ViCTSD_valid.csv")
test_df = pd.read_csv("/content/ViCTSD_test.csv")
```

Hoặc khi chạy local:

```python
train_df = pd.read_csv("data/ViCTSD_train.csv")
valid_df = pd.read_csv("data/ViCTSD_valid.csv")
test_df = pd.read_csv("data/ViCTSD_test.csv")
```

Sau khi đọc dữ liệu, notebook kiểm tra:

- Kích thước từng tập dữ liệu.
- Kiểu dữ liệu của từng cột.
- Một số dòng dữ liệu mẫu.
- Số lượng giá trị thiếu nếu có.
- Phân bố nhãn của từng bài toán.

### 6.2. Phân tích dữ liệu khám phá

Notebook thực hiện các bước phân tích như:

- Thống kê phân bố nhãn `Toxicity` và `Constructiveness`.
- Vẽ biểu đồ số lượng nhãn trên train, validation và test.
- Phân tích độ dài bình luận theo số từ.
- Lọc và quan sát các bình luận dài hơn 150 từ.
- Vẽ Word Cloud cho từng tập dữ liệu.
- Thống kê các từ phổ biến.
- Vẽ heatmap độ dài trung bình bình luận theo từng nhãn.

Các bước này giúp hiểu rõ đặc điểm dữ liệu trước khi huấn luyện mô hình, đặc biệt là vấn đề mất cân bằng nhãn và sự khác biệt giữa các nhóm bình luận.

### 6.3. Tiền xử lý văn bản tiếng Việt

Hàm `preprocess()` thực hiện các bước chính:

1. Chuẩn hóa Unicode về dạng NFC.
2. Chuyển một số từ viết tắt sang dạng chuẩn.
3. Chuyển văn bản về chữ thường.
4. Loại bỏ ký tự đặc biệt không cần thiết.
5. Tách từ tiếng Việt bằng `underthesea.word_tokenize()`.
6. Loại bỏ một số stopword tiếng Việt.
7. Ghép token lại thành chuỗi để phục vụ vector hóa TF-IDF.

Ví dụ một số từ viết tắt được chuẩn hóa:

| Từ viết tắt | Dạng chuẩn |
|---|---|
| `ko`, `k`, `khg`, `hok` | `không` |
| `dc`, `đc` | `được` |
| `vs` | `với` |
| `j`, `z` | `gì` |
| `mik` | `mình` |
| `ntn` | `như thế nào` |

Việc chuẩn hóa này đặc biệt quan trọng với dữ liệu bình luận mạng xã hội vì người dùng thường viết tắt, viết sai chính tả, dùng teencode hoặc sử dụng ký hiệu không chuẩn.

### 6.4. Vector hóa bằng TF-IDF

Sau khi tiền xử lý, văn bản được biểu diễn bằng TF-IDF:

```python
TfidfVectorizer(ngram_range=(1, 2), min_df=1)
```

Thiết lập này sử dụng:

- **Unigram**: từng từ riêng lẻ.
- **Bigram**: cụm hai từ liên tiếp.
- **`min_df=1`**: giữ lại cả những từ hoặc n-gram xuất hiện ít nhất một lần.

Ưu điểm của TF-IDF là đơn giản, hiệu quả và phù hợp với mô hình tuyến tính. Tuy nhiên, TF-IDF không hiểu sâu ngữ cảnh, thứ tự câu hoặc sắc thái mỉa mai, vì vậy vẫn có hạn chế đối với các bình luận phức tạp.

---

## 7. Các mô hình được huấn luyện

### 7.1. SVM với LinearSVC

Mô hình SVM được triển khai bằng `LinearSVC` kết hợp với TF-IDF trong `Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
    ('clf', LinearSVC(C=1.0, class_weight='balanced', max_iter=1000))
])
```

Mô hình được huấn luyện riêng cho hai nhãn:

- `Toxicity`
- `Constructiveness`

Notebook cũng thử nghiệm nhiều cấu hình khác nhau:

- So sánh nhiều khoảng `ngram_range`: `(1,1)`, `(1,2)`, `(1,3)`, `(1,4)`, `(1,5)`.
- So sánh nhiều giá trị `C`: `0.01`, `0.1`, `0.5`, `1.0`, `5.0`, `10.0`, `20.0`.

### 7.2. Logistic Regression

Mô hình Logistic Regression được triển khai bằng `Pipeline`:

```python
from sklearn.linear_model import LogisticRegression

logistic_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])
```

Mô hình này đóng vai trò baseline tuyến tính, dễ huấn luyện, dễ giải thích và phù hợp để so sánh với SVM.

### 7.3. PhoBERT

Notebook mở rộng thêm mô hình PhoBERT dựa trên checkpoint:

```text
vinai/phobert-base-v2
```

Các thông số chính:

| Thành phần | Giá trị |
|---|---|
| Tokenizer | `vinai/phobert-base-v2` |
| `MAX_LEN` | 128 |
| Batch size | 16 |
| Optimizer | `AdamW` |
| Learning rate | `2e-5` |
| Epoch | 10 |
| Dropout | 0.3 |
| Loss | `CrossEntropyLoss` có class weight |
| Đầu ra | 2 lớp |

Kiến trúc classifier:

```text
PhoBERT
   ↓
CLS embedding
   ↓
Dropout(0.3)
   ↓
Linear(768, 2)
   ↓
Logits cho 2 lớp
```

Mô hình được huấn luyện riêng cho:

- `Toxicity`
- `Constructiveness`

Trong quá trình huấn luyện, notebook lưu mô hình tốt nhất theo F1-score trên tập validation:

```python
best_model_path = f"best_model_{label_name}.pt"
```

---

## 8. Lý do lựa chọn mô hình

Dự án sử dụng cả mô hình học máy truyền thống và mô hình ngôn ngữ hiện đại để có góc nhìn so sánh toàn diện hơn.

| Mô hình | Lý do lựa chọn | Vai trò trong dự án |
|---|---|---|
| **SVM + TF-IDF** | SVM tuyến tính thường hoạt động tốt với dữ liệu văn bản có số chiều cao và dạng sparse. TF-IDF giúp biểu diễn từ/ngram theo mức độ quan trọng trong văn bản. | Mô hình truyền thống chính, kỳ vọng cho hiệu quả tốt và ổn định. |
| **Logistic Regression + TF-IDF** | Dễ huấn luyện, dễ giải thích trọng số đặc trưng, phù hợp làm baseline cho bài toán phân loại văn bản. | Baseline tuyến tính để so sánh với SVM. |
| **PhoBERT** | Là mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt, có khả năng học ngữ cảnh tốt hơn so với TF-IDF. | Mô hình deep learning hiện đại để so sánh với phương pháp truyền thống. |

Việc chọn ba mô hình này giúp dự án trả lời được câu hỏi quan trọng: với bộ dữ liệu ViCTSD, phương pháp truyền thống đơn giản như TF-IDF + SVM có đủ hiệu quả hay mô hình ngôn ngữ hiện đại như PhoBERT sẽ mang lại lợi thế rõ rệt hơn?

---

## 9. Kết quả thực nghiệm

### 9.1. Kết quả trên tập Validation

| Mô hình | Nhãn | Accuracy | Ghi chú |
|---|---|---:|---|
| SVM + TF-IDF | Toxicity | 0.8905 | Hiệu quả tốt với lớp không độc hại, lớp độc hại vẫn khó do mất cân bằng dữ liệu |
| SVM + TF-IDF | Constructiveness | 0.7765 | Cân bằng tương đối tốt giữa hai lớp |
| Logistic Regression + TF-IDF | Toxicity | 0.8630 | Recall lớp độc hại có thể tốt hơn, nhưng accuracy thấp hơn SVM |
| Logistic Regression + TF-IDF | Constructiveness | 0.7525 | Thấp hơn SVM trên validation |
| PhoBERT | Toxicity | Best F1 = 0.3079 | Kết quả chưa ổn định, chịu ảnh hưởng mạnh từ mất cân bằng dữ liệu |
| PhoBERT | Constructiveness | Best F1 = 0.7376 | Hoạt động tốt hơn trên nhãn Constructiveness |

### 9.2. Kết quả trên tập Test

| Mô hình | Nhãn | Accuracy | F1 lớp 1 |
|---|---|---:|---:|
| SVM + TF-IDF | Toxicity | 0.8950 | 0.47 |
| SVM + TF-IDF | Constructiveness | 0.7940 | 0.75 |
| Logistic Regression + TF-IDF | Toxicity | 0.8640 | 0.48 |
| Logistic Regression + TF-IDF | Constructiveness | 0.7790 | 0.75 |
| PhoBERT | Toxicity | 0.7730 | 0.26 |
| PhoBERT | Constructiveness | 0.7790 | 0.72 |

### 9.3. Nhận xét nhanh về kết quả

Nếu xét theo **Accuracy**, SVM + TF-IDF cho kết quả tốt nhất trên cả hai nhãn ở tập test:

- `Toxicity`: 0.8950
- `Constructiveness`: 0.7940

Tuy nhiên, với nhãn `Toxicity`, Logistic Regression đạt **F1 lớp 1 = 0.48**, nhỉnh hơn SVM với **F1 lớp 1 = 0.47**. Điều này cho thấy Logistic Regression có thể nhận diện lớp độc hại tốt hơn một chút, dù accuracy tổng thể thấp hơn.

Vì `Toxicity` là bài toán mất cân bằng dữ liệu, không nên chỉ dựa vào accuracy. Cần quan sát thêm các chỉ số như:

- Precision của lớp độc hại.
- Recall của lớp độc hại.
- F1-score của lớp độc hại.
- Macro F1.
- Confusion matrix.

---

## 10. Phân tích lỗi mô hình

Notebook không chỉ huấn luyện và đánh giá mô hình mà còn phân tích các trường hợp dự đoán sai.

Các bước phân tích gồm:

- Lọc các bình luận bị phân loại sai.
- In nội dung bình luận gốc.
- In câu sau khi tiền xử lý.
- So sánh nhãn thật và nhãn dự đoán.
- Trích xuất TF-IDF của từng từ/ngram.
- Lấy trọng số của mô hình tương ứng với từng từ/ngram.
- Tính đóng góp của từng đặc trưng.

Công thức đóng góp:

```text
Đóng góp = TF-IDF × trọng số mô hình
```

Cách phân tích này giúp giải thích vì sao mô hình đưa ra dự đoán sai, đặc biệt với các mô hình tuyến tính như SVM và Logistic Regression.

Ví dụ, nếu một bình luận bị dự đoán nhầm là độc hại, có thể kiểm tra xem các từ/ngram nào có trọng số cao đã kéo quyết định của mô hình về lớp độc hại. Ngược lại, nếu một bình luận độc hại bị bỏ sót, có thể xem liệu câu đó có quá ít từ khóa độc hại rõ ràng hoặc mang sắc thái mỉa mai khó nhận diện hay không.

---

## 11. Cách chạy notebook

### 11.1. Chuẩn bị dữ liệu

Có hai cách chuẩn bị dữ liệu.

#### Cách 1: Dùng file CSV

Đảm bảo có đủ ba file:

```text
ViCTSD_train.csv
ViCTSD_valid.csv
ViCTSD_test.csv
```

Nếu chạy trên Google Colab, có thể tải ba file này vào `/content/`.

Nếu chạy local, nên đặt vào thư mục:

```text
data/
```

#### Cách 2: Tải trực tiếp từ Hugging Face

```python
from datasets import load_dataset

raw_dataset = load_dataset("tarudesu/ViCTSD")

train_df = raw_dataset["train"].to_pandas()
valid_df = raw_dataset["validation"].to_pandas()
test_df = raw_dataset["test"].to_pandas()
```

### 11.2. Cài đặt thư viện

Chạy lệnh sau ở đầu notebook hoặc terminal:

```bash
pip install underthesea scikit-learn pandas numpy matplotlib seaborn wordcloud transformers torch tqdm datasets
```

### 11.3. Thứ tự chạy đề xuất

Đối với mô hình truyền thống:

1. Import thư viện.
2. Đọc dữ liệu.
3. Phân tích dữ liệu khám phá.
4. Tiền xử lý văn bản.
5. Vector hóa TF-IDF.
6. Huấn luyện SVM.
7. Huấn luyện Logistic Regression.
8. Đánh giá trên validation/test.
9. Phân tích lỗi mô hình.

Đối với PhoBERT:

1. Kiểm tra GPU.
2. Cài đặt `torch` và `transformers`.
3. Tải tokenizer và model `vinai/phobert-base-v2`.
4. Chuẩn bị `Dataset` và `DataLoader`.
5. Huấn luyện mô hình cho từng nhãn.
6. Lưu checkpoint tốt nhất.
7. Đánh giá trên tập test.

---

## 12. Lưu ý khi chạy PhoBERT

PhoBERT là mô hình deep learning lớn hơn đáng kể so với SVM hoặc Logistic Regression. Vì vậy:

- Nên chạy trên GPU để giảm thời gian huấn luyện.
- Nếu chạy trên CPU, thời gian huấn luyện có thể rất lâu.
- Cần đảm bảo đã cài `torch`, `transformers` và `datasets`.
- Lần chạy đầu tiên sẽ tải mô hình `vinai/phobert-base-v2` từ Hugging Face.

Kiểm tra GPU:

```python
import torch

print(torch.cuda.is_available())
```

Nếu kết quả là:

```text
True
```

thì notebook có thể sử dụng GPU.

Nếu kết quả là:

```text
False
```

thì nên cân nhắc chạy trên Google Colab GPU hoặc giảm batch size, số epoch để tiết kiệm thời gian.

---

## 13. Nhận xét kết quả

Một số nhận xét chính từ kết quả thực nghiệm:

- Nhãn `Toxicity` bị mất cân bằng mạnh, khiến mô hình dễ thiên về lớp `0` là không độc hại.
- SVM + TF-IDF đạt accuracy cao nhất trên tập test cho cả hai nhãn.
- Với `Toxicity`, Logistic Regression có F1 lớp độc hại nhỉnh hơn SVM một chút, cho thấy accuracy cao không đồng nghĩa với khả năng phát hiện lớp độc hại tốt nhất.
- Với `Constructiveness`, SVM và Logistic Regression cho F1 lớp 1 tương đương nhau trên tập test.
- PhoBERT chưa vượt qua SVM trong notebook hiện tại, đặc biệt ở nhãn `Toxicity`.
- Kết quả PhoBERT trên `Constructiveness` ổn hơn so với `Toxicity`, nhưng vẫn chưa tạo ra khoảng cách rõ rệt so với mô hình truyền thống.
- Việc phân tích đóng góp TF-IDF giúp hiểu được các từ/ngram nào ảnh hưởng mạnh đến quyết định của mô hình.

Kết luận thực nghiệm quan trọng là: trong phiên bản hiện tại, **TF-IDF + SVM là lựa chọn thực tế và hiệu quả nhất nếu ưu tiên độ đơn giản, tốc độ huấn luyện và accuracy tổng thể**. Tuy nhiên, nếu mục tiêu chính là phát hiện bình luận độc hại, cần ưu tiên thêm F1/Recall của lớp `Toxicity = 1` thay vì chỉ nhìn accuracy.

---

## 14. Hạn chế hiện tại

Dự án vẫn còn một số hạn chế:

- Dữ liệu `Toxicity` bị mất cân bằng lớn.
- Tiền xử lý văn bản còn đơn giản, chưa xử lý toàn diện emoji, teencode phức tạp, lỗi chính tả, mỉa mai hoặc ngữ cảnh dài.
- TF-IDF chưa hiểu sâu quan hệ ngữ nghĩa giữa các từ.
- Các mô hình truyền thống dễ phụ thuộc vào từ khóa và n-gram bề mặt.
- PhoBERT cần được tinh chỉnh thêm để ổn định hơn, đặc biệt với nhãn `Toxicity`.
- Chưa có bước threshold tuning để tối ưu precision/recall cho lớp độc hại.
- Chưa có bước lưu pipeline SVM/Logistic Regression thành file `.pkl` để tái sử dụng.
- Chưa có giao diện demo hoặc API phục vụ dự đoán thực tế.
- Chưa đánh giá sâu theo từng `Topic`, nên chưa biết mô hình hoạt động tốt/kém ở chủ đề nào.

---

## 15. Hướng phát triển

Một số hướng cải tiến có thể thực hiện:

### 15.1. Cải thiện dữ liệu

- Bổ sung kỹ thuật cân bằng dữ liệu cho nhãn `Toxicity`.
- Thử oversampling cho lớp độc hại.
- Thu thập thêm bình luận độc hại để giảm mất cân bằng.
- Phân tích lỗi theo từng chủ đề để xác định nhóm dữ liệu khó.

### 15.2. Cải thiện tiền xử lý

- Mở rộng bộ từ điển teencode và viết tắt tiếng Việt.
- Xử lý emoji, icon, ký tự kéo dài như `quáaaaa`, `đẹpppp`.
- Chuẩn hóa lỗi chính tả phổ biến.
- Thử giữ lại dấu câu quan trọng vì dấu chấm than, dấu hỏi hoặc viết hoa có thể liên quan đến cảm xúc.

### 15.3. Cải thiện mô hình truyền thống

- Thử `ngram_range=(1,3)` hoặc `(1,4)` với kiểm soát số chiều đặc trưng.
- Thử `min_df=2` hoặc `min_df=3` để giảm nhiễu từ các từ quá hiếm.
- Tối ưu tham số `C` bằng grid search hoặc cross-validation.
- Lưu pipeline bằng `joblib` để tái sử dụng.

### 15.4. Cải thiện PhoBERT

- Dùng learning rate scheduler.
- Dùng early stopping.
- Thử nhiều learning rate khác nhau như `1e-5`, `2e-5`, `3e-5`.
- Thử tăng/giảm `MAX_LEN`.
- Thử focal loss hoặc weighted loss tốt hơn cho bài toán mất cân bằng.
- Theo dõi thêm macro F1, weighted F1 và F1 lớp độc hại.

### 15.5. Triển khai ứng dụng

- Xây dựng demo bằng Gradio hoặc Streamlit.
- Đóng gói mô hình thành API bằng FastAPI hoặc Flask.
- Tạo giao diện nhập bình luận và trả về dự đoán `Toxicity`, `Constructiveness`.
- Hiển thị thêm xác suất hoặc độ tin cậy của mô hình.
- Thêm phần giải thích từ/ngram nào ảnh hưởng đến quyết định dự đoán.

---

## 16. Ví dụ dự đoán

Sau khi huấn luyện pipeline SVM, có thể dự đoán một bình luận mới như sau:

```python
comment = "Bình luận này thật vô văn hóa và không thể chấp nhận được"

processed_comment = preprocess(comment)
prediction = svm_pipeline.predict([processed_comment])

print(prediction)
```

Kết quả trả về:

```text
0 hoặc 1
```

Ý nghĩa của nhãn phụ thuộc vào mô hình đang được huấn luyện cho bài toán nào:

- Nếu mô hình huấn luyện cho `Toxicity`, `1` nghĩa là bình luận độc hại.
- Nếu mô hình huấn luyện cho `Constructiveness`, `1` nghĩa là bình luận có tính xây dựng.

Nếu muốn sử dụng lại mô hình sau khi huấn luyện, có thể lưu pipeline bằng `joblib`:

```python
import joblib

joblib.dump(svm_pipeline, "models/svm_toxicity.pkl")
```

Tải lại mô hình:

```python
svm_pipeline = joblib.load("models/svm_toxicity.pkl")
```

---

## 17. Trích dẫn dataset

Dataset ViCTSD được giới thiệu trong bài báo:

```text
Constructive and Toxic Speech Detection for Open-Domain Social Media Comments in Vietnamese
```

BibTeX tham khảo:

```bibtex
@InProceedings{nguyen2021victsd,
    author="Nguyen, Luan Thanh and Van Nguyen, Kiet and Nguyen, Ngan Luu-Thuy",
    title="Constructive and Toxic Speech Detection for Open-Domain Social Media Comments in Vietnamese",
    booktitle="Advances and Trends in Artificial Intelligence. Artificial Intelligence Practices",
    year="2021",
    publisher="Springer International Publishing",
    address="Cham",
    pages="572--583"
}
```

Nguồn dataset sử dụng trong project:

```text
https://huggingface.co/datasets/tarudesu/ViCTSD
```

---

## 18. Tác giả

Dự án được xây dựng phục vụ mục đích học tập, thực nghiệm xử lý ngôn ngữ tự nhiên tiếng Việt và so sánh các mô hình phân loại văn bản.


```text
Họ tên: Nguyễn Văn Phú
MSSV: 23521187
Lớp: KHMT
Môn học: Xử lý ngôn ngữ tự nhiên - CS221
Giảng viên hướng dẫn: Nguyễn Trọng Chỉnh
```

---

## 19. Kết luận

Notebook đã triển khai một pipeline tương đối hoàn chỉnh cho bài toán phân loại bình luận tiếng Việt theo hai tiêu chí `Toxicity` và `Constructiveness`. Quy trình bao gồm các bước quan trọng như phân tích dữ liệu, tiền xử lý văn bản, vector hóa TF-IDF, huấn luyện mô hình, đánh giá kết quả và phân tích lỗi dự đoán.

Kết quả thực nghiệm cho thấy **SVM + TF-IDF** là lựa chọn hiệu quả và thực tế nhất trong phiên bản hiện tại nếu xét theo accuracy tổng thể. Mô hình này có ưu điểm là đơn giản, tốc độ huấn luyện nhanh, dễ triển khai và cho kết quả ổn định trên cả hai nhãn.

Tuy nhiên, với bài toán phát hiện bình luận độc hại, accuracy không phải là thước đo duy nhất cần quan tâm. Do lớp độc hại bị mất cân bằng mạnh, cần xem xét thêm F1-score, recall và precision của lớp `Toxicity = 1`. Trong hướng phát triển tiếp theo, dự án nên tập trung vào cân bằng dữ liệu, tối ưu threshold, cải thiện PhoBERT và xây dựng giao diện demo để phục vụ dự đoán thực tế.
