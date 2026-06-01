# MobileNetV3 with CBAM Attention for Image Classification

Proyek ini mengimplementasikan model MobileNetV3 (Large dan Small) dengan dukungan modul atensi Squeeze-and-Excitation (SE) dan Convolutional Block Attention Module (CBAM).

## Instalasi

1. Buat dan aktifkan environment conda:
```bash
conda create -n comvis python=3.11  
conda activate comvis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Struktur Proyek

```
MobileNetV3_CBAM/
├── data/               # Dataset gambar
├── results/            # Checkpoints, logs, dan hasil evaluasi
├── scripts/            # Script eksekusi (train, eval, complexity)
├── src/                # Source code modular
│   ├── models/         # Definisi arsitektur model
│   ├── config.py       # Pengaturan path terpusat
│   └── ...
└── requirements.txt
```

## Cara Penggunaan

### 1. Training Model
Eksekusi training melalui terminal menggunakan script di dalam folder `scripts/`:

```bash
python scripts/train.py --model mobilenetv3_large --epochs 50
```

**Pilihan Model (`--model`):**
- `mobilenetv3_small` (Standard SE)
- `mobilenetv3_large` (Standard SE)
- `proposed_large_16` (CBAM, Reduction Ratio 16)
- `proposed_large_32` (CBAM, Reduction Ratio 32)
- `proposed_small_16` (CBAM, Reduction Ratio 16)
- `proposed_small_32` (CBAM, Reduction Ratio 32)

### 2. Evaluasi Model
Gunakan script evaluasi untuk menguji model pada data test:

```bash
python scripts/eval.py --model proposed_large_16 --weight Nama_File_Weight
```
*Catatan: `--weight` adalah nama file di dalam `results/checkpoints/` tanpa ekstensi `.pth`.*

### 3. Pengukuran Kompleksitas
Untuk menghitung FLOPs dan jumlah parameter dari berbagai varian model:

```bash
python scripts/measure_complexity.py
```

## Konfigurasi
Seluruh path direktori diatur secara terpusat di `src/config.py`. Anda dapat menyesuaikan lokasi folder data atau hasil di file tersebut jika diperlukan.
