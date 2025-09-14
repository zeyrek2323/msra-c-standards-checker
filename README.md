# Llama3 LoRA Fine-tuning for MSRA C Standard Dataset

Bu proje, Llama3 modelini MSRA C standardı dataset'i ile LoRA (Low-Rank Adaptation) kullanarak fine-tuning yapmak için tasarlanmıştır.

## 🚀 Özellikler

- **LoRA Fine-tuning**: Bellek verimli fine-tuning için LoRA kullanımı
- **4-bit Quantization**: BitsAndBytes ile bellek optimizasyonu
- **MSRA Dataset**: C programlama dili standartları için özel dataset
- **Instruction Following**: Instruction-following formatında eğitim
- **Wandb Integration**: Eğitim sürecini takip etmek için

## 📋 Gereksinimler

- Python 3.8+
- CUDA uyumlu GPU (en az 8GB VRAM önerilir)
- Hugging Face hesabı ve Llama3 model erişimi

## 🛠️ Kurulum

1. **Repository'yi klonlayın:**
```bash
git clone <repository-url>
cd dataset
```

2. **Gerekli paketleri yükleyin:**
```bash
pip install -r requirements.txt
```

3. **Hugging Face token'ınızı ayarlayın:**
```bash
huggingface-cli login
```

## 📊 Dataset

MSRA C standardı dataset'i şu formatta yapılandırılmıştır:
- **Instruction**: C kodu analizi için talimat
- **Input**: Analiz edilecek C kodu
- **Output**: MSRA standardına uygunluk değerlendirmesi

## 🎯 Kullanım

### 1. Eğitim

```bash
python train_lora.py
```

**Önemli Notlar:**
- `config.py` dosyasında `BASE_MODEL_NAME`'i kendi Llama3 modelinizle değiştirin
- GPU bellek yetersizse batch size'ı düşürün
- Eğitim süresi dataset boyutuna ve hardware'e bağlı olarak değişir

### 2. Inference

```bash
python inference.py
```

Bu script eğitilmiş modeli yükler ve test eder.

## ⚙️ Konfigürasyon

`config.py` dosyasında aşağıdaki parametreleri ayarlayabilirsiniz:

- **LoRA Parameters**: Rank, alpha, dropout
- **Training Parameters**: Epochs, batch size, learning rate
- **Hardware Settings**: Quantization, device mapping

## 📁 Dosya Yapısı

```
dataset/
├── dataset_msra_instruct.jsonl    # MSRA dataset
├── train_lora.py                  # Ana eğitim script'i
├── inference.py                   # Inference script'i
├── config.py                      # Konfigürasyon dosyası
├── requirements.txt               # Python dependencies
└── README.md                     # Bu dosya
```

## 🔧 LoRA Konfigürasyonu

```python
lora_config = LoraConfig(
    r=16,                          # Rank
    lora_alpha=32,                 # Alpha scaling
    target_modules=[               # Hedef modüller
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

## 📈 Eğitim Metrikleri

Eğitim sırasında aşağıdaki metrikler takip edilir:
- Training loss
- Evaluation loss
- Learning rate
- GPU memory usage

## 🚨 Sorun Giderme

### Yaygın Hatalar:

1. **CUDA Out of Memory**: Batch size'ı düşürün veya gradient accumulation steps'i artırın
2. **Model Loading Error**: Hugging Face token'ınızı kontrol edin
3. **Dataset Format Error**: JSONL formatını kontrol edin

### Bellek Optimizasyonu:

- 4-bit quantization kullanın
- Gradient checkpointing aktif edin
- Batch size'ı optimize edin

## 📝 Örnek Kullanım

```python
from config import TrainingConfig

# LoRA konfigürasyonu
lora_config = TrainingConfig.get_lora_config()

# Eğitim parametreleri
training_args = TrainingConfig.get_training_args()
```

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 🙏 Teşekkürler

- Hugging Face ekibine
- PEFT (Parameter-Efficient Fine-Tuning) geliştiricilerine
- MSRA dataset'i hazırlayanlara

## 📞 İletişim

Sorularınız için issue açabilir veya pull request gönderebilirsiniz.

