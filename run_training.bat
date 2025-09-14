@echo off
echo ========================================
echo Llama3 LoRA Fine-tuning Başlatılıyor
echo ========================================
echo.

echo Gerekli paketler yükleniyor...
pip install -r requirements.txt

echo.
echo Hugging Face token kontrol ediliyor...
huggingface-cli whoami

echo.
echo Eğitim başlatılıyor...
python train_lora.py

echo.
echo Eğitim tamamlandı!
pause

