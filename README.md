# Radar Sinyal Sınıflandırma ve Gürültü Giderme

Radar spektrogramları üzerinde derin öğrenme tabanlı **gürültü giderme (denoising)**,
**çok etiketli sinyal sınıflandırma** ve gürültüden arındırılmış spektrogram üzerinde
**kenar/parametre çıkarımı** için MATLAB kodlarını içerir.

İşlenen sinyal türleri: `LFM`, `NLFM`, `FSK`, `Barker`, `Frank`, `P_All` (poli-faz), `none`.

## İşlem Hattı (pipeline)

```
1. Veri seti üret        dataset/generate_patch_dataset.m
2. Denoiser eğit         training/train_denoiser_unet_resnet18.m
3. Gürültüsüz set üret   denoising/denoise_dataset.m
4. Sınıflandırıcı eğit   training/train_classifier_noisy.m
                         training/train_classifier_denoised.m
5. Değerlendir           evaluation/eval_models_gray.m  (veya eval_models_rgb.m)
6. Kenar/parametre       edge_detection/demo_edge_detection.m
```

## Klasör Yapısı

| Klasör | İçerik |
|--------|--------|
| `dataset/` | Veri seti üretimi ve örnek görselleştirme |
| `training/` | Denoiser ve sınıflandırıcı eğitim betikleri |
| `denoising/` | Eğitilmiş denoiser ile gürültüsüz veri seti üretimi |
| `evaluation/` | SNR'a göre performans ölçümü ve grafik üretimi |
| `demo/` | Denoiser kullanım demosu |
| `edge_detection/` | Gürültüsüz spektrogramda kenar/blob tespiti ve RDW çıkarımı |
| `models/` | Eğitilmiş model dosyaları (`.mat`) |
| `results/` | Grafikler ve metrik tabloları |
| `Literatür/` | Referans makaleler |
| `presentation/` | LaTeX Beamer sunumu |

## Dosyalar

**dataset/**
- `generate_patch_dataset.m` — Çok çerçeveli sahne simülasyonundan gürültülü + temiz
  spektrogram yamaları, çok etiketli etiketler ve meta veri üretir.
- `show_dataset_examples.m` — Veri setinden 3×3 örnek (gürültülü / temiz / fark) çizer.

**training/**
- `train_denoiser_unet_resnet18.m` — ResNet-18 enkoder + U-Net dekoder tabanlı
  gürültü giderme ağı (regresyon, checkpoint + resume destekli).
- `train_classifier_noisy.m` — GoogLeNet ile gürültülü spektrogram üzerinde çok etiketli sınıflandırıcı.
- `train_classifier_denoised.m` — Aynı mimari, gürültüsü giderilmiş spektrogram üzerinde.

**denoising/**
- `denoise_dataset.m` — Eğitilmiş denoiser'ı tüm gürültülü yamalara uygular ve
  `specDenoisedGray` klasörünü üretir.

**evaluation/**
- `eval_models_gray.m` — Gri girişli denoiser + sınıflandırıcıların SNR'a göre
  PSNR/SSIM ve F1/subset-accuracy metriklerini hesaplar.
- `eval_models_rgb.m` — RGB girişli sınıflandırıcı varyantı için aynı değerlendirme.

**demo/**
- `demo_denoiser_visual.m` — Bir örnek üzerinde CLEAN / NOISY / DENOISED panellerini
  ve PSNR/SSIM değerlerini gösterir.

**edge_detection/**
- `demo_edge_detection.m` — Kenar tespiti pipeline demosu (simülasyon → denoise →
  eşikleme → CCA maskesi → sınır kutuları).
- `radar_descriptive_word.m` — Spektrogramdan olay/küme çıkarıp PRI/PW/BW/Fc
  tabanlı radar tanımlayıcı kelime (RDW) üretir.
- `auto_tune_threshold_params.m` — Eşikleme parametrelerini yer gerçeğine göre ayarlar.
- `test_otsu_params.m` — Otsu eşik parametrelerini tarar.

## Notlar

- Betiklerin başındaki `dataRoot` yolları geliştirme ortamına aittir; kendi
  ortamınıza göre güncelleyin.
- Model/checkpoint dosyaları ve üretilen veri setleri boyut nedeniyle depoya
  dahil edilmez (bkz. `.gitignore`).
- Çıktı klasörleri (`models/`, `results/`, `checkpoints/`) betikler tarafından
  proje köküne göre otomatik oluşturulur.
