# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

---

## 1. Кратко: что сделано

- **Часть A (классификация):** Выбран датасет **STL10** — рекомендуется по умолчанию, содержит 10 классов, 96×96 изображения, хороший баланс между сложностью и размером для учебных экспериментов.
- **Часть B (detecion):** Выбран **Pascal VOC 2007** с треком **detection** — универсальный датасет для детекции объектов, доступен через `torchvision.datasets.VOCDetection`.
- **Сравнение:** 
  - Часть A: 4 эксперимента (C1-C4) — CNN vs ResNet, аугментации vs без, frozen backbone vs fine-tuning.
  - Часть B: 2 режима (V1-V2) — пороги уверенности 0.3 vs 0.7 для pretrained FasterRCNN.

---

## 2. Среда и воспроизводимость

| Параметр | Значение |
|----------|----------|
| **Python:** | 3.9+ |
| **torch / torchvision:** | 2.0+ |
| **Устройство:** | GPU |
| **Seed:** | 42 |
| **Как запустить:** | Открыть `HW10-11.ipynb` и выполнить Run All |

---

## 3. Данные

### 3.1. Часть A: классификация

| Параметр | Значение |
|----------|----------|
| **Датасет:** | STL10 |
| **Разделение:** | train=4000 (80%), val=1000 (20%), test=8000 (official) |
| **Базовые transforms:** | Resize(224) → ToTensor → Normalize(ImageNet) |
| **Augmentation transforms:** | RandomHorizontalFlip + RandomRotation(10) + ColorJitter |

**Комментарий:** STL10 содержит 10 классов (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck). Изображения 96×96, upscale до 224×224 для совместимости с ResNet. Датасет небольшой (5000 train + 8000 test), что делает transfer learning критически важным.

### 3.2. Часть B: structured vision

| Параметр | Значение |
|----------|----------|
| **Датасет:** | Pascal VOC 2007 |
| **Трек:** | detection |
| **Ground truth:** | Bounding boxes + class labels из XML-аннотаций |
| **Предсказания:** | FasterRCNN_ResNet50_FPN (pretrained на COCO) |

**Комментарий:** VOC содержит 20 классов объектов с bounding box аннотациями. Используется test split (4952 изображения). Модель pretrained на COCO (80 классов), поэтому требуется маппинг классов VOC→COCO для корректного сравнения.

---

## 4. Часть A: модели и обучение (C1-C4)

| Эксперимент | Описание |
|-------------|----------|
| **C1 (simple-cnn-base):** | SimpleCNN (4 conv блока + FC), без аугментаций |
| **C2 (simple-cnn-aug):** | SimpleCNN (та же архитектура), с аугментациями |
| **C3 (resnet18-head-only):** | ResNet18 pretrained, backbone frozen, обучается только FC |
| **C4 (resnet18-finetune):** | ResNet18 pretrained, layer4 + FC trainable |

| Параметр | Значение |
|----------|----------|
| **Loss:** | CrossEntropyLoss |
| **Optimizer:** | Adam (lr=1e-3) |
| **Batch size:** | 64 |
| **Epochs:** | 10 |
| **Критерий выбора лучшей модели:** | best_val_accuracy |

---

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

**Модель:** FasterRCNN_ResNet50_FPN (pretrained weights="DEFAULT")

| Режим | score_threshold |
|-------|-----------------|
| **V1:** | 0.3 |
| **V2:** | 0.7 |

**Как считался IoU:** 
$$\text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}}$$
Порог для True Positive: IoU ≥ 0.5

**Как считались precision / recall:**
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

---

## 6. Результаты

### Ссылки на файлы в репозитории:

| Артефакт | Путь |
|----------|------|
| Таблица результатов | `./artifacts/runs.csv` |
| Лучшая модель части A | `./artifacts/best_classifier.pt` |
| Конфиг лучшей модели | `./artifacts/best_classifier_config.json` |
| Кривые обучения | `./artifacts/figures/classification_curves_best.png` |
| Сравнение C1-C4 | `./artifacts/figures/classification_compare.png` |
| Визуализация аугментаций | `./artifacts/figures/augmentations_preview.png` |
| Detection примеры | `./artifacts/figures/detection_v1.png`, `detection_v2.png` |
| Detection метрики | `./artifacts/figures/detection_metrics.png` |

### Короткая сводка:

| Метрика | Значение |
|---------|----------|
| **Лучший эксперимент части A:** | C3 (ResNet18 Frozen) |
| **Лучшая val_accuracy:** | 94.8% |
| **Итоговая test_accuracy:** | 94.34% |
| **Что дали аугментации (C2 vs C1):** | +2.9% (55.1% vs 52.2%) |
| **Что дал transfer learning (C3/C4 vs C1/C2):** | +40% (94.8% vs 52.2%) |
| **Что оказалось лучше:** | head-only (C3) лучше fine-tuning (C4) на 2.7% |
| **V1 (threshold 0.3):** | Precision=0.011, Recall=0.031, FP=88 |
| **V2 (threshold 0.7):** | Precision=0.018, Recall=0.031, FP=55 |
| **Интерпретация:** | V2 снижает FP на 37%, precision растёт, recall стабильно низкий |

---

## 7. Анализ

**Простая CNN (C1/C2)** показывает ограниченные результаты (~52-55%) из-за малой глубины архитектуры (4 conv слоя) и отсутствия pretrained весов. STL10 — сложный датасет с большими вариациями внутри классов, что требует более выразительных признаков.

**Аугментации дали устойчивое улучшение** на +2.9% (C2 vs C1). RandomHorizontalFlip, RandomRotation и ColorJitter помогли модели стать инвариантной к небольшим трансформациям и предотвратили переобучение на маленьких данных.

**Pretrained ResNet18 значительно превосходит CNN** (+40% accuracy). Это демонстрирует критическую важность transfer learning: признаки, обученные на ImageNet (1.2M изображений), уже содержат универсальные детекторы краёв, текстур и паттернов.

**C3 (frozen) лучше C4 (fine-tune)** на 2.7% — неожиданный результат. Вероятные причины: (1) learning rate 1e-3 слишком высок для fine-tuning backbone — требуется ~1e-4; (2) 10 эпох недостаточно для сходимости при разморозке слоёв; (3) STL10 маленький — pretrained признаки уже оптимальны, fine-tuning может переобучить.

**Метрики детекции (precision/recall/IoU)** подходят под задачу локализации объектов. Accuracy недостаточна для детекции — важно оценивать качество bounding boxes через IoU и баланс находок/ложных срабатываний.

**Переход от V1 к V2** показал ожидаемый trade-off: при повышении порога с 0.3 до 0.7 количество False Positive снизилось с 88 до 55 (-37%), precision выросла с 0.011 до 0.018 (+64%), recall остался стабильно низким (0.031). Это подтверждает теорию: выше порог → меньше детекций → выше precision, ниже recall.

**Наиболее показательные ошибки:** (1) низкий recall указывает что модель пропускает большинство объектов — вероятно из-за mismatch классов VOC→COCO; (2) высокий FP говорит о ложных срабатываниях на фоновых объектах; (3) Mean IoU=0.775 высокий — когда модель находит объект, bounding box точный.

---

## 8. Итоговый вывод

**Базовый конфиг классификации:** ResNet18 Frozen (C3) с аугментациями. Transfer learning даёт +40% accuracy по сравнению с CNN с нуля — это критически важно для малых датасетов.

**Главное про transfer learning:** Pretrained веса на ImageNet — это "бесплатные" признаки, которые экономят время обучения и данные. Head-only training часто достаточно для адаптации под новую задачу.

**Главное про detection/метрики:** Для детекции accuracy не подходит — нужны precision, recall, IoU. Порог уверенности — ключевой гиперпараметр для баланса между находками и ложными срабатываниями. При оценке важно учитывать маппинг классов между датасетом и pretrained моделью.

---
