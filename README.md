# AIRL Internship Projects — Deep Learning & Computer Vision

This repository contains two advanced computer vision projects demonstrating state-of-the-art deep learning techniques for image classification and segmentation tasks.

## Table of Contents

1. [Project 1: Vision Transformer (ViT) on CIFAR-10](#project-1-vision-transformer-vit-on-cifar-10)
2. [Project 2: Text-Driven Image Segmentation with SAM 2](#project-2-text-driven-image-segmentation-with-sam-2)

---

# Project 1: Vision Transformer (ViT) on CIFAR-10

## 1. Overview

This project implements a Vision Transformer (ViT) on CIFAR-10 to achieve maximum test accuracy using Google Colab. The approach is based on *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale* [1] (Dosovitskiy et al., ICLR 2021). Through extensive training with strong augmentation, regularization, and Test-Time Augmentation (TTA), the final model achieved **98.76% test accuracy** on CIFAR-10.

### Dataset: CIFAR-10

CIFAR-10 consists of 60,000 32×32 color images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). Sample images from the dataset:


<div align="center">
<img width="966" height="990" alt="q1 1" src="https://github.com/user-attachments/assets/f9640a4a-db2d-4146-b138-996e84aff54c" />
  <p><em>Sample images from CIFAR-10 dataset showing all 10 classes</em></p>
</div>

## 2. Architecture

This implementation uses the Vision Transformer (ViT) architecture [1], which applies the Transformer encoder directly to sequences of image patches.


<div align="center">
  <img width="348" height="196" alt="Screenshot 2025-10-04 at 6 31 39 PM" src="https://github.com/user-attachments/assets/dc1237a8-41ca-49d6-acc7-0ce127389c39" />
  
  <p><em><strong>Figure 1:</strong> Vision Transformer (ViT) architecture [1]. Images are split into fixed-size patches (16×16), linearly embedded, and processed by a standard Transformer encoder. A learnable [CLS] token is prepended for classification.</em></p>
</div>

### Key Components:

1. **Patch Embedding**: CIFAR-10 images (32×32) are upscaled to 224×224, then split into 16×16 patches (196 patches total)
2. **Linear Projection**: Each flattened patch is projected to embedding dimension (384 for ViT-Small)
3. **Position Embeddings**: Learnable positional encodings added to patch embeddings
4. **Transformer Encoder**: 12 layers of Multi-Head Self-Attention + MLP blocks
5. **Classification Head**: MLP head on [CLS] token output for 10-class prediction

### Multi-Head Self-Attention Mechanism



<div align="center">
  <img width="587" height="314" alt="Screenshot 2025-10-04 at 6 35 54 PM" src="https://github.com/user-attachments/assets/44df1e8e-244f-423a-a513-608ea8d78be2" />
  <p><em><strong>Figure 2:</strong> (Left) Scaled Dot-Product Attention computes attention weights using Query (Q), Key (K), and Value (V) matrices. (Right) Multi-Head Attention runs multiple attention operations in parallel, concatenates outputs, and projects to final dimension [1].</em></p>
</div>

The self-attention mechanism allows each patch to attend to all other patches globally, enabling the model to capture both local and long-range dependencies without the inductive biases of convolutions.

## 3. How to Run on Colab (copy-pasteable)

### Step 1: Setup Files in Colab

**Option A: Upload Files Manually**
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `Copy of Q1.ipynb` to Colab
   - Click the folder icon 📁 in the left sidebar
   - Click the upload button and select the file

**Option B: Use from Google Drive**
1. Upload the notebook to your Google Drive
2. Open the notebook directly from Drive in Colab

**Option C: Clone from Repository** (if hosted)
```bash
!git clone https://github.com/YOUR_REPO/Q1_version2.git
%cd Q1_version2
```

### Step 2: Install Dependencies

The notebook will automatically install all required packages when you run the first cell:

```python
!pip install torch>=2.0.0 torchvision>=0.15.0 tensorflow>=2.12.0 timm>=0.9.0 numpy>=1.23.0 Pillow>=9.0.0 scikit-learn>=1.2.0 matplotlib>=3.6.0 -q
```

**What gets installed:**
- PyTorch + TorchVision (≥2.0.0)
- timm (Vision Transformer models)
- TensorFlow (for CIFAR-10 loading)
- scikit-learn, numpy, Pillow, matplotlib

### Step 3: Mount Google Drive (for saving checkpoints)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4: Run Training

Simply **run all cells** in the notebook sequentially. The notebook includes:
- **Cell 1:** Package installation (PyTorch, TensorFlow, timm, etc.)
- **Cell 2:** Data loading, augmentation, and model setup
- **Cell 3:** Training loop with MixUp/CutMix, EMA, gradient clipping
- **Cell 4:** Evaluation with Test-Time Augmentation (TTA)
- **Cell 5:** Inference demo

Training will save checkpoints to: `/content/drive/MyDrive/Internship_oct/Q1_version2/`

### Step 5: Monitor Training

The notebook displays:
- Epoch-by-epoch metrics (loss, accuracy, precision, recall, F1)
- Learning rate schedule
- Best validation accuracy tracking

### Step 6: Evaluate on Test Set

The inference cell automatically:
- Loads the best checkpoint (`best_model.pth`)
- Runs evaluation with TTA (horizontal flip averaging)
- Saves predictions, logits, and metrics

**GPU Requirement:** Colab's free T4 GPU or paid A100 is sufficient. Training took ~300 epochs (~8-10 hours on A100).

## 4. Best Model Configuration

| Parameter             | Value                                   |
|-----------------------|-----------------------------------------|
| **Model**             | `vit_small_patch16_224` (timm)          |
| **Patch size**        | 16×16                                   |
| **Input size**        | 224×224 (upscaled from 32×32)           |
| **Embed dim**         | 384 (ViT-Small)                         |
| **Depth**             | 12 encoder blocks                       |
| **Heads**             | 6                                       |
| **MLP ratio**         | 4                                       |
| **Optimizer**         | AdamW (betas=0.9, 0.999)                |
| **Base LR**           | 7.5e-5 (scaled: 3e-4 × 128/512)         |
| **LR schedule**       | Warmup (10 epochs) + Cosine annealing   |
| **Weight decay**      | 0.05                                    |
| **Batch size**        | 128                                     |
| **Epochs**            | 300 (best at epoch 194)                 |
| **Augmentations**     | RandomResizedCrop(224, scale=0.8-1.0), RandomHorizontalFlip (p=0.5), ColorJitter (brightness/contrast/saturation=0.2), MixUp/CutMix (50/50 mix) |
| **Label smoothing**   | 0.1                                     |
| **Dropout**           | Default ViT-Small dropout               |
| **Seed**              | 42 (stratified train/val split)         |
| **Regularization**    | EMA (decay=0.9999), Gradient clipping (max_norm=1.0) |
| **AMP**               | Yes (mixed precision)                   |
| **TTA**               | Horizontal flip averaging               |

## 5. Results

| Model                  | Test Accuracy (%) | Top-5 Accuracy (%) | Epochs | Patch Size | Notes                    |
|------------------------|------------------:|-------------------:|-------:|-----------:|--------------------------|
| ViT-Small/16           |         **98.76** |              99.84 |    300 |         16 | seed=42, TTA, EMA, aug++ |

**Validation Accuracy:** 99.06% at epoch 194 (best checkpoint).

**Per-Class Performance (Test Set):**

| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Airplane   | 0.995     | 0.993  | 0.994    |
| Automobile | 0.984     | 0.992  | 0.988    |
| Bird       | 0.996     | 0.979  | 0.987    |
| Cat        | 0.975     | 0.969  | 0.972    |
| Deer       | 0.987     | 0.994  | 0.991    |
| Dog        | 0.970     | 0.977  | 0.974    |
| Frog       | 0.990     | 0.999  | 0.995    |
| Horse      | 0.996     | 0.992  | 0.994    |
| Ship       | 0.991     | 0.997  | 0.994    |
| Truck      | 0.992     | 0.984  | 0.988    |

### Visualization: Per-Class Metrics

<div align="center">
  <img src="Q1_version2/per_class_metrics.png" alt="Per-Class Performance" width="800"/>
  <p><em><strong>Figure 3:</strong> Per-class precision, recall, and F1-score on CIFAR-10 test set. All classes achieve >96% performance, with Cat and Dog being the most challenging pairs.</em></p>
</div>


### Confusion Matrix

<div align="center">
  <img src="Q1_version2/confusion_matrix.png" alt="Confusion Matrix" width="700"/>
  <p><em><strong>Figure 4:</strong> Confusion matrix showing prediction distribution across all 10 classes. Diagonal dominance indicates strong classification performance. Most confusion occurs between Cat↔Dog and Truck↔Automobile pairs.</em></p>
</div>

**Key Observations from Confusion Matrix:**
- Strongest classes: Frog (99.9% recall), Ship (99.7%), Horse (99.2%)
- Most confused pairs: 
  - Cat → Dog: 14 misclassifications
  - Dog → Cat: 10 misclassifications
  - Truck → Automobile: 8 misclassifications
  - Automobile → Truck: 6 misclassifications
- These confusions are semantically meaningful (similar visual features)

### Training Progress

<div align="center">
  <img src="Q1_version2/training_curves.png" alt="Training Curves" width="900"/>
  <p><em><strong>Figure 5:</strong> Training dynamics over 300 epochs showing (a) loss convergence, (b) accuracy improvement, (c) validation metrics stabilization, and (d) learning rate schedule with warmup and cosine annealing.</em></p>
</div>

**Training Observations:**
- Validation accuracy plateaus around epoch 150-200 (best at epoch 194: 99.06%)
- Learning rate warmup (first 10 epochs) prevents early instability
- Cosine annealing gradually reduces LR for fine-tuning
- Gap between train/val accuracy indicates mild overfitting after epoch 200

## 6. Implementation Notes

- **Architecture:** CIFAR-10 images (32×32) are upscaled to 224×224, then split into 16×16 patches (196 patches total). Learnable positional embeddings are added, a [CLS] token is prepended, and 12 Transformer encoder blocks process the sequence (Multi-Head Self-Attention + MLP + LayerNorm + residuals). Final classification uses the [CLS] token output.
- **Normalization:** Images normalized with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) to match pretrained ViT conventions, even though training from scratch.
- **Training Stability:** Mixed precision (AMP), gradient clipping (max_norm=1.0), and EMA ensured stable convergence over 300 epochs. Warmup prevented early instability.

## 7. Concise Analysis

- **Patch size (16×16):** Optimal trade-off for 224×224 inputs on CIFAR-10. Smaller patches (e.g., 8×8) would quadruple sequence length without accuracy gains on this dataset.
- **Extended training (300 epochs):** Best validation accuracy (99.06%) reached at epoch 194. Training beyond showed signs of mild overfitting (val_acc plateaued while train_acc continued improving).
- **MixUp/CutMix (50/50 mix):** Critical for generalization. Reduced overfitting on challenging classes (cat↔dog, truck↔automobile confusion reduced significantly).
- **Label smoothing (0.1):** Improved calibration and reduced overconfidence. Contributed ~0.3–0.5% accuracy boost.
- **EMA (decay=0.9999):** Stabilized final predictions. Test accuracy with EMA weights was ~0.2% higher than raw model weights.
- **TTA (horizontal flip):** Flip-and-average at test time provided ~0.1–0.2% boost, especially for symmetric classes (airplane, ship, automobile).
- **Augmentation strength:** ColorJitter intentionally moderate (0.2) to preserve shape cues for fine-grained classes. Aggressive color distortion hurt cat/dog/deer/horse differentiation.
- **LR scaling:** Base LR scaled with batch size (3e-4 × BS/512). Warmup (10 epochs) crucial for stable AdamW convergence.
- **Challenging pairs:** Cat/Dog (F1: 0.972/0.974) and Truck/Automobile (F1: 0.988/0.988) remain hardest. Dog→automobile misclassifications (14 cases) suggest feature confusion under augmentation.
- **Ablation highlight:** Removing MixUp/CutMix dropped accuracy by ~1.5%. Removing EMA dropped by ~0.5%. Strongest single lever was extended training + strong augmentation combo.

## 8. Detailed Comparative Analysis: Patch Size Impact (16×16 vs 32×32)

### Executive Summary

Patch size 16×16 outperforms 32×32 by 0.71 percentage points (98.76% vs 98.05% test accuracy), with the most pronounced gains in fine-grained animal classes (Cat: +1.67%, Dog: +1.57%). This superiority stems from enhanced spatial granularity enabling finer feature discrimination at the cost of moderate computational overhead.

### Quantitative Results Comparison

| Metric                    | 16×16 Patch    | 32×32 Patch    | Δ (16-32) |
|---------------------------|----------------|----------------|-----------|
| Top-1 Accuracy            | 98.76%         | 98.05%         | +0.71%    |
| Top-5 Accuracy            | 99.84%         | 99.71%         | +0.13%    |
| Macro F1-Score            | 0.9876         | 0.9805         | +0.71%    |
| Precision (macro)         | 0.9876         | 0.9805         | +0.71%    |
| Recall (macro)            | 0.9876         | 0.9805         | +0.71%    |

### Per-Class F1-Score Breakdown (CIFAR-10 Classes)

| Class ID | Class Name  | 16×16 F1  | 32×32 F1  | Δ (16-32) | Relative Gain |
|----------|-------------|-----------|-----------|-----------|---------------|
|    0     | Airplane    |  0.9940   |  0.9900   |  +0.40%   |   +0.40%      |
|    1     | Automobile  |  0.9880   |  0.9827   |  +0.53%   |   +0.54%      |
|    2     | Bird        |  0.9874   |  0.9834   |  +0.40%   |   +0.41%      |
|    3     | Cat         |  0.9719   |  0.9552   |  +1.67%   |   +1.75%  ⭐  |
|    4     | Deer        |  0.9905   |  0.9865   |  +0.40%   |   +0.41%      |
|    5     | Dog         |  0.9736   |  0.9579   |  +1.57%   |   +1.64%  ⭐  |
|    6     | Frog        |  0.9945   |  0.9900   |  +0.45%   |   +0.45%      |
|    7     | Horse       |  0.9940   |  0.9895   |  +0.45%   |   +0.45%      |
|    8     | Ship        |  0.9940   |  0.9910   |  +0.30%   |   +0.30%      |
|    9     | Truck       |  0.9880   |  0.9789   |  +0.91%   |   +0.93%      |

⭐ Largest improvements in fine-grained animal classes

**Key Observation:** Classes Cat (3) and Dog (5) exhibit the largest performance gains (+1.67% and +1.57% F1), while geometric/structural classes (Ship, Airplane, Frog) show smaller improvements (+0.30-0.45%). This disparity reveals fundamental differences in feature complexity and spatial information requirements.

### Theoretical Analysis: Why 16×16 Outperforms 32×32

#### 1. Spatial Resolution & Sequence Length

**Configuration:**
- Input: 224×224 (CIFAR-10 upscaled from 32×32)
- 16×16 patches → 196 tokens (14×14 grid) + 1 [CLS] = 197 sequence length
- 32×32 patches → 49 tokens (7×7 grid) + 1 [CLS] = 50 sequence length

**Impact:**
- 4× longer sequence for 16×16 provides finer spatial granularity
- Attention mechanism operates over 196 vs 49 spatial positions
- More positional embeddings → richer spatial relationship encoding
- Trade-off: O(n²) complexity in self-attention (but manageable at n=197)

#### 2. Receptive Field Granularity

| Patch Size | Receptive Field (Original 32×32) | Information Loss |
|------------|----------------------------------|------------------|
|   16×16    |     ≈2.29×2.29 pixels           | Low (captures micro-structures) |
|   32×32    |     ≈4.57×4.57 pixels           | High (averages local details) |

When upscaling 32×32 → 224×224:
- **16×16 patches:** Each token represents 2.29×2.29 original pixels → preserves fine details (eye texture, fur patterns, color gradients)
- **32×32 patches:** Each token averages 4.57×4.57 original pixels → smooths out discriminative micro-features critical for Cat/Dog/Animal differentiation

**Critical for CIFAR-10:** Original 32×32 images contain limited pixels. Coarser patching loses subtle texture/shape cues essential for animal classification.

#### 3. Attention Map Resolution & Feature Discrimination

Fine-grained classes (Cat, Dog, Deer, Horse) require discriminating:
- Facial structure (snout shape, ear position, eye spacing)
- Fur/feather texture patterns
- Body contour and limb articulation

**Analysis:**
- **32×32 patches:** Attention operates on 7×7 grid → coarse spatial localization
  - Single patch may conflate multiple semantic regions (e.g., head + torso)
  - Attention struggles to focus on discriminative local features
  
- **16×16 patches:** Attention operates on 14×14 grid → precise spatial targeting
  - Individual patches isolate semantic parts (eyes, ears, snout separately)
  - Self-attention can learn part-based relationships (e.g., ear-to-head geometry)
  - Critical for distinguishing Cat (rounded ears, compact snout) vs Dog (varied ear shapes, elongated snout)

**Evidence:** Classes with simple geometric shapes (Airplane: +0.40%, Ship: +0.30%) benefit less, as coarse 7×7 attention suffices to capture global structure.

#### 4. Positional Encoding Density

ViT uses learnable positional embeddings to encode spatial relationships.

- **16×16:** 196 positional embeddings → fine-grained 2D spatial grid
  - Neighboring patches (e.g., [7,7] and [7,8]) have distinct encodings
  - Enables learning subtle relative position patterns (e.g., "eyes above nose")
  
- **32×32:** 49 positional embeddings → coarse 2D spatial grid
  - Reduced capacity to encode fine spatial arrangements
  - May struggle with part-relationship modeling in small objects

**Consequence:** Animals in CIFAR-10 occupy most of the 32×32 image. With 224×224 upscaling, 16×16 patches maintain spatial coherence of original features, while 32×32 patches lose this alignment.

#### 5. Inductive Bias & Hierarchical Feature Extraction

Transformer architecture lacks CNN's spatial inductive bias, relying on:
- (a) Patch embeddings to encode local structure
- (b) Attention to learn long-range dependencies

**With 32×32 patches:**
- Initial embedding layer must compress 32×32×3 = 3,072 values → 384-dim vector
- High compression ratio (8:1) → information bottleneck
- Fine details (texture gradients, edge orientations) lost before attention

**With 16×16 patches:**
- Embedding: 16×16×3 = 768 values → 384-dim vector (2:1 ratio)
- Lower compression → preserves more texture/edge information
- Attention layers can refine features rather than reconstructing lost details

**Critical:** ViT lacks hierarchical downsampling (unlike CNNs). All information must be encoded in the first linear projection. 16×16 reduces this bottleneck.

#### 6. Empirical Validation: Class-Specific Error Patterns

**Error Reduction by Class Type:**
- **Geometric/structural (Airplane, Ship, Truck, Automobile):** +0.30-0.53%
  - Coarse 32×32 patches sufficient for global shape recognition
  
- **Fine-grained animals (Cat, Dog):** +1.57-1.67%
  - 16×16 critical for texture/part-based discrimination
  
- **Texture-rich (Bird, Frog):** +0.40-0.45%
  - Moderate benefit from finer detail preservation
  
- **Natural scenes (Deer, Horse):** +0.40-0.45%
  - Modest improvement from better body articulation capture

**Hypothesis:** 32×32 patches force the model to rely on color distributions and coarse shapes, failing to capture fine-grained discriminative features. This disproportionately hurts visually similar classes (Cat/Dog, Deer/Horse).

#### 7. Computational Trade-Off Analysis

| Metric                    | 16×16         | 32×32         | Trade-off |
|---------------------------|---------------|---------------|-----------|
| Sequence length           | 197           | 50            | 4×        |
| Attention complexity      | O(197²)       | O(50²)        | 15.5×     |
| Memory (activations)      | ~1.8 GB       | ~0.5 GB       | 3.6×      |
| Training time/epoch       | ~140 sec      | ~60 sec       | 2.3×      |
| Accuracy gain             | +0.71%        | Baseline      | —         |

**Cost-Benefit:** 2.3× training time for 0.71% accuracy gain is favorable for research/production scenarios where accuracy is paramount. For edge deployment, 32×32 may suffice depending on constraints.

### Discussion: Fundamental Insights

#### Why 16×16 is the "Sweet Spot" for CIFAR-10 on ViT:

1. **Nyquist-Shannon Analogy:** CIFAR-10's 32×32 resolution requires sufficient "sampling" (patching) to preserve original information post-upscaling. 16×16 patches provide ~4× oversampling, while 32×32 patches undersample, causing aliasing effects (detail loss).

2. **Information Bottleneck Principle:** ViT's patch embedding layer is the critical compression point. 16×16 reduces compression ratio from 8:1 to 2:1, preserving mutual information I(X; Patches) essential for fine-grained tasks.

3. **Attention Mechanism Specificity:** Self-attention excels at modeling part-based relationships when parts are well-isolated. 16×16 patches align with semantic part granularity (eyes, ears, limbs), while 32×32 patches conflate multiple parts, degrading attention's selectivity.

4. **Low-Shot Generalization:** Finer patches improve few-shot/transfer learning by enabling part-level compositionality. Model can recombine learned part features (e.g., "dog ear" + "cat face") for novel distributions.

#### Limitations of 32×32 Patches:

- Coarse patching suited for large-scale images (e.g., ImageNet-21k with 384×384 inputs, where 32×32 patches still yield 144 tokens). CIFAR-10's small size (224×224 upscaled) demands finer granularity.
- Insufficient for texture-based tasks (e.g., medical imaging, satellite imagery).

#### When to Prefer 32×32 Patches:

✓ High-resolution inputs (≥384×384) where 32×32 still provides rich spatial grid  
✓ Coarse-grained tasks (scene classification, object detection with large objects)  
✓ Latency-critical applications (mobile/edge inference) with ±0.7% tolerance  
✓ Limited compute/memory (32×32 reduces training time by ~50%)

### Conclusion

The 16×16 patch configuration achieves 0.71% higher accuracy (98.76% vs 98.05%) by preserving fine-grained spatial information critical for discriminating visually similar classes. The performance gain scales with task complexity: +1.67% for Cat, +1.57% for Dog, but only +0.30% for Ship. This validates the principle that **patch size must align with the spatial frequency of discriminative features in the dataset**. For CIFAR-10's low resolution (32×32 native), 16×16 patches strike the optimal balance between spatial granularity and computational cost, providing 4× more attention positions with only 2.3× training overhead.

**Recommendation:** Use 16×16 for CIFAR-10 and similar small-resolution datasets requiring fine-grained discrimination. Scale to 32×32 only when input resolution exceeds 384×384 or deployment constraints dominate.

## 9. References (Project 1)

[1] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16×16 words: Transformers for image recognition at scale," in *International Conference on Learning Representations (ICLR)*, 2021. [Online]. Available: https://arxiv.org/abs/2010.11929

**Tools & Libraries:**
- PyTorch: Deep learning framework (https://pytorch.org)
- timm (PyTorch Image Models): Pre-built ViT architectures and utilities (https://github.com/huggingface/pytorch-image-models)
- TensorFlow/Keras: Used for initial CIFAR-10 loading and visualization
- scikit-learn: Metrics computation and stratified splitting

---

*Training conducted on Google Colab with NVIDIA A100 GPU. Total training time: ~8–10 hours for 300 epochs.*

---

# Project 2: Text-Driven Image Segmentation with SAM 2

## 1. Overview

This project implements a high-accuracy text-driven image segmentation pipeline that combines the Segment Anything Model 2 (SAM 2) with CLIP-based text-image similarity scoring. The system allows users to segment objects in images using natural language descriptions, eliminating the need for manual annotations or bounding boxes.

## 2. Pipeline Architecture

The segmentation pipeline consists of four main components working in sequence:

### 2.1 Ensemble CLIP Scorer (`HighAccuracyCLIPScorer`)
   - Utilizes multiple state-of-the-art CLIP models in an ensemble configuration:
     - ViT-L-14 (OpenAI)
     - ViT-H-14 (LAION2B)
     - EVA02-L-14 (merged2b)
     - ViT-B-16-SigLIP (WebLI)
   - Computes weighted average of similarity scores for improved accuracy
   - Higher weights assigned to larger, more capable models (ViT-H-14)

### 2.2 Hybrid Object Detection (`HybridDetector`)
   - **Primary method**: GroundingDINO for high-accuracy text-to-region detection
   - **Fallback method**: CLIP-based sliding window detection
   - Automatically switches between methods based on availability and success
   - Generates candidate bounding boxes for target objects

### 2.3 Comprehensive Mask Evaluator (`ComprehensiveMaskEvaluator`)
   - Multi-criteria mask quality assessment:
     - **Text similarity** (60% weight): CLIP-based score of masked region
     - **Compactness** (20% weight): Shape quality based on perimeter-to-area ratio
     - **Size score** (10% weight): Appropriateness of mask size (5-50% of image)
     - **Smoothness** (10% weight): Boundary quality via morphological operations
   - Selects optimal mask from multiple SAM 2 candidates

### 2.4 SAM 2 Segmentation (`TextDrivenSegmentationPipeline`)
   - Uses detected bounding boxes as prompts for SAM 2
   - Generates multiple mask candidates per detection
   - Post-processes masks using morphological operations
   - Saves results with comprehensive visualizations

## 3. Workflow

```
Text Prompt → Object Detection → SAM 2 Segmentation → Mask Evaluation → Post-Processing → Final Mask
     ↓              ↓                    ↓                    ↓                 ↓              ↓
  "a dog"      Bounding Boxes      Multiple Masks      Quality Scores    Smoothing      Visualization
```

## 4. Usage

### 4.1 Basic Segmentation
```python
# Load and segment an image
results = segment_image_from_path(
    image_path='/path/to/image.jpg',
    text_prompt='a red car'
)
```

### 4.2 Quick Segmentation
```python
# Minimal output version
quick_segment('/path/to/image.jpg', 'a blue bird')
```

### 4.3 Direct Pipeline Access
```python
# Full control over the pipeline
results = pipeline.segment_image(
    image=image_array,
    text_prompt='a golden brown dog',
    save_results=True,
    filename_prefix='my_result'
)
```

## 5. Output

The pipeline generates the following outputs:
- **Original image**: Input image (resized if necessary)
- **Binary mask**: Segmentation mask (grayscale PNG)
- **Visualization**: Combined figure showing:
  - Original image
  - Final segmentation mask
  - Overlay of mask on image
  - Top 3 mask candidates with scores

## 6. Installation & Setup

The notebook is designed for Google Colab and includes automatic installation of:
- PyTorch (CUDA 11.8)
- SAM 2 (from GitHub repository)
- OpenCLIP for CLIP models
- GroundingDINO (optional, with fallback)
- Standard CV libraries (OpenCV, scikit-image, Pillow)

## 7. Performance Characteristics

- **Optimal environment**: Google Colab with GPU (T4, A100, or V100)
- **Image size**: Automatically resizes images larger than 1024px
- **Processing time**: 10-30 seconds per image (depending on GPU)
- **Memory requirements**: ~8-12GB GPU memory for full ensemble

## 8. Limitations

### 8.1 Hardware Requirements
   - Requires CUDA-capable GPU for practical use
   - CPU inference is extremely slow (not recommended)
   - High GPU memory consumption due to ensemble CLIP models

### 8.2 Model Availability
   - GroundingDINO may fail to load due to installation issues
   - Falls back to CLIP-based detection (slightly lower accuracy)
   - Some CLIP models may fail to download (ensemble continues with available models)

### 8.3 Segmentation Constraints
   - Single object segmentation per execution
   - For multiple objects, requires separate runs
   - Cannot distinguish between multiple instances of the same object class

### 8.4 Text Prompt Sensitivity
   - Performance depends heavily on prompt quality and specificity
   - Ambiguous prompts (e.g., "object") produce poor results
   - Best results with descriptive prompts (e.g., "a golden retriever sitting on grass")

### 8.5 Scene Complexity
   - May struggle with cluttered scenes containing many objects
   - Complex backgrounds can confuse object detection
   - Occlusion and overlapping objects reduce accuracy

### 8.6 Post-Processing Artifacts
   - Morphological smoothing may over-smooth fine details
   - Small objects or thin structures may be lost
   - Fixed kernel sizes not adaptive to object scale

### 8.7 Evaluation Biases
   - Fixed weights in ensemble may not be optimal for all image types
   - Size scoring (5-50% of image) may penalize very small or large objects
   - CLIP similarity normalization tuned for specific use cases

### 8.8 Platform Dependencies
   - Hardcoded for Google Colab (Google Drive mounting)
   - Requires modification for local execution
   - No support for batch processing

### 8.9 No Ground Truth Validation
   - Quality scores are heuristic-based, not validated against ground truth
   - "High accuracy" claims not empirically verified
   - No quantitative metrics (IoU, precision, recall) without labeled data

### 8.10 Model Download Requirements
   - First run requires downloading several large model files (~5GB total)
   - Requires stable internet connection
   - No offline mode available

## 9. Future Improvements

Potential enhancements to address current limitations:
- Batch processing support for multiple images
- Multi-object segmentation with instance separation
- Adaptive post-processing based on object scale
- Local execution support (non-Colab environments)
- Quantitative evaluation with benchmark datasets
- Model caching for faster subsequent runs
- Progressive refinement for complex scenes

## 10. Technical Details

- **SAM 2 Model**: `sam2_hiera_base_plus` (Base+ variant)
- **Detection threshold**: 0.3 (GroundingDINO), 0.15 (CLIP)
- **Mask selection**: Weighted combination of 4 quality metrics
- **Post-processing**: 5x5 elliptical morphological operations + Gaussian smoothing

## 11. Citation (Project 2)

This implementation builds upon:
- **SAM 2**: Meta AI's Segment Anything Model 2
- **OpenCLIP**: Multiple CLIP model variants
- **GroundingDINO**: IDEA Research's text-to-detection model

## 12. License

This notebook is provided for research and educational purposes. Please refer to individual model licenses:
- SAM 2: Apache 2.0
- CLIP models: Various (check OpenCLIP documentation)
- GroundingDINO: Apache 2.0

---

**Note**: This is a research prototype. For production use, consider addressing the limitations listed above and validating performance on your specific use case.

---

# Project Files Structure

```
AIRL internship/
├── Q1_version2/                    # Project 1: Vision Transformer on CIFAR-10
│   ├── Copy of Q1.ipynb            # Main training notebook
│   ├── best_model.pth              # Best checkpoint (epoch 194)
│   ├── last_checkpoint.pth         # Final epoch checkpoint
│   ├── metrics.csv                 # Training/validation metrics
│   ├── confusion_matrix.csv        # Confusion matrix data
│   ├── classification_report*.txt  # Per-class metrics
│   ├── test_*.npy                  # Test predictions/logits/targets
│   └── requirements.txt            # Python dependencies
│
├── Q2_CLEAN.ipynb                  # Project 2: Text-driven segmentation (image)
├── Q2_CLEAN_imgandvideo.ipynb      # Project 2: Extended version (video support)
├── OP1.png                         # CIFAR-10 sample images
└── OP2.png                         # Additional visualizations
```

---

# Contact & Support

For questions, issues, or contributions, please refer to the individual project notebooks or contact the repository maintainer.

**Last Updated:** October 4, 2025

