# In-Depth Model Architecture: Siam-ViT for Optical PUF Verification

## 1. Executive Summary
This document proposes a neural network architecture designed to authenticate physical objects using random particle distributions (Physical Unclonable Functions) on adhesive tape. The architecture, **Siam-ViT (Siamese Vision Transformer with Spatial Alignment)**, is optimized for high-entropy feature extraction and robustness against real-world scanning artifacts such as perspective warp and specular glare.

---

## 2. System Overview
The authentication pipeline follows a "Metric Learning" paradigm. Rather than classifying an image, the model transforms a high-resolution scan into a low-dimensional **feature vector (embedding)**.

1.  **Registration:** A "Master Scan" is converted to a vector and stored.
2.  **Verification:** A "Query Scan" is converted to a vector.
3.  **Comparison:** The system calculates the distance between the two vectors. If the distance is below a threshold ($	au$), the tape is authenticated.

---

## 3. Detailed Architecture Layers

### Layer 1: Spatial Transformer Network (STN)
To handle the rotation and perspective issues seen in handheld phone scans, an STN is placed at the front of the pipeline.
- **Localization Network:** A lightweight CNN that regresses the parameters of an Affine or Homography transform.
- **Grid Generator:** Creates a sampling grid based on the predicted transform.
- **Sampler:** Warps the input image into a canonical "flat" orientation.
- *Purpose:* Ensures the subsequent layers always see a normalized strip of tape.

### Layer 2: Feature Extractor (Backbone)
**Architecture: Vision Transformer (ViT-Tiny/16)**
- **Patch Embedding:** The $224 	imes 224$ normalized image is broken into $14 	imes 14$ patches.
- **Self-Attention:** Unlike CNNs, which have a limited receptive field, ViT uses Multi-Head Self-Attention to correlate the position of a bead on the top of the tape with a bead on the bottom.
- **Positional Encoding:** Crucial for PUFs, as it preserves the exact spatial coordinates of the bead "constellation."

### Layer 3: Siamese Projector
A small Multi-Layer Perceptron (MLP) that maps the ViT's latent representation into a 128-dimensional embedding space.
- **Activation:** ReLU
- **Normalization:** L2-Normalization (ensures all embeddings lie on a hypersphere).

---

## 4. Training Strategy

### Loss Function: Triplet Loss
The model is trained using triplets: an **Anchor** (Master), a **Positive** (same tape, different scan), and a **Negative** (different tape or tampered tape).
$$\mathcal{L} = \max(0, \|f(x_a) - f(x_p)\|^2 - \|f(x_a) - f(x_n)\|^2 + lpha)$$
Where $lpha$ is a margin that forces the model to push negative samples significantly further away than positives.

### Data Augmentation Policy
- **Synthetic-to-Real (S2R):** The model is pre-trained on the Python-generated dataset to learn geometric consistency.
- **Noise Injection:** Gaussian and Poisson noise are added to mimic IR/UV sensor grain.
- **Occlusion:** Random white masks are applied to simulate the specular flash glare seen in physical prototypes.

---

## 5. Tamper Detection Logic
Tamper detection is treated as a **local distance anomaly**.
1. The model divides the tape into a $4 	imes 4$ grid.
2. It computes a distance for each grid cell.
3. **Decision Rule:** If any single cell exceeds the distance threshold by $>3\sigma$, but the rest of the cells are consistent, the tape is flagged as **Partially Tampered** (indicating a local peel or cut).

---

## 6. Hardware & Deployment Considerations
- **Format:** Export via ONNX for cross-platform compatibility.
- **Quantization:** INT8 quantization to allow the model to run in <50ms on mobile NPU (Neural Processing Unit).
- **Inference:** The model should reside on-device to ensure privacy and prevent "Man-in-the-Middle" attacks on the PUF data.
