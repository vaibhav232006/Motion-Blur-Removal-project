# Motion Blur Removal Using Frequency Domain Techniques

An image restoration project that demonstrates how to simulate and remove motion blur using **Fourier Transform** and **Wiener Deconvolution Filtering**. It provides both an interactive **Streamlit web application** and a standalone Python script with quantitative evaluation metrics (MSE and PSNR).

---

## 📌 Features

- **Interactive Web App (Streamlit)**: Upload custom images, view real-time motion blur simulation, inspect the Fourier frequency spectrum, and download the restored output.
- **Standalone Processing Script (`main.py`)**: Scriptable workflow for image blurring, spectrum computation, Wiener filtering, and saving comparison plots.
- **Image Restoration Pipeline**:
  - Point Spread Function (PSF) motion blur modeling
  - 2D Fast Fourier Transform (FFT) & frequency shifting
  - Wiener deconvolution with noise-to-signal ratio constant ($K$)
- **Quality Metrics**: Automatic calculation of Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR in dB).

---

## 📁 Project Structure

```text
Motion_Blur_Removal/
├── app.py                  # Streamlit interactive web interface
├── main.py                 # Standalone image processing script
├── requirements.txt        # Python package dependencies
├── .gitignore              # Git ignore rules
├── input_images/           # Sample input & generated blurred images
│   ├── input.jpg
│   └── motion_blurred.jpg
└── output_images/          # Output results and comparison plots
    ├── final_comparison.png
    └── wiener_restored.jpg
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Installation
Clone this repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run the Standalone Script
```bash
python main.py
```

---

## 📊 Methodology

1. **Blur Modeling**: A horizontal motion blur kernel is convolved with the input image.
2. **Frequency Domain Transformation**: 2D FFT transforms spatial domain data into the frequency spectrum.
3. **Wiener Filter Application**: 
   $$\hat{F}(u, v) = \left[ \frac{H^*(u, v)}{|H(u, v)|^2 + K} \right] \cdot G(u, v)$$
4. **Inverse FFT & Normalization**: The image is mapped back to the spatial domain and normalized to $[0, 255]$.
5. **Evaluation**: Restored image quality is measured against the ground truth using MSE and PSNR.
