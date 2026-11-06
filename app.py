
import streamlit as st
import cv2
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Motion Blur Removal",
    page_icon="🖼️",
    layout="wide"
)

# Sidebar
st.sidebar.title("Project Details")
st.sidebar.write("Digital Image Processing")
st.sidebar.write("Motion Blur Removal Using Frequency Domain Techniques")
st.sidebar.markdown("---")
st.sidebar.info("Upload an image to remove simulated motion blur.")

# Main title
st.title("Motion Blur Removal Using Frequency Domain Techniques")
st.write("Restore a motion-blurred image using Fourier Transform and Wiener filtering.")

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Unable to read this image.")
        st.stop()

    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create motion blur
    kernel_size = 15
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1 / kernel_size

    blurred = cv2.filter2D(original, -1, kernel)

    # Convert blurred image to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32)

    # Fourier Transform
    F = np.fft.fft2(gray)
    F_shift = np.fft.fftshift(F)
    spectrum = 20 * np.log(np.abs(F_shift) + 1)

    # Create frequency-domain blur kernel
    H = np.zeros(gray.shape, dtype=np.float32)
    H[kernel_size // 2, :] = 1 / kernel_size

    H = np.roll(H, -(kernel_size // 2), axis=0)
    H = np.roll(H, -(kernel_size // 2), axis=1)

    H_fft = np.fft.fft2(H)

    # Wiener filtering
    K = 0.01

    restored = np.fft.ifft2(
        (np.conj(H_fft) / (np.abs(H_fft) ** 2 + K)) * F
    )

    restored = np.abs(restored)
    restored = cv2.normalize(
        restored, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Image quality evaluation
    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

    mse = np.mean(
        (original_gray.astype(float) - restored.astype(float)) ** 2
    )

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10((255 ** 2) / mse)

    # Display images
    st.header("Image Processing Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(original, use_container_width=True)

    with col2:
        st.subheader("Motion Blurred Image")
        st.image(blurred, use_container_width=True)

    st.subheader("Fourier Transform - Frequency Spectrum")
    st.image(spectrum, clamp=True, use_container_width=True)

    st.subheader("Restored Image")
    st.image(restored, clamp=True, use_container_width=True)

    # Quality metrics
    st.header("Image Quality Evaluation")

    col1, col2 = st.columns(2)
    col1.metric("MSE", f"{mse:.2f}")
    col2.metric("PSNR", f"{psnr:.2f} dB")

    # Download button
    success, encoded_image = cv2.imencode(".png", restored)

    if success:
        st.download_button(
            label="Download Restored Image",
            data=encoded_image.tobytes(),
            file_name="restored_image.png",
            mime="image/png"
        )

else:
    st.info("Please upload an image to begin processing.")
