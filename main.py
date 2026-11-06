
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read original image
image = cv2.imread("input_images/input.jpg")

if image is None:
    print("Image not found! Check the filename.")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Create motion blur kernel
kernel_size = 15
kernel = np.zeros((kernel_size, kernel_size))
kernel[kernel_size // 2, :] = 1
kernel = kernel / kernel_size

# 3. Apply motion blur
motion_blurred = cv2.filter2D(image, -1, kernel)

# 4. Save blurred image
cv2.imwrite(
    "input_images/motion_blurred.jpg",
    cv2.cvtColor(motion_blurred, cv2.COLOR_RGB2BGR)
)

# 5. Display both images together
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(motion_blurred)
plt.title("Motion Blurred Image")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Motion blur created successfully!")


# STEP 14: Apply Fourier Transform

# Convert blurred image to grayscale
gray_blurred = cv2.cvtColor(
    motion_blurred, cv2.COLOR_RGB2GRAY
)

# Apply 2D Fourier Transform
f = np.fft.fft2(gray_blurred)

# Shift zero frequency to the center
f_shift = np.fft.fftshift(f)

# Calculate magnitude spectrum
magnitude_spectrum = 20 * np.log(
    np.abs(f_shift) + 1
)

# Display frequency spectrum
plt.figure(figsize=(8, 5))
plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("Fourier Transform of Motion Blurred Image")
plt.axis("off")
plt.show()

print("Fourier Transform applied successfully!")


# STEP 15: Wiener Filtering

# Convert image to grayscale
blurred_gray = cv2.cvtColor(
    motion_blurred, cv2.COLOR_RGB2GRAY
)

# Create motion blur kernel
kernel_size = 15
psf = np.zeros((kernel_size, kernel_size))
psf[kernel_size // 2, :] = 1
psf = psf / kernel_size

# Resize kernel to image dimensions
H = np.zeros(blurred_gray.shape)
H[:kernel_size, :kernel_size] = psf

# Fourier Transform of blur kernel
H_fft = np.fft.fft2(H)

# Fourier Transform of blurred image
G = np.fft.fft2(blurred_gray)

# Wiener filter
K = 0.01
H_conj = np.conj(H_fft)

F_estimated = (
    H_conj / (np.abs(H_fft)**2 + K)
) * G

# Inverse Fourier Transform
restored = np.fft.ifft2(F_estimated)
restored = np.abs(restored)

# Normalize image
restored = cv2.normalize(
    restored, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# Save restored image
cv2.imwrite(
    "output_images/wiener_restored.jpg",
    restored
)

# Display result
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(blurred_gray, cmap="gray")
plt.title("Motion Blurred Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(restored, cmap="gray")
plt.title("Wiener Restored Image")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Wiener filtering completed!")


# STEP 16: Final comparison

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(motion_blurred)
plt.title("Motion Blurred Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(restored, cmap="gray")
plt.title("Wiener Restored Image")
plt.axis("off")

plt.tight_layout()
plt.savefig("output_images/final_comparison.png")
plt.show()

print("Final comparison saved successfully!")

# STEP 17: Image quality evaluation

# Convert original image to grayscale
original_gray = cv2.cvtColor(
    image, cv2.COLOR_RGB2GRAY
)

# Calculate MSE
mse = np.mean(
    (original_gray.astype(float) - restored.astype(float)) ** 2
)

# Calculate PSNR
psnr = cv2.PSNR(original_gray, restored)

print("\n--- Image Quality Evaluation ---")
print("MSE:", mse)
print("PSNR:", psnr, "dB")
