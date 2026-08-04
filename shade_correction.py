import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2lab, lab2rgb

# --- CONFIGURATION ---
# 1. Path to your EMPTY plate scan (Media + Dry)
reference_file = '/Users/astr/Documents/Pigments/ScanQuantif/Scans/IMG_20260619_T0_empty.tif'

# 2. Path to the EXPERIMENTAL plate you want to correct
experimental_file = '/Users/astr/Documents/Pigments/ScanQuantif/Scans/IMG_20260622_Vio_Red_D2Ext.tif' # Change this

# 3. TUNABLE PARAMETER: Smoothing Sigma
# This controls how "blurry" the shading map is.
# - Too low: It sees the well edges and tries to correct them (bad).
# - Too high: It misses the gradient of the shadow.
# Start with 50 or 100. Adjust based on image size.
SMOOTH_SIGMA = 80 

# --- STEP 1: LOAD REFERENCE (EMPTY PLATE) ---
print("Loading Reference...")
ref_img = np.array(Image.open(reference_file)).astype(float)

# If RGB, convert to Grayscale for the shading map (light is intensity-based)
if ref_img.ndim == 3:
    # Simple luminance: 0.299*R + 0.587*G + 0.114*B
    ref_gray = np.dot(ref_img[...,:3], [0.299, 0.587, 0.114])
else:
    ref_gray = ref_img

# --- STEP 2: CREATE THE SHADING MAP ---
print(f"Generating Shading Map (Sigma={SMOOTH_SIGMA})...")
# Gaussian blur removes the high-frequency details (wells) and keeps the low-frequency shadow
shading_map = gaussian_filter(ref_gray, sigma=SMOOTH_SIGMA)

# Normalize the map so its average is 1.0
# This ensures we don't brighten/darken the whole image, just correct relative differences
shading_map = shading_map / np.mean(shading_map)

# --- STEP 3: APPLY CORRECTION TO EXPERIMENTAL IMAGE ---
print("Correcting Experimental Image...")
exp_img = np.array(Image.open(experimental_file)).astype(float)

# We apply the correction channel-by-channel to preserve color
corrected_img = np.zeros_like(exp_img)

for c in range(3): # R, G, B
    channel = exp_img[:, :, c]
    # Formula: Corrected = Raw / Shading_Map
    # We add a tiny epsilon (1e-5) to avoid division by zero if shadow is black
    corrected_img[:, :, c] = (channel / (shading_map + 1e-5)) 

# Clip values to valid range 0-255
corrected_img = np.clip(corrected_img, 0, 255).astype(np.uint8)

# --- STEP 4: VISUALIZE RESULTS ---
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Original Reference
axes[0, 0].imshow(ref_img.astype(int))
axes[0, 0].set_title("1. Reference (Empty Plate)")
axes[0, 0].axis('off')

# 2. The Calculated Shading Map
axes[0, 1].imshow(shading_map, cmap='gray')
axes[0, 1].set_title(f"2. Shading Map (Smoothed)\nSigma={SMOOTH_SIGMA}")
axes[0, 1].axis('off')

# 3. Original Experimental
axes[1, 0].imshow(exp_img.astype(int))
axes[1, 0].set_title("3. Raw Experimental (With Shade)")
axes[1, 0].axis('off')

# 4. Corrected Experimental
axes[1, 1].imshow(corrected_img)
axes[1, 1].set_title("4. Corrected Experimental (Flat-Field)")
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# --- STEP 5: SAVE CORRECTED IMAGE ---
output_path = experimental_file.replace('.tif', '_CORRECTED.tif')
Image.fromarray(corrected_img).save(output_path)
print(f"Saved corrected image to: {output_path}")

# --- DIAGNOSTIC: Check the corners vs center ---
h, w = shading_map.shape
center_val = shading_map[h//2, w//2]
corner_val = shading_map[50, 50] # Approximate corner
print(f"\nDiagnostic:")
print(f"Center Intensity (normalized): {center_val:.2f}")
print(f"Corner Intensity (normalized): {corner_val:.2f}")
print(f"Correction Factor needed at corners: {1/corner_val:.2f}x")