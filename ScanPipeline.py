import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage.draw import disk
import pandas as pd


picF = '/Users/astr/Documents/Pigments/ScanQuantif/Scans/IMG_20260622_Vio_Red_D2Ext.tif'
im = np.array(Image.open(picF))
# MATLAB rot90(im, 3) rotates 270 degrees counter-clockwise. 
# numpy.rot90 rotates 90 degrees counter-clockwise by default, so we rotate 3 times.
#im = np.rot90(im, k=3)  # add for 4 plates scans

plt.figure()
plt.imshow(im)
plt.title("Loaded Image")
plt.axis('off')
plt.show()

# %% check corners 

PlateA = np.array([[1297,1835], [1285,3443], [3749,1830], [3729,3434]])
PlateB = np.array([[1296,3851], [1285,5459], [3735,3855], [3727,5454]]) #D2Ext

allPlates = [PlateA, PlateB] #PlateC, PlateD

plt.figure()
plt.imshow(im)
plt.scatter(PlateA[:, 0], PlateA[:, 1], s=50, c='red', marker='o')
plt.title("Check Corners")
plt.axis('off')
plt.show()

# %% extract all XY

def getAllCenters(Corners, nRows, nCols):
    # corners is a 4x2, with lines TL, BL, TR, BR
    C_TL = Corners[0, :]  # top left
    C_BL = Corners[1, :]  # bottom left
    C_TR = Corners[2, :]  # top right
    C_BR = Corners[3, :]  # bottom right

    # Interpolation horizontale (haut et bas)
    # MATLAB: linspace(start, end, count)
    # Python: np.linspace(start, end, count)
    x_top = np.linspace(C_TL[0], C_TR[0], nCols)
    y_top = np.linspace(C_TL[1], C_TR[1], nCols)

    x_bot = np.linspace(C_BL[0], C_BR[0], nCols)
    y_bot = np.linspace(C_BL[1], C_BR[1], nCols)

    # Interpolation verticale pour chaque colonne
    X = np.zeros((nRows, nCols))
    Y = np.zeros((nRows, nCols))

    for c in range(nCols):
        # MATLAB: X(:,c) = linspace(..., nRows)
        # Python: X[:, c] = np.linspace(..., nRows)
        X[:, c] = np.linspace(x_top[c], x_bot[c], nRows)
        Y[:, c] = np.linspace(y_top[c], y_bot[c], nRows)

    return X, Y

AllPlateColors = []
#AllPlateColors_Norm = []

# Pre-calculate grid dimensions
nRows, nCols = 16, 24
rad = 10

# Generate relative offsets for the circular mask ONCE (saves massive computation)
# Create a grid centered at 0,0
#y_off, x_off = np.ogrid[-rad:rad+1, -rad:rad+1]
#mask_template = (x_off**2 + y_off**2 <= rad**2)
#mask_y, mask_x = np.where(mask_template)

# %% enhance brightness
imLab = rgb2lab(im)
c = 2
imLab[:, :, 0] = (imLab[:, :, 0] - 57) * c + 57
im2 = lab2rgb(imLab)

plt.figure()
plt.imshow(im2)
plt.title("Enhanced Contrast Image")
plt.axis('off')
plt.show()

# Convert im2 back to 0-255 range for saving consistency (lab2rgb returns 0.0-1.0)
im2_255 = (im2 * 255).astype(np.uint8)

#%% Extract RGB values
print("Starting extraction...")

for pi in range(2): ## Adapt number of plates
    # Use 'plate' variable if you intended to use different corners per plate, 
    # otherwise keep PlateA as per original script.
    # X, Y = getAllCenters(plate, nCols, nRows) 
    plate = allPlates[pi]
    X, Y = getAllCenters(plate, nRows, nCols) 
    
    # Flatten coordinates to a list of points (N_points, 2)
    # X and Y are (16, 24). We need them as a list of (x, y) pairs.
    x_coords = X.flatten().astype(int)
    y_coords = Y.flatten().astype(int)
    
    # Adjust for 0-based indexing if your getAllCenters returns 1-based coords
    # If you previously had index errors, uncomment the next two lines:
    # x_coords = x_coords - 1
    # y_coords = y_coords - 1

    # Prepare array to hold results: (384 wells, 3 channels)
    well_colors = np.zeros((nRows * nCols, 3))
    
    # Vectorized Extraction Loop (Much faster than pixel-grid recreation)
    # We iterate over the 384 wells, but for each well we only look at ~700 pixels (mask)
    # instead of the whole image grid.
    # Vectorized Extraction Loop
    for i in range(len(x_coords)):
        cx, cy = x_coords[i], y_coords[i]
        
        # Define bounding box
        # Ensure we don't go out of global image bounds
        y_min = max(0, cy - rad)
        y_max = min(im2_255.shape[0], cy + rad + 1)
        x_min = max(0, cx - rad)
        x_max = min(im2_255.shape[1], cx + rad + 1)
        
        # Extract local region
        region = im2_255[y_min:y_max, x_min:x_max]
        
        # Calculate the center of the well RELATIVE to the extracted region
        # Original center: (cx, cy). Region top-left: (x_min, y_min)
        center_y_rel = cy - y_min
        center_x_rel = cx - x_min
        
        # Create a fresh mask specifically for this region's size
        # This avoids any off-by-one errors from global masks
        h_reg, w_reg, _ = region.shape
        y_reg, x_reg = np.ogrid[:h_reg, :w_reg]
        
        # Distance from the relative center
        dist_sq = (x_reg - center_x_rel)**2 + (y_reg - center_y_rel)**2
        mask = dist_sq <= rad**2
        
        # Apply mask
        if np.any(mask):
            valid_pixels = region[mask]
            # Reshape to (N_pixels, 3) to compute mean across channels correctly
            # region[mask] returns a flat list of pixels if region is 3D? 
            # No, region[mask] on a (H,W,3) array with a (H,W) mask returns (N, 3)
            well_colors[i] = np.mean(valid_pixels, axis=0)
        else:
            well_colors[i] = [0, 0, 0]

    # Reshape back to (16, 24, 3)
    colors = well_colors.reshape(nRows, nCols, 3)
    AllPlateColors.append(colors)
    print(f"Plate {pi+1} completed.")

print("Extraction finished.")

X, Y = getAllCenters(PlateA, nRows, nCols)
colors = AllPlateColors[0] 
# Flatten X and Y for scatter
X_flat = X.flatten()
Y_flat = Y.flatten()
# Reshape colors to (384, 3) for plotting
# MATLAB: colorsP=reshape(colors, 384, 3);
# Python: reshape to (-1, 3) flattens the first two dimensions
colorsP = colors.reshape(-1, 3)

# Normalize colors to 0-1 for matplotlib
colorsP = colorsP / 255.0


plt.figure()
plt.imshow(im2_255)
# Re-flatten X and Y just in case
plt.scatter(X_flat, Y_flat, s=50, c=colorsP, marker='o')
plt.scatter(PlateA[:, 0], PlateA[:, 1], s=50, c='red', marker='o')
plt.title("Enhanced Colors Overlay")
plt.axis('off')
plt.show()


# %% save allPlateColors
channels = ['R', 'G', 'B']
dirF = '/Users/astr/Documents/Pigments/ScanQuantif' ## create new folder !

# Ensure directory exists
os.makedirs(dirF, exist_ok=True)

for p in range(2): ## Adapt to number of plates in scan
    for c_idx in range(3):
        # 1. Save Original Values
        data_orig = AllPlateColors[p][:, :, c_idx]
        filename_orig = f'blank_Plate{p+1}_{channels[c_idx]}.csv' #plate number
        pd.DataFrame(data_orig).to_csv(os.path.join(dirF, filename_orig), index=False, header=False)
        
        print(f"Saved: {filename_orig}")


# Ensure you have your list of plates and their corresponding corner data
# We only take the first 2 since you have 2 plates per picture

# %% Plot Overlay for BOTH Plates

# Configuration
plate_names = ["Plate A", "Plate B"]

plt.figure(figsize=(15, 10))

# 1. Display the background image ONCE
plt.imshow(im2_255) 
plt.title("Overlay: Plate A & Plate B", fontsize=16)

# 2. Loop through both plates (0 and 1)
for i in range(2):
    # A. Get corners for the current plate
    corners = allPlates[i]
    
    # B. Recalculate grid (X, Y) specifically for these corners
    X, Y = getAllCenters(corners, nRows, nCols)
    
    # C. Get the color data for this plate from the list
    colors_plate = AllPlateColors[i] 
    
    # D. Flatten and Normalize
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    colorsP = colors_plate.reshape(-1, 3) / 255.0  # Normalize to 0-1
    
    # E. Plot the wells
    plt.scatter(X_flat, Y_flat, s=50, c=colorsP, marker='o')

# 3. Finalize
plt.legend(loc='upper right')
plt.axis('off')
plt.tight_layout()
plt.show()