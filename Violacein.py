import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colour
from scipy.interpolate import interp1d



##### Extract absorbance data from Fiji 

# Load data (assuming Excel file; adjust if CSV)
viola = pd.read_csv("Violacein_curve.csv")

# Convert to numpy array for easier indexing (optional but matches MATLAB style)
a = viola.values
#print(a)

# Extract Axes and XY
Axes = a[0:3, 5:7]   # MATLAB 2:5,6:7 → Python 1:5,5:7 (0-indexed)
XY = a[3:, 5:7]     # MATLAB 6:end,6:7 → Python 5:,5:7

# Extract values
Y0 = Axes[1, 1]     # MATLAB Axes(1,2)
Y1p2 = Axes[0, 1]   # MATLAB Axes(2,2)
X300 = Axes[1, 0]   # MATLAB Axes(3,1)
X800 = Axes[2, 0]   # MATLAB Axes(4,1)

# Compute normalized X and Y
Xn = 300 + 500 * (XY[:, 0] - X300) / (X800 - X300)
Yn = 1.8 - 1.8 * (XY[:, 1] - Y0) / (Y1p2 - Y0)

# Plot
plt.figure()
plt.plot(Xn, Yn)
plt.gca().tick_params(direction='out', labelsize=20)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.show()


wl_original = Xn
absorbance = Yn
transmission_original = 10**(-absorbance)

wl_target = np.arange(380, 781, 10)
# Create interpolation function
f = interp1d(wl_original, transmission_original, kind='linear', fill_value="extrapolate")

# Interpolate to target wavelengths
transmission_target = f(wl_target)

# Plot original data (as points) & interpolated data (as line)
plt.plot(wl_original, transmission_original, 'o', label='Original data (25 nm)', color='red', markersize=8)
plt.plot(wl_target, transmission_target, '-', label='Interpolated (10 nm)', color='blue', linewidth=2)
plt.show()

# --- CONVERT TRANSMITTANCE TO COLOR ---

# Créer l'objet colour du spectre
sd = colour.SpectralDistribution(transmission_target, wl_target)
# Convertir en XYZ (utilise CIE 1931 2°)
xyz = colour.sd_to_XYZ(sd, illuminant=colour.SDS_ILLUMINANTS['D65'])

# COnvert to LAB system
lab = colour.XYZ_to_Lab(xyz, illuminant=colour.SDS_ILLUMINANTS['D65'])

print("XYZ:", xyz)
print("LAB:", lab)