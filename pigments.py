import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colour
from scipy.interpolate import interp1d

# --- IMPORT DATA & INTERPOLATE ---
### For 1 sample - A1
df = pd.read_excel("251104_R_SpectralScanning.xlsx", sheet_name=0)

wl_original = df['Wavelength']
absorbance = df['A2']
transmission_original = 10**(-absorbance)

#plt.plot(wl_original, absorbance, 'o', label='Absorbance (25 nm)', color='red', markersize=8)
#plt.plot(wl_original, transmission_original, 'o', label='transmision (25 nm)', color='blue', markersize=8)
#plt.show()

# Target: 380 to 780 nm, every 10 nm
wl_target = np.arange(380, 781, 10)  # 380, 390, 400, ..., 780


# Create interpolation function
f = interp1d(wl_original, transmission_original, kind='linear', fill_value="extrapolate")

# Interpolate to target wavelengths
transmission_target = f(wl_target)



# --- PLOT ORIGINAL vs INTERPOLATED ---

#plt.figure(figsize=(10, 6))

# Plot original data (as points)
#plt.plot(wl_original, transmission_original, 'o', label='Original data (25 nm)', color='red', markersize=8)

# Plot interpolated data (as line)
#plt.plot(wl_target, transmission_target, '-', label='Interpolated (10 nm)', color='blue', linewidth=2)

# Optional: Add grid, labels, title
#plt.xlabel('Wavelength (nm)')
#plt.ylabel('Transmission')
#plt.title('Original vs Interpolated Spectral Transmission')
#plt.legend()
#plt.grid(True, alpha=0.3)

# Optional: Set y-limits to [0, 1] for transmission
#plt.ylim(0, 1)

# Show plot
#plt.show()

# --- CONVERT TRANSMITTANCE TO COLOR ---

# Créer l'objet colour du spectre
sd = colour.SpectralDistribution(transmission_target, wl_target)
# Convertir en XYZ (utilise CIE 1931 2°)
xyz = colour.sd_to_XYZ(sd, illuminant=colour.SDS_ILLUMINANTS['D65'])

# COnvert to LAB system
lab = colour.XYZ_to_Lab(xyz, illuminant=colour.SDS_ILLUMINANTS['D65'])

# Convertir en RGB (sRGB)
rgb = colour.XYZ_to_sRGB(xyz / 100)  # Normalisé

# Convertir en hex
rgb_hex = '#{:02X}{:02X}{:02X}'.format(
    int(np.clip(rgb[0], 0, 1) * 255),
    int(np.clip(rgb[1], 0, 1) * 255),
    int(np.clip(rgb[2], 0, 1) * 255)
)

print("A2")
print("XYZ:", xyz)
print("LAB:", lab)
print("RGB (sRGB):", rgb)
print("Hex:", rgb_hex)



