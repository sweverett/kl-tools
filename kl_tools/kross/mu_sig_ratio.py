import numpy as np
import matplotlib.pyplot as plt

# Constants and parameters
vtf = 200  # Tully-Fisher velocity (km/s)
sigma_tf = 0.05  # Uncertainty in TF relation (dex)
ln10 = np.log(10)
sigma_vcirc_updated = ln10 * vtf * sigma_tf  # Corrected expression for sigma_vcirc

sigma_vmajors = [5, 10, 20, 50, 100]  # Uncertainty in major axis velocity (km/s)
v_major = 100  # Fixed major axis velocity (km/s)
i_values = np.linspace(0.01, np.pi/2, 500)  # Inclination angle from small to 90 degrees

# Function to calculate mu(i) and sigma(i) with updated sigma_vcirc
def mu_sigma_updated(i, v_major, vtf, sigma_vmajor, sigma_vcirc):
    mu_i = (sigma_vcirc**2 * v_major * np.sin(i) + sigma_vmajor**2 * vtf) / \
           (sigma_vmajor**2 + sigma_vcirc**2 * np.sin(i)**2)
    sigma_i_squared = (sigma_vmajor**2 * sigma_vcirc**2) / \
                      (sigma_vmajor**2 + sigma_vcirc**2 * np.sin(i)**2)
    sigma_i = np.sqrt(sigma_i_squared)
    return mu_i, sigma_i, mu_i / sigma_i

# Calculate mu(i), sigma(i), and their ratio for each sigma_vmajor with the updated sigma_vcirc
results_updated = {}
for sigma_vmajor in sigma_vmajors:
    mu_values, sigma_values, ratio_values = mu_sigma_updated(i_values, v_major, vtf, sigma_vmajor, sigma_vcirc_updated)
    results_updated[sigma_vmajor] = (mu_values, sigma_values, ratio_values)

# Plotting updated results
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

for sigma_vmajor, (mu_values, sigma_values, ratio_values) in results_updated.items():
    label = fr'$\sigma_{{v_\text{{maj}}}} = {sigma_vmajor} \, \text{{km/s}}$'
    axes[0].plot(i_values, mu_values, label=label)
    axes[1].plot(i_values, sigma_values, label=label)
    axes[2].plot(i_values, ratio_values, label=label)

# Titles and labels with LaTeX formatting
axes[0].set_title(rf'$\mu(i)$ vs. Inclination Angle $\left( v_{{\text{{TF}}}} = {vtf} \, \text{{km/s}}, \, \sigma_{{\text{{TF}}}} = {sigma_tf}, \, \sigma_{{v_\text{{circ}}}} = {sigma_vcirc_updated:.2f} \, \text{{km/s}} \right)$')
axes[0].set_xlabel(r'Inclination Angle $i$')
axes[0].set_ylabel(r'$\mu(i) \, \text{(km/s)}$')

axes[1].set_title(rf'$\sigma(i)$ vs. Inclination Angle $\left( v_{{\text{{TF}}}} = {vtf} \, \text{{km/s}}, \, \sigma_{{\text{{TF}}}} = {sigma_tf}, \, \sigma_{{v_\text{{circ}}}} = {sigma_vcirc_updated:.2f} \, \text{{km/s}} \right)$')
axes[1].set_xlabel(r'Inclination Angle $i$')
axes[1].set_ylabel(r'$\sigma(i) \, \text{(km/s)}$')

axes[2].set_title(rf'$\frac{{\mu(i)}}{{\sigma(i)}}$ vs. Inclination Angle $\left( v_{{\text{{TF}}}} = {vtf} \, \text{{km/s}}, \, \sigma_{{\text{{TF}}}} = {sigma_tf}, \, \sigma_{{v_\text{{circ}}}} = {sigma_vcirc_updated:.2f} \, \text{{km/s}} \right)$')
axes[2].set_xlabel(r'Inclination Angle $i$')
axes[2].set_ylabel(r'$\frac{{\mu(i)}}{{\sigma(i)}}$')

for ax in axes:
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()