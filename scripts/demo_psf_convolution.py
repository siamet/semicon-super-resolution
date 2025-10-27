"""
Demo script for PSF/OTF models and pattern degradation.

This script demonstrates:
1. Creating different PSF models (Airy, Hopkins, Chromatic, Aberration)
2. Computing OTF and MTF
3. Convolving patterns with PSF to simulate optical blurring
4. Visualizing PSF, OTF, and degraded patterns

Usage:
    python scripts/demo_psf_convolution.py

Author: Semiconductor Super-Resolution Research Project
Date: 2025-10-23
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic.psf_models import create_psf_model, OpticalConfig
from src.data.synthetic.pattern_generator import create_pattern_generator, PatternConfig


def plot_psf_analysis(psf_model, title="PSF Analysis"):
    """Plot PSF, OTF, MTF, and radial profiles."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Compute PSF and OTF
    psf = psf_model.compute_psf()
    otf = psf_model.compute_otf()
    mtf = psf_model.get_mtf()
    ptf = psf_model.get_ptf()

    # 1. PSF 2D
    im1 = axes[0, 0].imshow(psf, cmap='hot', interpolation='nearest')
    axes[0, 0].set_title('PSF (Point Spread Function)')
    axes[0, 0].set_xlabel('Pixels')
    axes[0, 0].set_ylabel('Pixels')
    plt.colorbar(im1, ax=axes[0, 0])

    # 2. PSF radial profile
    center = psf.shape[0] // 2
    radial_profile = psf[center, center:]
    pixel_nm = psf_model.config.pixel_size_nm
    r_nm = np.arange(len(radial_profile)) * pixel_nm

    axes[0, 1].plot(r_nm, radial_profile, 'b-', linewidth=2)
    axes[0, 1].axvline(psf_model.config.rayleigh_limit_nm, color='r', linestyle='--',
                       label=f'Rayleigh limit ({psf_model.config.rayleigh_limit_nm:.1f} nm)')
    axes[0, 1].set_title('PSF Radial Profile')
    axes[0, 1].set_xlabel('Radius (nm)')
    axes[0, 1].set_ylabel('Intensity (normalized)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. PSF 3D surface (optional)
    X, Y = np.meshgrid(range(psf.shape[1]), range(psf.shape[0]))
    axes[0, 2].contourf(X, Y, psf, levels=20, cmap='hot')
    axes[0, 2].set_title('PSF Contour Plot')
    axes[0, 2].set_xlabel('Pixels')
    axes[0, 2].set_ylabel('Pixels')

    # 4. MTF 2D
    im4 = axes[1, 0].imshow(mtf, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    axes[1, 0].set_title('MTF (Modulation Transfer Function)')
    axes[1, 0].set_xlabel('Frequency (pixels)')
    axes[1, 0].set_ylabel('Frequency (pixels)')
    plt.colorbar(im4, ax=axes[1, 0], label='MTF (0-1)')

    # 5. MTF radial profile
    mtf_radial = mtf[center, center:]
    fx, fy = psf_model.get_frequency_grid()
    f_radial = np.sqrt(fx[center, center:]**2 + fy[center, center:]**2)

    axes[1, 1].plot(f_radial, mtf_radial, 'g-', linewidth=2)
    axes[1, 1].axvline(psf_model.config.cutoff_frequency_nm, color='r', linestyle='--',
                       label=f'Coherent cutoff ({psf_model.config.cutoff_frequency_nm:.5f} nm⁻¹)')
    axes[1, 1].axvline(psf_model.config.incoherent_cutoff_frequency_nm, color='orange', linestyle='--',
                       label=f'Incoherent cutoff ({psf_model.config.incoherent_cutoff_frequency_nm:.5f} nm⁻¹)')
    axes[1, 1].set_title('MTF Radial Profile')
    axes[1, 1].set_xlabel('Spatial Frequency (nm⁻¹)')
    axes[1, 1].set_ylabel('MTF')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.1])

    # 6. PTF (Phase Transfer Function)
    im6 = axes[1, 2].imshow(ptf, cmap='twilight', interpolation='nearest', vmin=-np.pi, vmax=np.pi)
    axes[1, 2].set_title('PTF (Phase Transfer Function)')
    axes[1, 2].set_xlabel('Frequency (pixels)')
    axes[1, 2].set_ylabel('Frequency (pixels)')
    plt.colorbar(im6, ax=axes[1, 2], label='Phase (radians)')

    plt.tight_layout()
    return fig


def plot_pattern_degradation(pattern, psf_model, title="Pattern Degradation"):
    """Plot original pattern and PSF-degraded version."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Degrade pattern
    degraded = psf_model.convolve(pattern)

    # 1. Original pattern
    axes[0, 0].imshow(pattern, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original Pattern (Sharp)')
    axes[0, 0].axis('off')

    # 2. Degraded pattern
    axes[0, 1].imshow(degraded, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Degraded Pattern (PSF Blurred)')
    axes[0, 1].axis('off')

    # 3. Difference
    diff = np.abs(pattern - degraded)
    im3 = axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('Absolute Difference')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])

    # 4. Horizontal profile (original)
    center = pattern.shape[0] // 2
    pixel_nm = psf_model.config.pixel_size_nm
    x_nm = np.arange(pattern.shape[1]) * pixel_nm

    axes[1, 0].plot(x_nm, pattern[center, :], 'b-', linewidth=2, label='Original')
    axes[1, 0].plot(x_nm, degraded[center, :], 'r--', linewidth=2, label='Degraded')
    axes[1, 0].set_title('Horizontal Profile (Center)')
    axes[1, 0].set_xlabel('Position (nm)')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. FFT comparison
    fft_original = np.abs(np.fft.fftshift(np.fft.fft2(pattern)))
    fft_degraded = np.abs(np.fft.fftshift(np.fft.fft2(degraded)))

    # Log scale for better visualization
    axes[1, 1].imshow(np.log1p(fft_original), cmap='viridis')
    axes[1, 1].set_title('FFT Original (log scale)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(np.log1p(fft_degraded), cmap='viridis')
    axes[1, 2].set_title('FFT Degraded (log scale)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    return fig


def main():
    """Run PSF/OTF demonstration."""
    print("=" * 80)
    print("PSF/OTF Models and Pattern Degradation Demo")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path("results/psf_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Demo 1: Airy Disk PSF (Coherent Illumination)
    # -------------------------------------------------------------------------
    print("Demo 1: Airy Disk PSF (Coherent, DUV λ=248nm, NA=0.95)")
    print("-" * 80)

    psf_airy = create_psf_model(
        wavelength_nm=248,
        NA=0.95,
        pixel_size_nm=2.0,
        kernel_size=33,
        sigma=0.01,  # Nearly coherent
        model_type='airy'
    )

    print(f"  Rayleigh limit: {psf_airy.config.rayleigh_limit_nm:.2f} nm")
    print(f"  Depth of focus: {psf_airy.config.depth_of_focus_nm:.2f} nm")
    print(f"  Coherent cutoff: {psf_airy.config.cutoff_frequency_nm:.6f} nm⁻¹")
    print()

    fig1 = plot_psf_analysis(psf_airy, "Airy Disk PSF (DUV, λ=248nm, NA=0.95)")
    fig1.savefig(output_dir / "01_airy_psf_analysis.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # Demo 2: Hopkins Partial Coherence PSF
    # -------------------------------------------------------------------------
    print("Demo 2: Hopkins Partial Coherence PSF (σ=0.5)")
    print("-" * 80)

    psf_hopkins = create_psf_model(
        wavelength_nm=248,
        NA=0.95,
        pixel_size_nm=2.0,
        kernel_size=33,
        sigma=0.5,  # Partial coherence
        model_type='hopkins'
    )

    print(f"  Partial coherence σ: {psf_hopkins.config.sigma}")
    print(f"  Incoherent cutoff: {psf_hopkins.config.incoherent_cutoff_frequency_nm:.6f} nm⁻¹")
    print()

    fig2 = plot_psf_analysis(psf_hopkins, "Hopkins Partial Coherence PSF (σ=0.5)")
    fig2.savefig(output_dir / "02_hopkins_psf_analysis.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # -------------------------------------------------------------------------
    # Demo 3: Chromatic PSF (Polychromatic Illumination)
    # -------------------------------------------------------------------------
    print("Demo 3: Chromatic PSF (Mercury lamp: 365nm + 436nm + 550nm)")
    print("-" * 80)

    psf_chromatic = create_psf_model(
        wavelength_nm=436,  # Central wavelength
        NA=0.95,
        pixel_size_nm=2.0,
        kernel_size=33,
        sigma=0.5,
        model_type='chromatic',
        wavelengths_chromatic=[365, 436, 550],  # Mercury i, g, h lines
    )

    print(f"  Wavelengths: {[365, 436, 550]} nm")
    print(f"  Weights: uniform (1/3 each)")
    print()

    fig3 = plot_psf_analysis(psf_chromatic, "Chromatic PSF (Mercury Lamp)")
    fig3.savefig(output_dir / "03_chromatic_psf_analysis.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)

    # -------------------------------------------------------------------------
    # Demo 4: Aberration PSF
    # -------------------------------------------------------------------------
    print("Demo 4: Aberration PSF (Spherical + Coma)")
    print("-" * 80)

    psf_aberration = create_psf_model(
        wavelength_nm=248,
        NA=0.95,
        pixel_size_nm=2.0,
        kernel_size=33,
        sigma=0.5,
        model_type='aberration',
        aberrations={'spherical': 0.1, 'coma': 0.05}
    )

    print(f"  Aberrations: spherical=0.1λ, coma=0.05λ")
    print()

    fig4 = plot_psf_analysis(psf_aberration, "Aberration PSF (Spherical + Coma)")
    fig4.savefig(output_dir / "04_aberration_psf_analysis.png", dpi=150, bbox_inches='tight')
    plt.close(fig4)

    # -------------------------------------------------------------------------
    # Demo 5: Pattern Degradation - Gratings
    # -------------------------------------------------------------------------
    print("Demo 5: Pattern Degradation - Line/Space Gratings")
    print("-" * 80)

    # Generate grating pattern
    config = PatternConfig(image_size=512, pixel_size_nm=2.0)
    grating_gen = create_pattern_generator('grating', config=config)
    grating = grating_gen.generate(
        pitch_nm=100.0,
        duty_cycle=0.5,
        orientation_deg=0.0,
        add_ler=True,
        ler_sigma_nm=2.0,
        ler_correlation_nm=20.0
    )[0]  # Get pattern only, not metadata

    print(f"  Pattern: 100nm pitch gratings")
    print(f"  LER: σ=2nm, correlation=20nm")
    print()

    fig5 = plot_pattern_degradation(grating, psf_hopkins, "Grating Degradation (Hopkins PSF)")
    fig5.savefig(output_dir / "05_grating_degradation.png", dpi=150, bbox_inches='tight')
    plt.close(fig5)

    # -------------------------------------------------------------------------
    # Demo 6: Pattern Degradation - Contact Holes
    # -------------------------------------------------------------------------
    print("Demo 6: Pattern Degradation - Contact Holes")
    print("-" * 80)

    # Generate contact hole pattern
    contact_gen = create_pattern_generator('contacts', config=config)
    contacts = contact_gen.generate(
        diameter_nm=60.0,
        pitch_nm=120.0,
        shape='circular',
        array_type='regular'
    )[0]  # Get pattern only, not metadata

    print(f"  Pattern: 60nm diameter contact holes")
    print(f"  Pitch: 120nm (regular array)")
    print()

    fig6 = plot_pattern_degradation(contacts, psf_hopkins, "Contact Holes Degradation (Hopkins PSF)")
    fig6.savefig(output_dir / "06_contacts_degradation.png", dpi=150, bbox_inches='tight')
    plt.close(fig6)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print(f"\nAll figures saved to: {output_dir.absolute()}")
    print("\nFiles generated:")
    print("  01_airy_psf_analysis.png       - Airy disk PSF analysis")
    print("  02_hopkins_psf_analysis.png    - Hopkins partial coherence PSF")
    print("  03_chromatic_psf_analysis.png  - Chromatic aberration PSF")
    print("  04_aberration_psf_analysis.png - Lens aberration PSF")
    print("  05_grating_degradation.png     - Grating pattern degradation")
    print("  06_contacts_degradation.png    - Contact holes degradation")
    print("\nPSF/OTF models successfully demonstrated! ✓")


if __name__ == "__main__":
    main()
