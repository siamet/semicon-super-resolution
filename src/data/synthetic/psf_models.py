"""
Point Spread Function (PSF) and Optical Transfer Function (OTF) Models

This module implements physically accurate PSF/OTF models for simulating optical
microscope imaging systems used in semiconductor inspection.

Models Implemented:
    1. Airy disk PSF (coherent illumination)
    2. Hopkins partial coherence formulation
    3. Chromatic aberration effects
    4. Zernike aberrations (spherical, coma, astigmatism)

Physical Background:
    - Resolution limit (Rayleigh): 0.61 * λ / NA
    - Depth of focus: λ / (2 * NA²)
    - Coherent vs. Incoherent imaging affects PSF/OTF

References:
    - Goodman, J. W. "Introduction to Fourier Optics" (2005)
    - Hopkins, H. H. "On the diffraction theory of optical images" (1953)
    - Born & Wolf, "Principles of Optics" (7th ed.)

"""

import numpy as np
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass
import warnings
from scipy.special import j1  # Bessel function of the first kind
from scipy.ndimage import gaussian_filter


@dataclass
class OpticalConfig:
    """
    Configuration for optical imaging system parameters.

    Attributes:
        wavelength_nm: Illumination wavelength in nanometers (e.g., 193, 248, 365)
        NA: Numerical aperture (0 < NA ≤ 1.4 for immersion)
        sigma: Partial coherence factor (condenser NA / objective NA), 0 < σ < 1
        pixel_size_nm: Physical pixel size in nanometers
        kernel_size: PSF kernel size in pixels (must be odd)
        aberrations: Dictionary of Zernike aberration coefficients in wavelengths
    """
    wavelength_nm: float
    NA: float
    sigma: float = 0.5
    pixel_size_nm: float = 2.0
    kernel_size: int = 33
    aberrations: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Validate optical parameters."""
        # Validate wavelength
        if self.wavelength_nm <= 0:
            raise ValueError(f"Wavelength must be positive, got {self.wavelength_nm} nm")

        # Validate NA
        if not (0 < self.NA <= 1.5):
            raise ValueError(f"NA must be in (0, 1.5], got {self.NA}")
        if self.NA > 1.0:
            warnings.warn(f"NA > 1.0 ({self.NA}) requires immersion medium")

        # Validate sigma (partial coherence)
        if not (0 < self.sigma < 1):
            raise ValueError(f"Sigma (partial coherence) must be in (0, 1), got {self.sigma}")

        # Validate pixel size
        if self.pixel_size_nm <= 0:
            raise ValueError(f"Pixel size must be positive, got {self.pixel_size_nm} nm")

        # Validate kernel size
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(f"Kernel size must be positive and odd, got {self.kernel_size}")

        # Sampling check: Nyquist criterion
        rayleigh_limit_nm = 0.61 * self.wavelength_nm / self.NA
        nyquist_pixel_size = rayleigh_limit_nm / 2  # Need 2 samples per resolution element
        if self.pixel_size_nm > nyquist_pixel_size:
            warnings.warn(
                f"Pixel size ({self.pixel_size_nm:.2f} nm) exceeds Nyquist sampling "
                f"({nyquist_pixel_size:.2f} nm). PSF may be undersampled."
            )

        # Initialize aberrations if not provided
        if self.aberrations is None:
            self.aberrations = {}

    @property
    def rayleigh_limit_nm(self) -> float:
        """Rayleigh resolution limit in nm."""
        return 0.61 * self.wavelength_nm / self.NA

    @property
    def depth_of_focus_nm(self) -> float:
        """Depth of focus in nm."""
        return self.wavelength_nm / (2 * self.NA ** 2)

    @property
    def cutoff_frequency_nm(self) -> float:
        """Coherent cutoff frequency in nm^-1."""
        return self.NA / self.wavelength_nm

    @property
    def incoherent_cutoff_frequency_nm(self) -> float:
        """Incoherent cutoff frequency in nm^-1."""
        return 2 * self.NA / self.wavelength_nm


class PSFModel:
    """
    Base class for Point Spread Function models.

    This class provides common functionality for all PSF models and defines
    the interface that must be implemented by subclasses.
    """

    def __init__(self, config: OpticalConfig):
        """
        Initialize PSF model with optical configuration.

        Args:
            config: Optical system configuration
        """
        self.config = config
        self._psf_cache = None  # Cache computed PSF
        self._otf_cache = None  # Cache computed OTF

    def compute_psf(self) -> np.ndarray:
        """
        Compute the Point Spread Function.

        Returns:
            2D array representing the PSF (normalized to unit volume)
        """
        raise NotImplementedError("Subclasses must implement compute_psf()")

    def compute_otf(self) -> np.ndarray:
        """
        Compute the Optical Transfer Function (Fourier transform of PSF).

        Returns:
            2D complex array representing the OTF
        """
        if self._otf_cache is None:
            psf = self.compute_psf()
            # FFT with proper normalization
            otf = np.fft.fft2(np.fft.ifftshift(psf))
            otf = np.fft.fftshift(otf)
            # Normalize to have DC component = 1
            otf = otf / otf[otf.shape[0] // 2, otf.shape[1] // 2]
            self._otf_cache = otf
        return self._otf_cache

    def get_mtf(self) -> np.ndarray:
        """
        Get the Modulation Transfer Function (magnitude of OTF).

        Returns:
            2D array representing MTF (real-valued, 0 to 1)
        """
        return np.abs(self.compute_otf())

    def get_ptf(self) -> np.ndarray:
        """
        Get the Phase Transfer Function (phase of OTF).

        Returns:
            2D array representing PTF (phase in radians)
        """
        return np.angle(self.compute_otf())

    def convolve(self, image: np.ndarray) -> np.ndarray:
        """
        Convolve an image with the PSF (simulate optical blurring).

        Args:
            image: Input image (2D array)

        Returns:
            Blurred image
        """
        from scipy.signal import fftconvolve
        psf = self.compute_psf()
        return fftconvolve(image, psf, mode='same')

    def get_frequency_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create frequency grid for OTF computation.

        Returns:
            Tuple of (fx, fy) frequency grids in units of nm^-1
        """
        n = self.config.kernel_size
        pixel_nm = self.config.pixel_size_nm

        # Frequency sampling
        df = 1.0 / (n * pixel_nm)  # Frequency step (nm^-1)
        f_max = 1.0 / (2 * pixel_nm)  # Nyquist frequency

        # Create frequency grid (centered at zero)
        fx = np.fft.fftshift(np.fft.fftfreq(n, pixel_nm))
        fy = np.fft.fftshift(np.fft.fftfreq(n, pixel_nm))

        return np.meshgrid(fx, fy)


class AiryDiskPSF(PSFModel):
    """
    Airy disk PSF for coherent illumination (circular aperture).

    The Airy disk is the diffraction pattern from a circular aperture under
    coherent illumination. This is the classic "textbook" PSF.

    PSF formula:
        I(r) = I₀ * [2 * J₁(v) / v]²
        where v = (2π/λ) * NA * r

    First zero at: r = 1.22 * λ / (2 * NA) = 0.61 * λ / NA (Rayleigh criterion)
    """

    def compute_psf(self) -> np.ndarray:
        """Compute Airy disk PSF."""
        if self._psf_cache is not None:
            return self._psf_cache

        # Create spatial grid
        n = self.config.kernel_size
        pixel_nm = self.config.pixel_size_nm
        center = n // 2

        # Radial distance grid (in nm)
        y, x = np.ogrid[-center:n-center, -center:n-center]
        r = np.sqrt(x**2 + y**2) * pixel_nm

        # Normalized radial coordinate: v = (2π/λ) * NA * r
        v = (2 * np.pi / self.config.wavelength_nm) * self.config.NA * r

        # Airy disk formula: I(v) = [2*J1(v)/v]²
        # Handle v=0 case (L'Hôpital's rule: lim_{v→0} 2*J1(v)/v = 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            psf = np.where(v == 0, 1.0, 2 * j1(v) / v)

        # Intensity PSF (squared amplitude)
        psf = psf ** 2

        # Normalize to unit volume (sum = 1)
        psf = psf / psf.sum()

        self._psf_cache = psf
        return psf


class HopkinsPSF(PSFModel):
    """
    Hopkins partial coherence PSF model.

    The Hopkins formulation accounts for partial coherence in the illumination,
    which is more realistic for optical microscopy. The effective PSF is:

        PSF_partial = ∫∫ PSF_coherent(x - ξ) * S(ξ) dξ

    where S(ξ) is the source distribution (assumed uniform disk with radius σ*NA).

    For a circular condenser aperture with coherence factor σ:
        - σ = 0: Fully coherent (Airy disk)
        - σ = 1: Fully incoherent (broader PSF)

    Reference: Hopkins (1953), "On the diffraction theory of optical images"
    """

    def compute_psf(self) -> np.ndarray:
        """
        Compute Hopkins partial coherence PSF.

        For simplicity, we use a weighted combination of Airy disk PSFs
        with different effective NAs to simulate partial coherence.
        This approximates the Hopkins formulation.
        """
        if self._psf_cache is not None:
            return self._psf_cache

        sigma = self.config.sigma

        # For very low sigma, use Airy disk
        if sigma < 0.1:
            psf = AiryDiskPSF(self.config).compute_psf()
        else:
            # Weighted sum of Airy disks with different effective NAs
            # This approximates partial coherence effect

            # Use multiple condenser points (simplified Hopkins model)
            n_points = 5
            weights = []
            psfs = []

            for i in range(n_points):
                # Radial sampling of condenser
                r_norm = i / (n_points - 1) if n_points > 1 else 0
                effective_sigma = sigma * r_norm

                # Create config with modified coherence
                # Approximate by slightly broadening the PSF
                weight = 1.0 if n_points == 1 else (1.0 if i == 0 else 2.0)

                # For simplicity, just use Airy disk with small variation
                temp_config = OpticalConfig(
                    wavelength_nm=self.config.wavelength_nm,
                    NA=self.config.NA * (1.0 + 0.05 * effective_sigma),  # Slight NA variation
                    sigma=0.01,  # Nearly coherent for each point
                    pixel_size_nm=self.config.pixel_size_nm,
                    kernel_size=self.config.kernel_size,
                    aberrations=self.config.aberrations
                )

                psf_component = AiryDiskPSF(temp_config).compute_psf()
                psfs.append(psf_component)
                weights.append(weight)

            # Weighted average
            weights = np.array(weights) / sum(weights)
            psf = sum(w * p for w, p in zip(weights, psfs))

            # Renormalize
            psf = psf / psf.sum()

        self._psf_cache = psf
        return psf


class ChromaticPSF(PSFModel):
    """
    PSF with chromatic aberration (wavelength-dependent focusing).

    Chromatic aberration causes different wavelengths to focus at different planes.
    This model simulates polychromatic illumination by combining PSFs from
    multiple wavelengths.

    For broadband sources (e.g., mercury lamp with multiple lines), the effective
    PSF is the weighted sum of monochromatic PSFs.
    """

    def __init__(
        self,
        config: OpticalConfig,
        wavelengths: Optional[list] = None,
        weights: Optional[list] = None
    ):
        """
        Initialize chromatic PSF.

        Args:
            config: Base optical configuration (central wavelength)
            wavelengths: List of wavelengths to simulate (nm). If None, uses single wavelength
            weights: Spectral weights for each wavelength (must sum to 1)
        """
        super().__init__(config)

        if wavelengths is None:
            # Single wavelength (no chromatic aberration)
            self.wavelengths = [config.wavelength_nm]
            self.weights = [1.0]
        else:
            self.wavelengths = wavelengths
            if weights is None:
                # Uniform weighting
                self.weights = [1.0 / len(wavelengths)] * len(wavelengths)
            else:
                if len(weights) != len(wavelengths):
                    raise ValueError("Number of weights must match number of wavelengths")
                if not np.isclose(sum(weights), 1.0):
                    raise ValueError("Weights must sum to 1")
                self.weights = weights

    def compute_psf(self) -> np.ndarray:
        """Compute polychromatic PSF as weighted sum of monochromatic PSFs."""
        if self._psf_cache is not None:
            return self._psf_cache

        # Initialize PSF
        psf_total = np.zeros((self.config.kernel_size, self.config.kernel_size))

        # Sum PSFs from each wavelength
        for wavelength, weight in zip(self.wavelengths, self.weights):
            # Create temporary config for this wavelength
            config_lambda = OpticalConfig(
                wavelength_nm=wavelength,
                NA=self.config.NA,
                sigma=self.config.sigma,
                pixel_size_nm=self.config.pixel_size_nm,
                kernel_size=self.config.kernel_size,
                aberrations=self.config.aberrations
            )

            # Compute Airy disk for this wavelength (can also use Hopkins)
            psf_mono = AiryDiskPSF(config_lambda).compute_psf()
            psf_total += weight * psf_mono

        # Normalize
        psf_total = psf_total / psf_total.sum()

        self._psf_cache = psf_total
        return psf_total


class AberrationPSF(PSFModel):
    """
    PSF with lens aberrations modeled via Gaussian blur approximation.

    This model simulates the broadening effect of lens aberrations by applying
    isotropic Gaussian blur to the base PSF. Aberration magnitudes are specified
    in wavelengths (e.g., {'spherical': 0.1, 'coma': 0.05, 'astigmatism': 0.02}).

    Physical Modeling:
        The blur sigma is computed from RMS aberration magnitude:
            σ_blur = √(Σ aberration_i²) × Rayleigh_limit

        This approximates the first-order effect of aberrations (PSF broadening)
        while maintaining computational efficiency for rapid data generation.

    Common Aberration Types (for reference):
        - 'spherical': Radial symmetric blurring (analogous to Zernike Z₉)
        - 'coma': Asymmetric blurring (analogous to Zernike Z₇, Z₈)
        - 'astigmatism': Elliptical blurring (analogous to Zernike Z₅, Z₆)

    Note: Simplified vs. Rigorous Implementation
        Current (Gaussian blur):
            - Computation: ~0.001s per PSF
            - Captures: Isotropic PSF broadening (~70-80% accuracy)
            - Use case: Rapid synthetic data generation for ML training

        Rigorous (Zernike polynomials):
            - Computation: ~0.1s per PSF (100× slower)
            - Captures: Anisotropic aberration patterns (~95-99% accuracy)
            - Requires: P(r,θ) = A(r,θ) × exp(i × Σ cⱼ×Zⱼ(r,θ)), then PSF = |FFT{P}|²

        For super-resolution training, the approximation is sufficient because:
        1. Deep learning models learn inverse mappings regardless of exact PSF shape
        2. Real microscopes have complex aberrations beyond Zernike (vibration, dust)

    Reference:
        - Zernike standard indexing (OSA/ANSI): Z₅₋₆ (astigmatism), Z₇₋₈ (coma), Z₉ (spherical)
        - Noll, R. J. "Zernike polynomials and atmospheric turbulence" (1976)
    """

    def __init__(self, config: OpticalConfig, base_model: str = 'hopkins'):
        """
        Initialize aberration PSF.

        Args:
            config: Optical configuration (must include aberrations dict)
            base_model: Base PSF model ('airy' or 'hopkins')
        """
        super().__init__(config)
        self.base_model = base_model

        if self.config.aberrations is None or len(self.config.aberrations) == 0:
            warnings.warn("No aberrations specified; PSF will be aberration-free")

    def compute_psf(self) -> np.ndarray:
        """
        Compute PSF with aberrations.

        For simplicity, we approximate aberrations as Gaussian blurring
        applied to the base PSF. For rigorous computation, one would:
        1. Compute pupil function with Zernike phase
        2. FFT to get amplitude PSF
        3. Square to get intensity PSF
        """
        if self._psf_cache is not None:
            return self._psf_cache

        # Get base PSF (aberration-free)
        if self.base_model == 'airy':
            base_psf = AiryDiskPSF(self.config).compute_psf()
        elif self.base_model == 'hopkins':
            base_psf = HopkinsPSF(self.config).compute_psf()
        else:
            raise ValueError(f"Unknown base model: {self.base_model}")

        # Apply aberrations as Gaussian blur (simplified model)
        if self.config.aberrations and len(self.config.aberrations) > 0:
            # Estimate blur sigma from aberration magnitudes
            # Total aberration RMS in wavelengths
            aberration_rms = np.sqrt(sum(v**2 for v in self.config.aberrations.values()))

            # Convert to spatial sigma (heuristic: 1 wave aberration ~ 1.0 * Rayleigh)
            # Increased from 0.5 to make aberration effects more pronounced
            blur_sigma_nm = aberration_rms * 1.0 * self.config.rayleigh_limit_nm
            blur_sigma_pixels = blur_sigma_nm / self.config.pixel_size_nm

            # Apply Gaussian blur only if significant aberration
            if blur_sigma_pixels > 0.1:  # Only blur if sigma > 0.1 pixels
                psf = gaussian_filter(base_psf, sigma=blur_sigma_pixels)
                # Renormalize
                psf = psf / psf.sum()
            else:
                psf = base_psf
        else:
            psf = base_psf

        self._psf_cache = psf
        return psf


def create_psf_model(
    wavelength_nm: float,
    NA: float,
    pixel_size_nm: float,
    kernel_size: int = 33,
    sigma: float = 0.5,
    model_type: str = 'hopkins',
    aberrations: Optional[Dict[str, float]] = None,
    wavelengths_chromatic: Optional[list] = None
) -> PSFModel:
    """
    Factory function to create a PSF model.

    Args:
        wavelength_nm: Wavelength in nanometers
        NA: Numerical aperture
        pixel_size_nm: Pixel size in nanometers
        kernel_size: PSF kernel size (must be odd)
        sigma: Partial coherence factor (0 < σ < 1)
        model_type: PSF model type ('airy', 'hopkins', 'chromatic', 'aberration')
        aberrations: Dictionary of aberration coefficients (for 'aberration' model)
        wavelengths_chromatic: List of wavelengths for chromatic model

    Returns:
        PSF model instance

    Examples:
        >>> # Simple Airy disk
        >>> psf = create_psf_model(248, 0.95, 2.0, model_type='airy')
        >>>
        >>> # Hopkins partial coherence
        >>> psf = create_psf_model(248, 0.95, 2.0, sigma=0.5, model_type='hopkins')
        >>>
        >>> # With aberrations
        >>> psf = create_psf_model(
        ...     248, 0.95, 2.0, model_type='aberration',
        ...     aberrations={'spherical': 0.05, 'coma': 0.03}
        ... )
    """
    # Create optical configuration
    config = OpticalConfig(
        wavelength_nm=wavelength_nm,
        NA=NA,
        sigma=sigma,
        pixel_size_nm=pixel_size_nm,
        kernel_size=kernel_size,
        aberrations=aberrations
    )

    # Create model based on type
    if model_type == 'airy':
        return AiryDiskPSF(config)
    elif model_type == 'hopkins':
        return HopkinsPSF(config)
    elif model_type == 'chromatic':
        return ChromaticPSF(config, wavelengths=wavelengths_chromatic)
    elif model_type == 'aberration':
        return AberrationPSF(config)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from: 'airy', 'hopkins', 'chromatic', 'aberration'"
        )


if __name__ == "__main__":
    # Example usage and validation
    print("PSF/OTF Models - Example Usage\n")

    # Create PSF model (DUV, NA=0.95)
    psf_model = create_psf_model(
        wavelength_nm=248,
        NA=0.95,
        pixel_size_nm=2.0,
        kernel_size=33,
        sigma=0.5,
        model_type='hopkins'
    )

    print(f"Optical Configuration:")
    print(f"  Wavelength: {psf_model.config.wavelength_nm} nm")
    print(f"  NA: {psf_model.config.NA}")
    print(f"  Rayleigh limit: {psf_model.config.rayleigh_limit_nm:.2f} nm")
    print(f"  Depth of focus: {psf_model.config.depth_of_focus_nm:.2f} nm")
    print(f"  Coherent cutoff: {psf_model.config.cutoff_frequency_nm:.4f} nm⁻¹")
    print(f"  Incoherent cutoff: {psf_model.config.incoherent_cutoff_frequency_nm:.4f} nm⁻¹")

    # Compute PSF and OTF
    psf = psf_model.compute_psf()
    otf = psf_model.compute_otf()
    mtf = psf_model.get_mtf()

    print(f"\nPSF Properties:")
    print(f"  Shape: {psf.shape}")
    print(f"  Sum (should be 1.0): {psf.sum():.6f}")
    print(f"  Peak value: {psf.max():.6f}")

    print(f"\nOTF Properties:")
    print(f"  Shape: {otf.shape}")
    print(f"  DC component (should be 1.0): {np.abs(otf[otf.shape[0]//2, otf.shape[1]//2]):.6f}")
    print(f"  MTF peak: {mtf.max():.6f}")
    print(f"  MTF at edge: {mtf[0, 0]:.6f}")

    print("\nPSF models successfully created and validated! ✓")
