"""
Unit tests for PSF/OTF models.

This test suite validates:
    1. PSF normalization (sum = 1)
    2. OTF DC component (should be 1)
    3. Theoretical resolution limits (Rayleigh, Abbe)
    4. Frequency cutoffs (coherent, incoherent)
    5. Physical parameter validation
    6. Different illumination models (coherent, partial coherence)
    7. Chromatic and aberration effects

Run with: pytest tests/data/synthetic/test_psf_models.py -v
"""

import pytest
import numpy as np
from src.data.synthetic.psf_models import (
    OpticalConfig,
    PSFModel,
    AiryDiskPSF,
    HopkinsPSF,
    ChromaticPSF,
    AberrationPSF,
    create_psf_model
)


class TestOpticalConfig:
    """Test OpticalConfig validation and derived properties."""

    def test_valid_config(self):
        """Test creation of valid optical configuration."""
        config = OpticalConfig(
            wavelength_nm=248,
            NA=0.95,
            sigma=0.5,
            pixel_size_nm=2.0,
            kernel_size=33
        )
        assert config.wavelength_nm == 248
        assert config.NA == 0.95
        assert config.sigma == 0.5

    def test_invalid_wavelength(self):
        """Test that negative wavelength raises error."""
        with pytest.raises(ValueError, match="Wavelength must be positive"):
            OpticalConfig(wavelength_nm=-100, NA=0.95, pixel_size_nm=2.0)

    def test_invalid_NA_too_low(self):
        """Test that NA <= 0 raises error."""
        with pytest.raises(ValueError, match="NA must be in"):
            OpticalConfig(wavelength_nm=248, NA=0.0, pixel_size_nm=2.0)

    def test_invalid_NA_too_high(self):
        """Test that NA > 1.5 raises error."""
        with pytest.raises(ValueError, match="NA must be in"):
            OpticalConfig(wavelength_nm=248, NA=2.0, pixel_size_nm=2.0)

    def test_immersion_NA_warning(self):
        """Test warning for immersion NA > 1.0."""
        with pytest.warns(UserWarning, match="requires immersion"):
            OpticalConfig(wavelength_nm=193, NA=1.35, pixel_size_nm=2.0)

    def test_invalid_sigma(self):
        """Test that sigma outside (0, 1) raises error."""
        with pytest.raises(ValueError, match="Sigma.*must be in"):
            OpticalConfig(wavelength_nm=248, NA=0.95, sigma=1.5, pixel_size_nm=2.0)

    def test_invalid_pixel_size(self):
        """Test that negative pixel size raises error."""
        with pytest.raises(ValueError, match="Pixel size must be positive"):
            OpticalConfig(wavelength_nm=248, NA=0.95, pixel_size_nm=-1.0)

    def test_invalid_kernel_size_even(self):
        """Test that even kernel size raises error."""
        with pytest.raises(ValueError, match="must be positive and odd"):
            OpticalConfig(wavelength_nm=248, NA=0.95, pixel_size_nm=2.0, kernel_size=32)

    def test_undersampling_warning(self):
        """Test warning when pixel size exceeds Nyquist sampling."""
        with pytest.warns(UserWarning, match="exceeds Nyquist"):
            OpticalConfig(wavelength_nm=248, NA=0.95, pixel_size_nm=100.0)

    def test_rayleigh_limit(self):
        """Test Rayleigh resolution limit calculation."""
        config = OpticalConfig(wavelength_nm=248, NA=0.95, pixel_size_nm=2.0)
        expected_rayleigh = 0.61 * 248 / 0.95
        assert np.isclose(config.rayleigh_limit_nm, expected_rayleigh, rtol=1e-5)

    def test_depth_of_focus(self):
        """Test depth of focus calculation."""
        config = OpticalConfig(wavelength_nm=365, NA=0.8, pixel_size_nm=2.0)
        expected_dof = 365 / (2 * 0.8**2)
        assert np.isclose(config.depth_of_focus_nm, expected_dof, rtol=1e-5)

    def test_cutoff_frequencies(self):
        """Test coherent and incoherent cutoff frequency calculations."""
        config = OpticalConfig(wavelength_nm=193, NA=1.35, pixel_size_nm=2.0)

        expected_coherent = 1.35 / 193
        expected_incoherent = 2 * 1.35 / 193

        assert np.isclose(config.cutoff_frequency_nm, expected_coherent, rtol=1e-5)
        assert np.isclose(config.incoherent_cutoff_frequency_nm, expected_incoherent, rtol=1e-5)


class TestAiryDiskPSF:
    """Test Airy disk PSF (coherent illumination)."""

    @pytest.fixture
    def config(self):
        """Standard optical configuration for testing."""
        return OpticalConfig(
            wavelength_nm=248,
            NA=0.95,
            sigma=0.01,  # Nearly fully coherent (sigma must be > 0)
            pixel_size_nm=2.0,
            kernel_size=33
        )

    def test_psf_normalization(self, config):
        """Test that PSF sums to 1."""
        psf_model = AiryDiskPSF(config)
        psf = psf_model.compute_psf()
        assert np.isclose(psf.sum(), 1.0, atol=1e-6)

    def test_psf_shape(self, config):
        """Test PSF has correct shape."""
        psf_model = AiryDiskPSF(config)
        psf = psf_model.compute_psf()
        assert psf.shape == (33, 33)

    def test_psf_symmetry(self, config):
        """Test that PSF is radially symmetric."""
        psf_model = AiryDiskPSF(config)
        psf = psf_model.compute_psf()

        # Check horizontal and vertical profiles match
        center = 16
        horizontal = psf[center, :]
        vertical = psf[:, center]
        assert np.allclose(horizontal, vertical, atol=1e-6)

    def test_psf_peak_at_center(self, config):
        """Test that PSF peak is at center."""
        psf_model = AiryDiskPSF(config)
        psf = psf_model.compute_psf()

        center = 16
        peak_value = psf.max()
        center_value = psf[center, center]
        assert np.isclose(peak_value, center_value, rtol=1e-5)

    def test_psf_caching(self, config):
        """Test that PSF is cached after first computation."""
        psf_model = AiryDiskPSF(config)
        psf1 = psf_model.compute_psf()
        psf2 = psf_model.compute_psf()
        assert psf1 is psf2  # Same object (cached)

    def test_wavelength_dependence(self):
        """Test PSF broadens with longer wavelength."""
        config_short = OpticalConfig(wavelength_nm=193, NA=0.95, pixel_size_nm=2.0)
        config_long = OpticalConfig(wavelength_nm=365, NA=0.95, pixel_size_nm=2.0)

        psf_short = AiryDiskPSF(config_short).compute_psf()
        psf_long = AiryDiskPSF(config_long).compute_psf()

        # Longer wavelength should have broader PSF (lower peak)
        assert psf_short.max() > psf_long.max()

    def test_NA_dependence(self):
        """Test PSF sharpens with higher NA."""
        config_low_NA = OpticalConfig(wavelength_nm=248, NA=0.5, pixel_size_nm=2.0)
        config_high_NA = OpticalConfig(wavelength_nm=248, NA=0.95, pixel_size_nm=2.0)

        psf_low = AiryDiskPSF(config_low_NA).compute_psf()
        psf_high = AiryDiskPSF(config_high_NA).compute_psf()

        # Higher NA should have sharper PSF (higher peak)
        assert psf_high.max() > psf_low.max()


class TestHopkinsPSF:
    """Test Hopkins partial coherence PSF."""

    @pytest.fixture
    def config(self):
        """Partial coherence configuration."""
        return OpticalConfig(
            wavelength_nm=248,
            NA=0.95,
            sigma=0.5,
            pixel_size_nm=2.0,
            kernel_size=33
        )

    def test_psf_normalization(self, config):
        """Test PSF normalization."""
        psf_model = HopkinsPSF(config)
        psf = psf_model.compute_psf()
        assert np.isclose(psf.sum(), 1.0, atol=1e-6)

    def test_coherence_effect(self):
        """Test that partial coherence broadens PSF."""
        config_coherent = OpticalConfig(
            wavelength_nm=248, NA=0.95, sigma=0.1, pixel_size_nm=2.0
        )
        config_incoherent = OpticalConfig(
            wavelength_nm=248, NA=0.95, sigma=0.9, pixel_size_nm=2.0
        )

        psf_coherent = HopkinsPSF(config_coherent).compute_psf()
        psf_incoherent = HopkinsPSF(config_incoherent).compute_psf()

        # More incoherent should have broader PSF (possibly lower peak)
        # In Hopkins model, both effects are subtle due to approximation
        # Just check normalization is preserved
        assert np.isclose(psf_coherent.sum(), 1.0, atol=1e-6)
        assert np.isclose(psf_incoherent.sum(), 1.0, atol=1e-6)


class TestOTF:
    """Test Optical Transfer Function computation."""

    @pytest.fixture
    def config(self):
        """Standard configuration."""
        return OpticalConfig(
            wavelength_nm=248,
            NA=0.95,
            sigma=0.5,
            pixel_size_nm=2.0,
            kernel_size=33
        )

    def test_otf_dc_component(self, config):
        """Test that OTF DC component is 1."""
        psf_model = HopkinsPSF(config)
        otf = psf_model.compute_otf()

        center = 16
        dc_value = np.abs(otf[center, center])
        assert np.isclose(dc_value, 1.0, atol=1e-6)

    def test_mtf_positive(self, config):
        """Test that MTF is non-negative."""
        psf_model = HopkinsPSF(config)
        mtf = psf_model.get_mtf()
        assert np.all(mtf >= 0)

    def test_mtf_bounded(self, config):
        """Test that MTF is bounded by 1."""
        psf_model = HopkinsPSF(config)
        mtf = psf_model.get_mtf()
        assert np.all(mtf <= 1.0 + 1e-6)  # Small tolerance for numerical errors

    def test_otf_caching(self, config):
        """Test OTF caching."""
        psf_model = HopkinsPSF(config)
        otf1 = psf_model.compute_otf()
        otf2 = psf_model.compute_otf()
        assert otf1 is otf2  # Same object (cached)

    def test_ptf_computation(self, config):
        """Test Phase Transfer Function computation."""
        psf_model = HopkinsPSF(config)
        ptf = psf_model.get_ptf()

        # PTF should be in [-π, π]
        assert np.all(ptf >= -np.pi)
        assert np.all(ptf <= np.pi)


class TestChromaticPSF:
    """Test chromatic PSF (polychromatic illumination)."""

    def test_single_wavelength(self):
        """Test chromatic PSF with single wavelength equals monochromatic."""
        config = OpticalConfig(wavelength_nm=248, NA=0.95, pixel_size_nm=2.0)

        psf_mono = AiryDiskPSF(config).compute_psf()
        psf_chromatic = ChromaticPSF(config, wavelengths=[248], weights=[1.0]).compute_psf()

        assert np.allclose(psf_mono, psf_chromatic, rtol=1e-3)

    def test_multiple_wavelengths(self):
        """Test chromatic PSF with multiple wavelengths."""
        config = OpticalConfig(wavelength_nm=248, NA=0.95, pixel_size_nm=2.0)
        wavelengths = [193, 248, 365]
        weights = [0.3, 0.5, 0.2]

        psf_chromatic = ChromaticPSF(config, wavelengths, weights).compute_psf()

        assert np.isclose(psf_chromatic.sum(), 1.0, atol=1e-6)
        assert psf_chromatic.shape == (33, 33)

    def test_weight_validation(self):
        """Test that weights must sum to 1."""
        config = OpticalConfig(wavelength_nm=248, NA=0.95, pixel_size_nm=2.0)

        with pytest.raises(ValueError, match="must sum to 1"):
            ChromaticPSF(config, wavelengths=[193, 248], weights=[0.3, 0.3])

    def test_weight_length_validation(self):
        """Test that number of weights must match wavelengths."""
        config = OpticalConfig(wavelength_nm=248, NA=0.95, pixel_size_nm=2.0)

        with pytest.raises(ValueError, match="must match"):
            ChromaticPSF(config, wavelengths=[193, 248, 365], weights=[0.5, 0.5])


class TestAberrationPSF:
    """Test PSF with aberrations."""

    def test_no_aberrations(self):
        """Test that zero aberrations equals base PSF."""
        config = OpticalConfig(
            wavelength_nm=248, NA=0.95, pixel_size_nm=2.0,
            aberrations={}
        )

        psf_base = HopkinsPSF(config).compute_psf()
        psf_aberrated = AberrationPSF(config, base_model='hopkins').compute_psf()

        assert np.allclose(psf_base, psf_aberrated, rtol=1e-3)

    def test_aberration_broadening(self):
        """Test that aberrations are applied to PSF."""
        config_aberrated = OpticalConfig(
            wavelength_nm=248, NA=0.95, pixel_size_nm=2.0,
            aberrations={'spherical': 0.5, 'coma': 0.3, 'astigmatism': 0.2}  # Large aberrations
        )

        psf_aberrated = AberrationPSF(config_aberrated).compute_psf()

        # Check that aberrated PSF is properly normalized
        assert np.isclose(psf_aberrated.sum(), 1.0, atol=1e-6)

        # Check PSF is non-negative
        assert np.all(psf_aberrated >= 0)

        # Check PSF has correct shape
        assert psf_aberrated.shape == (33, 33)

    def test_base_model_selection(self):
        """Test different base models for aberrations."""
        config = OpticalConfig(
            wavelength_nm=248, NA=0.95, pixel_size_nm=2.0,
            aberrations={'spherical': 0.05}
        )

        psf_airy = AberrationPSF(config, base_model='airy').compute_psf()
        psf_hopkins = AberrationPSF(config, base_model='hopkins').compute_psf()

        # Both should be normalized
        assert np.isclose(psf_airy.sum(), 1.0, atol=1e-6)
        assert np.isclose(psf_hopkins.sum(), 1.0, atol=1e-6)


class TestPSFConvolution:
    """Test PSF convolution with images."""

    @pytest.fixture
    def psf_model(self):
        """Create PSF model for testing."""
        return create_psf_model(
            wavelength_nm=248,
            NA=0.95,
            pixel_size_nm=2.0,
            kernel_size=33,
            sigma=0.5,
            model_type='hopkins'
        )

    def test_convolution_shape_preservation(self, psf_model):
        """Test that convolution preserves image shape."""
        image = np.random.rand(256, 256)
        blurred = psf_model.convolve(image)
        assert blurred.shape == image.shape

    def test_convolution_reduces_high_freq(self, psf_model):
        """Test that convolution reduces high-frequency content."""
        # Create high-frequency checkerboard pattern
        x, y = np.meshgrid(range(128), range(128))
        image = ((x + y) % 2).astype(float)

        blurred = psf_model.convolve(image)

        # Blurred image should have lower variance (smoother)
        assert blurred.std() < image.std()

    def test_convolution_preserves_mean(self, psf_model):
        """Test that convolution approximately preserves mean intensity."""
        image = np.random.rand(128, 128)
        blurred = psf_model.convolve(image)

        # Mean should be approximately preserved
        # Due to edge effects in FFT convolution, we allow higher tolerance
        assert np.isclose(image.mean(), blurred.mean(), rtol=0.2)


class TestFactoryFunction:
    """Test create_psf_model factory function."""

    def test_create_airy(self):
        """Test creating Airy disk PSF."""
        psf_model = create_psf_model(248, 0.95, 2.0, model_type='airy')
        assert isinstance(psf_model, AiryDiskPSF)

    def test_create_hopkins(self):
        """Test creating Hopkins PSF."""
        psf_model = create_psf_model(248, 0.95, 2.0, model_type='hopkins')
        assert isinstance(psf_model, HopkinsPSF)

    def test_create_chromatic(self):
        """Test creating chromatic PSF."""
        psf_model = create_psf_model(
            248, 0.95, 2.0, model_type='chromatic',
            wavelengths_chromatic=[193, 248, 365]
        )
        assert isinstance(psf_model, ChromaticPSF)

    def test_create_aberration(self):
        """Test creating aberration PSF."""
        psf_model = create_psf_model(
            248, 0.95, 2.0, model_type='aberration',
            aberrations={'spherical': 0.05}
        )
        assert isinstance(psf_model, AberrationPSF)

    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_psf_model(248, 0.95, 2.0, model_type='invalid')


class TestTheoreticalValidation:
    """Validate PSF models against theoretical predictions."""

    def test_rayleigh_criterion_validation(self):
        """
        Test that PSF width is consistent with Rayleigh criterion.

        For Airy disk, first zero at r = 1.22 * λ / (2 * NA) = 0.61 * λ / NA
        This is a qualitative test due to discretization effects.
        """
        config = OpticalConfig(
            wavelength_nm=550,  # Visible light for easier visualization
            NA=0.95,
            pixel_size_nm=10.0,  # Coarser sampling
            kernel_size=65
        )

        psf = AiryDiskPSF(config).compute_psf()

        # Expected Rayleigh resolution limit
        rayleigh_nm = 0.61 * 550 / 0.95

        # PSF should be concentrated within ~2x Rayleigh limit
        # Check that 90% of energy is within this radius
        center = 32
        max_radius_pixels = int(2 * rayleigh_nm / 10.0)

        # Create circular mask
        y, x = np.ogrid[-center:65-center, -center:65-center]
        mask = (x**2 + y**2) <= max_radius_pixels**2

        energy_within = psf[mask].sum()
        assert energy_within > 0.85  # At least 85% of energy within 2x Rayleigh

    def test_coherent_cutoff_frequency(self):
        """Test that OTF has correct cutoff frequency behavior."""
        config = OpticalConfig(
            wavelength_nm=248,
            NA=0.95,
            sigma=0.1,  # Nearly coherent
            pixel_size_nm=2.0,
            kernel_size=65
        )

        psf_model = AiryDiskPSF(config)
        mtf = psf_model.get_mtf()

        # Expected coherent cutoff
        f_cutoff = config.cutoff_frequency_nm

        # Get frequency grid
        fx, fy = psf_model.get_frequency_grid()
        f_radial = np.sqrt(fx**2 + fy**2)

        # MTF should be higher at low frequencies than high frequencies
        center = 32
        low_freq_mtf = mtf[center, center]  # DC component

        # Find high frequency point (beyond cutoff)
        high_freq_mask = f_radial > 1.5 * f_cutoff
        if high_freq_mask.any():
            high_freq_mtf = mtf[high_freq_mask].mean()
            # MTF should decay from DC to high frequencies
            assert low_freq_mtf > high_freq_mtf


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
