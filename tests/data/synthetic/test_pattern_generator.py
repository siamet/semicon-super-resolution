"""
Unit tests for pattern_generator module.

Tests all pattern generator classes with expected cases, edge cases, and failure cases.
"""

import pytest
import numpy as np
from src.data.synthetic.pattern_generator import (
    PatternConfig,
    PatternGenerator,
    GratingGenerator,
    ContactHoleGenerator,
    IsolatedFeatureGenerator,
    create_pattern_generator
)


class TestPatternConfig:
    """Test PatternConfig dataclass."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = PatternConfig()
        assert config.image_size == 512
        assert config.pixel_size_nm == 2.0
        assert config.field_size_nm == 1024.0  # 512 * 2.0

    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = PatternConfig(image_size=256, pixel_size_nm=1.0)
        assert config.image_size == 256
        assert config.pixel_size_nm == 1.0
        assert config.field_size_nm == 256.0

    def test_field_size_calculation(self):
        """Test automatic field size calculation."""
        config = PatternConfig(image_size=1024, pixel_size_nm=0.5)
        assert config.field_size_nm == 512.0


class TestGratingGenerator:
    """Test GratingGenerator class."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return PatternConfig(image_size=128, pixel_size_nm=2.0)

    @pytest.fixture
    def generator(self, config):
        """Provide grating generator instance."""
        return GratingGenerator(config)

    def test_basic_generation(self, generator):
        """Test basic grating generation."""
        pattern, metadata = generator.generate(pitch_nm=40.0, duty_cycle=0.5, add_ler=False)
        
        assert pattern.shape == (128, 128)
        assert pattern.min() >= 0.0
        assert pattern.max() <= 1.0
        assert metadata['pattern_type'] == 'grating'
        assert metadata['pitch_nm'] == 40.0
        assert metadata['duty_cycle'] == 0.5

    def test_duty_cycle_variations(self, generator):
        """Test different duty cycles."""
        # 50% duty cycle
        pattern_50, _ = generator.generate(pitch_nm=40.0, duty_cycle=0.5, add_ler=False)
        mean_50 = pattern_50.mean()

        # 25% duty cycle (less lines, more background)
        pattern_25, _ = generator.generate(pitch_nm=40.0, duty_cycle=0.25, add_ler=False)
        mean_25 = pattern_25.mean()

        # 75% duty cycle (more lines, less background)
        pattern_75, _ = generator.generate(pitch_nm=40.0, duty_cycle=0.75, add_ler=False)
        mean_75 = pattern_75.mean()

        # Mean intensity should increase with duty cycle
        assert mean_25 < mean_50 < mean_75

    def test_orientation(self, generator):
        """Test pattern orientation."""
        pattern_0, _ = generator.generate(pitch_nm=40.0, orientation_deg=0.0, add_ler=False)
        pattern_90, _ = generator.generate(pitch_nm=40.0, orientation_deg=90.0, add_ler=False)

        # Patterns should be different
        assert not np.allclose(pattern_0, pattern_90)

    def test_ler_addition(self, generator):
        """Test line edge roughness addition."""
        pattern_no_ler, _ = generator.generate(pitch_nm=40.0, add_ler=False)
        pattern_with_ler, _ = generator.generate(pitch_nm=40.0, add_ler=True, ler_sigma_nm=2.0)

        # Patterns should be different
        assert not np.allclose(pattern_no_ler, pattern_with_ler)
        # LER pattern should have some variation (may still clip to 0 or 1 at extremes)
        # Check that the pattern has been modified by LER
        assert pattern_with_ler.std() > 0

    def test_invalid_pitch(self, generator):
        """Test negative pitch raises ValueError."""
        with pytest.raises(ValueError, match="pitch_nm must be positive"):
            generator.generate(pitch_nm=-10.0)

    def test_invalid_duty_cycle(self, generator):
        """Test invalid duty cycle raises ValueError."""
        with pytest.raises(ValueError, match="duty_cycle must be in"):
            generator.generate(duty_cycle=1.5)

        with pytest.raises(ValueError, match="duty_cycle must be in"):
            generator.generate(duty_cycle=-0.1)

    def test_small_pitch_warning(self, generator):
        """Test warning for very small pitch."""
        with pytest.warns(UserWarning, match="very small"):
            generator.generate(pitch_nm=2.0, add_ler=False)


class TestContactHoleGenerator:
    """Test ContactHoleGenerator class."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return PatternConfig(image_size=128, pixel_size_nm=2.0)

    @pytest.fixture
    def generator(self, config):
        """Provide contact hole generator instance."""
        return ContactHoleGenerator(config)

    def test_basic_generation_circular(self, generator):
        """Test basic circular contact hole generation."""
        pattern, metadata = generator.generate(
            diameter_nm=30.0, 
            pitch_nm=60.0, 
            shape='circular',
            add_ler=False
        )
        
        assert pattern.shape == (128, 128)
        assert pattern.min() >= 0.0
        assert pattern.max() <= 1.0
        assert metadata['pattern_type'] == 'contact_holes'
        assert metadata['shape'] == 'circular'

    def test_basic_generation_square(self, generator):
        """Test basic square contact hole generation."""
        pattern, metadata = generator.generate(
            diameter_nm=30.0,
            shape='square',
            add_ler=False
        )
        
        assert pattern.shape == (128, 128)
        assert metadata['shape'] == 'square'

    def test_array_types(self, generator):
        """Test regular vs staggered arrays."""
        pattern_regular, _ = generator.generate(
            diameter_nm=20.0,
            pitch_nm=50.0,
            array_type='regular',
            add_ler=False
        )

        pattern_staggered, _ = generator.generate(
            diameter_nm=20.0,
            pitch_nm=50.0,
            array_type='staggered',
            add_ler=False
        )

        # Patterns should be different
        assert not np.allclose(pattern_regular, pattern_staggered)

    def test_default_pitch(self, generator):
        """Test default pitch calculation (2× diameter)."""
        pattern, metadata = generator.generate(diameter_nm=30.0, pitch_nm=None, add_ler=False)
        assert metadata['pitch_nm'] == 60.0  # 2 × 30

    def test_invalid_diameter(self, generator):
        """Test negative diameter raises ValueError."""
        with pytest.raises(ValueError, match="diameter_nm must be positive"):
            generator.generate(diameter_nm=-10.0)

    def test_invalid_shape(self, generator):
        """Test invalid shape raises ValueError."""
        with pytest.raises(ValueError, match="shape must be"):
            generator.generate(shape='triangular')

    def test_invalid_array_type(self, generator):
        """Test invalid array type raises ValueError."""
        with pytest.raises(ValueError, match="array_type must be"):
            generator.generate(array_type='hexagonal')

    def test_overlapping_holes_warning(self, generator):
        """Test warning when pitch < diameter."""
        with pytest.warns(UserWarning, match="will overlap"):
            generator.generate(diameter_nm=50.0, pitch_nm=30.0, add_ler=False)


class TestIsolatedFeatureGenerator:
    """Test IsolatedFeatureGenerator class."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return PatternConfig(image_size=128, pixel_size_nm=2.0)

    @pytest.fixture
    def generator(self, config):
        """Provide isolated feature generator instance."""
        return IsolatedFeatureGenerator(config)

    def test_isolated_line(self, generator):
        """Test isolated line generation."""
        pattern, metadata = generator.generate(
            feature_type='line',
            width_nm=30.0,
            length_nm=150.0,
            add_ler=False
        )
        
        assert pattern.shape == (128, 128)
        assert pattern.max() == 1.0  # Should have bright line
        assert pattern.mean() < 0.5  # Mostly dark background
        assert metadata['feature_type'] == 'line'

    def test_isolated_space(self, generator):
        """Test isolated space (trench) generation."""
        pattern, metadata = generator.generate(
            feature_type='space',
            width_nm=30.0,
            length_nm=150.0,
            add_ler=False
        )
        
        assert pattern.shape == (128, 128)
        assert pattern.min() == 0.0  # Should have dark space
        assert pattern.mean() > 0.5  # Mostly bright background
        assert metadata['feature_type'] == 'space'

    def test_isolated_post(self, generator):
        """Test isolated post generation."""
        pattern, metadata = generator.generate(
            feature_type='post',
            width_nm=40.0,
            add_ler=False
        )
        
        assert pattern.shape == (128, 128)
        assert pattern.max() == 1.0  # Should have bright post
        assert pattern.mean() < 0.5  # Mostly dark background
        assert metadata['feature_type'] == 'post'

    def test_orientation(self, generator):
        """Test feature orientation."""
        pattern_0, _ = generator.generate(
            feature_type='line',
            width_nm=20.0,
            orientation_deg=0.0,
            add_ler=False
        )

        pattern_45, _ = generator.generate(
            feature_type='line',
            width_nm=20.0,
            orientation_deg=45.0,
            add_ler=False
        )

        # Patterns should be different
        assert not np.allclose(pattern_0, pattern_45)

    def test_default_length(self, generator):
        """Test default length calculation (70% of field)."""
        pattern, metadata = generator.generate(
            feature_type='line',
            width_nm=20.0,
            length_nm=None,
            add_ler=False
        )
        
        # Field size is 128 * 2.0 = 256nm, so 70% = 179.2nm
        assert metadata['length_nm'] == pytest.approx(179.2, rel=1e-5)

    def test_invalid_feature_type(self, generator):
        """Test invalid feature type raises ValueError."""
        with pytest.raises(ValueError, match="feature_type must be"):
            generator.generate(feature_type='circle')

    def test_invalid_width(self, generator):
        """Test negative width raises ValueError."""
        with pytest.raises(ValueError, match="width_nm must be positive"):
            generator.generate(width_nm=-10.0)

    def test_invalid_length(self, generator):
        """Test negative length raises ValueError."""
        with pytest.raises(ValueError, match="length_nm must be positive"):
            generator.generate(length_nm=-10.0)


class TestPatternGeneratorBase:
    """Test base PatternGenerator class methods."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return PatternConfig(image_size=64, pixel_size_nm=2.0)

    @pytest.fixture
    def generator(self, config):
        """Provide concrete generator for testing base class methods."""
        return GratingGenerator(config)

    def test_add_line_edge_roughness(self, generator):
        """Test LER addition to pattern."""
        # Create simple binary pattern
        pattern = np.zeros((64, 64))
        pattern[:, 20:40] = 1.0  # Vertical lines

        pattern_with_ler = generator.add_line_edge_roughness(
            pattern, 
            sigma_nm=2.0,
            correlation_length_nm=20.0
        )

        # Pattern should have intermediate values at edges
        assert 0 <= pattern_with_ler.min() < 1
        assert 0 < pattern_with_ler.max() <= 1
        # Patterns should be different
        assert not np.allclose(pattern, pattern_with_ler)

    def test_ler_invalid_sigma(self, generator):
        """Test negative sigma raises ValueError."""
        pattern = np.ones((64, 64))
        
        with pytest.raises(ValueError, match="sigma_nm must be non-negative"):
            generator.add_line_edge_roughness(pattern, sigma_nm=-1.0)

    def test_ler_invalid_correlation(self, generator):
        """Test invalid correlation length raises ValueError."""
        pattern = np.ones((64, 64))
        
        with pytest.raises(ValueError, match="correlation_length_nm must be positive"):
            generator.add_line_edge_roughness(pattern, correlation_length_nm=-10.0)

    def test_add_corner_rounding(self, generator):
        """Test corner rounding addition."""
        # Create sharp corner pattern
        pattern = np.zeros((64, 64))
        pattern[20:40, 20:40] = 1.0  # Square

        pattern_rounded = generator.add_corner_rounding(pattern, blur_sigma_nm=5.0)

        # Rounded pattern should be smoother
        assert pattern_rounded.shape == pattern.shape
        assert 0 <= pattern_rounded.min() < pattern_rounded.max() <= 1
        # Should have intermediate values (not binary)
        unique_values = np.unique(pattern_rounded)
        assert len(unique_values) > 2

    def test_corner_rounding_invalid_sigma(self, generator):
        """Test invalid blur sigma raises ValueError."""
        pattern = np.ones((64, 64))
        
        with pytest.raises(ValueError, match="blur_sigma_nm must be positive"):
            generator.add_corner_rounding(pattern, blur_sigma_nm=-1.0)


class TestFactoryFunction:
    """Test create_pattern_generator factory function."""

    def test_create_grating_generator(self):
        """Test factory creates GratingGenerator."""
        generator = create_pattern_generator('grating')
        assert isinstance(generator, GratingGenerator)

    def test_create_contacts_generator(self):
        """Test factory creates ContactHoleGenerator."""
        generator = create_pattern_generator('contacts')
        assert isinstance(generator, ContactHoleGenerator)

    def test_create_isolated_generator(self):
        """Test factory creates IsolatedFeatureGenerator."""
        generator = create_pattern_generator('isolated')
        assert isinstance(generator, IsolatedFeatureGenerator)

    def test_custom_config(self):
        """Test factory with custom configuration."""
        config = PatternConfig(image_size=256, pixel_size_nm=1.0)
        generator = create_pattern_generator('grating', config)
        
        assert generator.config.image_size == 256
        assert generator.config.pixel_size_nm == 1.0

    def test_invalid_pattern_type(self):
        """Test invalid pattern type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown pattern_type"):
            create_pattern_generator('invalid_type')


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_grating(self):
        """Test complete grating generation workflow."""
        config = PatternConfig(image_size=128, pixel_size_nm=2.0)
        generator = GratingGenerator(config)
        
        # Generate pattern with all features
        pattern, metadata = generator.generate(
            pitch_nm=50.0,
            duty_cycle=0.5,
            orientation_deg=45.0,
            add_ler=True,
            ler_sigma_nm=2.0,
            ler_correlation_nm=25.0
        )

        # Verify pattern
        assert pattern.shape == (128, 128)
        assert 0 <= pattern.min() <= pattern.max() <= 1
        assert metadata['pitch_nm'] == 50.0
        assert metadata['ler_sigma_nm'] == 2.0

    def test_full_workflow_contacts(self):
        """Test complete contact hole generation workflow."""
        config = PatternConfig(image_size=128, pixel_size_nm=2.0)
        generator = ContactHoleGenerator(config)
        
        pattern, metadata = generator.generate(
            diameter_nm=40.0,
            pitch_nm=80.0,
            shape='circular',
            array_type='staggered',
            add_ler=True
        )

        assert pattern.shape == (128, 128)
        assert metadata['shape'] == 'circular'
        assert metadata['array_type'] == 'staggered'

    def test_reproducibility_with_seed(self):
        """Test that patterns are reproducible with same random seed."""
        config = PatternConfig(image_size=64, pixel_size_nm=2.0)
        
        # Generate with seed
        np.random.seed(42)
        gen1 = GratingGenerator(config)
        gen1.rng = np.random.default_rng(42)
        pattern1, _ = gen1.generate(pitch_nm=40.0, add_ler=True)

        # Generate again with same seed
        np.random.seed(42)
        gen2 = GratingGenerator(config)
        gen2.rng = np.random.default_rng(42)
        pattern2, _ = gen2.generate(pitch_nm=40.0, add_ler=True)

        # Patterns should be identical
        assert np.allclose(pattern1, pattern2)
