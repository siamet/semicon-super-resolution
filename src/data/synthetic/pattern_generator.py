"""
Synthetic Pattern Generator for Semiconductor Structures

This module generates synthetic semiconductor patterns including:
- Line/space gratings (various pitches, duty cycles, orientations)
- Contact holes (circular and square arrays)
- Isolated features (lines, spaces, posts)
- Line edge roughness (LER) modeling

Author: Physics-Informed SR Research Project
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter, rotate
from scipy.interpolate import interp1d


@dataclass
class PatternConfig:
    """Configuration for pattern generation."""
    image_size: int = 512  # pixels
    pixel_size_nm: float = 2.0  # nanometers per pixel (HR image resolution)

    def __post_init__(self):
        """Calculate derived properties."""
        self.field_size_nm = self.image_size * self.pixel_size_nm


class PatternGenerator:
    """Base class for all pattern generators."""

    def __init__(self, config: PatternConfig):
        """
        Initialize pattern generator.

        Args:
            config: PatternConfig object with generation parameters
        """
        self.config = config
        self.rng = np.random.default_rng()

    def generate(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate a pattern. To be implemented by subclasses.

        Returns:
            pattern: 2D numpy array with values in [0, 1]
            metadata: Dictionary with generation parameters
        """
        raise NotImplementedError("Subclasses must implement generate()")

    def add_line_edge_roughness(
        self,
        pattern: np.ndarray,
        sigma_nm: float = 2.0,
        correlation_length_nm: float = 20.0
    ) -> np.ndarray:
        """
        Add line edge roughness (LER) to binary pattern.

        Uses a Gaussian random process with specified correlation length
        to simulate realistic line edge variations.

        Args:
            pattern: Binary pattern (0 or 1)
            sigma_nm: 1σ (standard deviation) LER in nanometers
                     Note: In semiconductor metrology, LER is often reported as 3σ,
                     so multiply by 3 if you have 3σ specifications
            correlation_length_nm: Spatial correlation length (typically 20-50nm)

        Returns:
            pattern_with_ler: Pattern with edge roughness added
        """
        # Validate inputs
        if sigma_nm < 0:
            raise ValueError(f"sigma_nm must be non-negative, got {sigma_nm}")
        if correlation_length_nm <= 0:
            raise ValueError(f"correlation_length_nm must be positive, got {correlation_length_nm}")

        # Convert to pixels
        sigma_pixels = sigma_nm / self.config.pixel_size_nm
        correlation_pixels = correlation_length_nm / self.config.pixel_size_nm

        # Find edges using gradient
        grad_y, grad_x = np.gradient(pattern.astype(float))
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edges = edge_magnitude > 0.1

        if not np.any(edges):
            return pattern

        # Generate correlated noise
        noise = self.rng.standard_normal(pattern.shape)
        # Correlation length controls blur sigma
        noise = gaussian_filter(noise, sigma=correlation_pixels)
        # Normalize to unit std, then scale to desired sigma (1σ)
        noise = noise / np.std(noise) * sigma_pixels

        # Apply noise only at edges
        pattern_float = pattern.astype(float)
        pattern_with_noise = pattern_float + noise * edges

        # Clip to [0, 1]
        pattern_with_noise = np.clip(pattern_with_noise, 0, 1)

        return pattern_with_noise

    def add_corner_rounding(
        self,
        pattern: np.ndarray,
        blur_sigma_nm: float = 5.0
    ) -> np.ndarray:
        """
        Add corner rounding to simulate optical proximity effects (OPE).

        Real lithography processes cause corners to round due to diffraction
        and resist chemistry effects.

        Args:
            pattern: Pattern array with values in [0, 1]
            blur_sigma_nm: Gaussian blur sigma in nanometers (typically 3-10nm)

        Returns:
            pattern_rounded: Pattern with rounded corners
        """
        if blur_sigma_nm <= 0:
            raise ValueError(f"blur_sigma_nm must be positive, got {blur_sigma_nm}")

        blur_sigma_pixels = blur_sigma_nm / self.config.pixel_size_nm
        pattern_rounded = gaussian_filter(pattern, sigma=blur_sigma_pixels)

        return pattern_rounded


class GratingGenerator(PatternGenerator):
    """Generator for line/space gratings."""

    def generate(
        self,
        pitch_nm: float = 100.0,
        duty_cycle: float = 0.5,
        orientation_deg: float = 0.0,
        add_ler: bool = True,
        ler_sigma_nm: float = 2.0,
        ler_correlation_nm: float = 20.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate a line/space grating pattern.

        Args:
            pitch_nm: Grating pitch (period) in nanometers (must be > 0)
            duty_cycle: Fraction of period occupied by lines (0 to 1)
            orientation_deg: Rotation angle in degrees (0=vertical lines)
            add_ler: Whether to add line edge roughness
            ler_sigma_nm: LER 1σ standard deviation in nanometers
            ler_correlation_nm: LER correlation length in nanometers

        Returns:
            pattern: 2D array with values in [0, 1], shape (image_size, image_size)
            metadata: Dictionary with generation parameters

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate inputs
        if pitch_nm <= 0:
            raise ValueError(f"pitch_nm must be positive, got {pitch_nm}")
        if not 0 <= duty_cycle <= 1:
            raise ValueError(f"duty_cycle must be in [0, 1], got {duty_cycle}")

        size = self.config.image_size
        pitch_pixels = pitch_nm / self.config.pixel_size_nm

        # Warn if pitch is too small (sub-pixel features)
        if pitch_pixels < 2.0:
            import warnings
            warnings.warn(f"Pitch ({pitch_nm:.1f}nm = {pitch_pixels:.1f}px) is very small. "
                         f"Consider using finer pixel_size_nm for better resolution.")

        # Create coordinate grid
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y)

        # Generate vertical grating (before rotation)
        # Periodic function with specified duty cycle
        phase = (X % pitch_pixels) / pitch_pixels
        pattern = (phase < duty_cycle).astype(float)

        # Add LER before rotation for realistic edge variation
        if add_ler:
            pattern = self.add_line_edge_roughness(
                pattern,
                sigma_nm=ler_sigma_nm,
                correlation_length_nm=ler_correlation_nm
            )

        # Rotate to desired orientation
        if orientation_deg != 0:
            pattern = rotate(pattern, orientation_deg, reshape=False, order=3)
            pattern = np.clip(pattern, 0, 1)

        metadata = {
            'pattern_type': 'grating',
            'pitch_nm': pitch_nm,
            'duty_cycle': duty_cycle,
            'orientation_deg': orientation_deg,
            'add_ler': add_ler,
            'ler_sigma_nm': ler_sigma_nm if add_ler else 0.0,
            'ler_correlation_nm': ler_correlation_nm if add_ler else 0.0,
            'image_size': size,
            'pixel_size_nm': self.config.pixel_size_nm
        }

        return pattern, metadata


class ContactHoleGenerator(PatternGenerator):
    """Generator for contact hole arrays."""

    def generate(
        self,
        diameter_nm: float = 50.0,
        pitch_nm: Optional[float] = None,
        shape: str = 'circular',
        array_type: str = 'regular',
        add_ler: bool = True,
        ler_sigma_nm: float = 2.0,
        ler_correlation_nm: float = 20.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate contact hole array pattern.

        Args:
            diameter_nm: Contact hole diameter/width in nanometers (must be > 0)
            pitch_nm: Array pitch (if None, uses 2× diameter)
            shape: 'circular' or 'square'
            array_type: 'regular' or 'staggered'
            add_ler: Whether to add line edge roughness
            ler_sigma_nm: LER 1σ standard deviation in nanometers
            ler_correlation_nm: LER correlation length in nanometers

        Returns:
            pattern: 2D array with values in [0, 1]
            metadata: Dictionary with generation parameters

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate inputs
        if diameter_nm <= 0:
            raise ValueError(f"diameter_nm must be positive, got {diameter_nm}")
        if shape not in ['circular', 'square']:
            raise ValueError(f"shape must be 'circular' or 'square', got '{shape}'")
        if array_type not in ['regular', 'staggered']:
            raise ValueError(f"array_type must be 'regular' or 'staggered', got '{array_type}'")

        size = self.config.image_size

        if pitch_nm is None:
            pitch_nm = diameter_nm * 2.0

        if pitch_nm <= 0:
            raise ValueError(f"pitch_nm must be positive, got {pitch_nm}")

        # Warn if holes will overlap
        if pitch_nm < diameter_nm:
            import warnings
            warnings.warn(f"Pitch ({pitch_nm:.1f}nm) < diameter ({diameter_nm:.1f}nm). "
                         f"Contact holes will overlap!")

        diameter_pixels = diameter_nm / self.config.pixel_size_nm
        pitch_pixels = pitch_nm / self.config.pixel_size_nm
        radius_pixels = diameter_pixels / 2.0

        # Create coordinate grid
        y, x = np.ogrid[:size, :size]
        pattern = np.zeros((size, size), dtype=float)

        # Generate array positions
        n_contacts = int(size / pitch_pixels) + 2

        for i in range(n_contacts):
            for j in range(n_contacts):
                # Center position
                cx = j * pitch_pixels
                cy = i * pitch_pixels

                # Staggered offset for odd rows
                if array_type == 'staggered' and i % 2 == 1:
                    cx += pitch_pixels / 2.0

                # Skip if outside image
                if cx < -pitch_pixels or cx > size + pitch_pixels:
                    continue
                if cy < -pitch_pixels or cy > size + pitch_pixels:
                    continue

                # Generate contact hole
                if shape == 'circular':
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    contact = (dist <= radius_pixels).astype(float)
                elif shape == 'square':
                    contact = ((np.abs(x - cx) <= radius_pixels) &
                              (np.abs(y - cy) <= radius_pixels)).astype(float)
                else:
                    raise ValueError(f"Unknown shape: {shape}")

                pattern = np.maximum(pattern, contact)

        # Add LER
        if add_ler:
            pattern = self.add_line_edge_roughness(
                pattern,
                sigma_nm=ler_sigma_nm,
                correlation_length_nm=ler_correlation_nm
            )

        metadata = {
            'pattern_type': 'contact_holes',
            'diameter_nm': diameter_nm,
            'pitch_nm': pitch_nm,
            'shape': shape,
            'array_type': array_type,
            'add_ler': add_ler,
            'ler_sigma_nm': ler_sigma_nm if add_ler else 0.0,
            'ler_correlation_nm': ler_correlation_nm if add_ler else 0.0,
            'image_size': size,
            'pixel_size_nm': self.config.pixel_size_nm
        }

        return pattern, metadata


class IsolatedFeatureGenerator(PatternGenerator):
    """Generator for isolated features (lines, spaces, posts)."""

    def generate(
        self,
        feature_type: str = 'line',
        width_nm: float = 50.0,
        length_nm: Optional[float] = None,
        orientation_deg: float = 0.0,
        add_ler: bool = True,
        ler_sigma_nm: float = 2.0,
        ler_correlation_nm: float = 20.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate isolated feature pattern.

        Args:
            feature_type: 'line', 'space', or 'post' (must be one of these)
            width_nm: Feature width in nanometers (must be > 0)
            length_nm: Feature length (if None, uses 70% of field size for lines)
            orientation_deg: Rotation angle in degrees
            add_ler: Whether to add line edge roughness
            ler_sigma_nm: LER 1σ standard deviation in nanometers
            ler_correlation_nm: LER correlation length in nanometers

        Returns:
            pattern: 2D array with values in [0, 1]
            metadata: Dictionary with generation parameters

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate inputs
        if feature_type not in ['line', 'space', 'post']:
            raise ValueError(f"feature_type must be 'line', 'space', or 'post', got '{feature_type}'")
        if width_nm <= 0:
            raise ValueError(f"width_nm must be positive, got {width_nm}")

        size = self.config.image_size

        if length_nm is None:
            length_nm = self.config.field_size_nm * 0.7

        if length_nm <= 0:
            raise ValueError(f"length_nm must be positive, got {length_nm}")

        width_pixels = width_nm / self.config.pixel_size_nm
        length_pixels = length_nm / self.config.pixel_size_nm

        # Start with background
        if feature_type == 'space':
            pattern = np.ones((size, size), dtype=float)
        else:
            pattern = np.zeros((size, size), dtype=float)

        # Create feature centered in image
        cy, cx = size // 2, size // 2

        if feature_type in ['line', 'space']:
            # Create rectangular feature
            half_width = width_pixels / 2.0
            half_length = length_pixels / 2.0

            y, x = np.ogrid[:size, :size]
            feature_mask = ((np.abs(x - cx) <= half_width) &
                           (np.abs(y - cy) <= half_length))

        elif feature_type == 'post':
            # Create square post
            half_width = width_pixels / 2.0

            y, x = np.ogrid[:size, :size]
            feature_mask = ((np.abs(x - cx) <= half_width) &
                           (np.abs(y - cy) <= half_width))
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        # Apply feature
        if feature_type == 'space':
            pattern[feature_mask] = 0.0  # Cut out space
        else:
            pattern[feature_mask] = 1.0  # Add feature

        # Add LER before rotation
        if add_ler:
            pattern = self.add_line_edge_roughness(
                pattern,
                sigma_nm=ler_sigma_nm,
                correlation_length_nm=ler_correlation_nm
            )

        # Rotate if needed
        if orientation_deg != 0:
            pattern = rotate(pattern, orientation_deg, reshape=False, order=3)
            pattern = np.clip(pattern, 0, 1)

        metadata = {
            'pattern_type': 'isolated_feature',
            'feature_type': feature_type,
            'width_nm': width_nm,
            'length_nm': length_nm,
            'orientation_deg': orientation_deg,
            'add_ler': add_ler,
            'ler_sigma_nm': ler_sigma_nm if add_ler else 0.0,
            'ler_correlation_nm': ler_correlation_nm if add_ler else 0.0,
            'image_size': size,
            'pixel_size_nm': self.config.pixel_size_nm
        }

        return pattern, metadata


def create_pattern_generator(
    pattern_type: str,
    config: Optional[PatternConfig] = None
) -> PatternGenerator:
    """
    Factory function to create appropriate pattern generator.

    Args:
        pattern_type: Type of pattern ('grating', 'contacts', 'isolated')
        config: Optional PatternConfig (uses defaults if None)

    Returns:
        generator: Appropriate PatternGenerator subclass instance
    """
    if config is None:
        config = PatternConfig()

    generators = {
        'grating': GratingGenerator,
        'contacts': ContactHoleGenerator,
        'isolated': IsolatedFeatureGenerator
    }

    if pattern_type not in generators:
        raise ValueError(f"Unknown pattern_type: {pattern_type}. "
                        f"Must be one of {list(generators.keys())}")

    return generators[pattern_type](config)
