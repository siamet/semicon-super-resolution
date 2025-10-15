"""
Pattern Visualization Tools

Tools for visualizing and validating generated semiconductor patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path


class PatternVisualizer:
    """Visualizer for synthetic semiconductor patterns."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize

    def visualize_pattern(
        self,
        pattern: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
        show: bool = True,
        show_physical_scale: bool = True
    ) -> None:
        """
        Visualize a single pattern.

        Args:
            pattern: 2D pattern array
            metadata: Optional metadata dictionary
            title: Optional plot title
            save_path: Optional path to save figure
            show: Whether to display the figure
            show_physical_scale: If True and metadata available, show axes in nm
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        # Determine extent for physical scale
        extent = None
        xlabel = 'X (pixels)'
        ylabel = 'Y (pixels)'

        if show_physical_scale and metadata and 'pixel_size_nm' in metadata:
            pixel_size = metadata['pixel_size_nm']
            image_size = metadata.get('image_size', pattern.shape[0])
            field_size = image_size * pixel_size
            extent = [0, field_size, 0, field_size]
            xlabel = 'X (nm)'
            ylabel = 'Y (nm)'

        im = ax.imshow(pattern, cmap='gray', origin='lower', extent=extent)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title)
        elif metadata:
            title = self._generate_title_from_metadata(metadata)
            ax.set_title(title)

        plt.colorbar(im, ax=ax, label='Intensity')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def visualize_pattern_with_profile(
        self,
        pattern: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
        show: bool = True,
        show_physical_scale: bool = True
    ) -> None:
        """
        Visualize pattern with horizontal and vertical line profiles.

        Args:
            pattern: 2D pattern array
            metadata: Optional metadata dictionary
            title: Optional plot title
            save_path: Optional path to save figure
            show: Whether to display the figure
            show_physical_scale: If True and metadata available, show axes in nm
        """
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                             hspace=0.05, wspace=0.05)

        # Determine extent and axis labels
        extent = None
        xlabel = 'X (pixels)'
        ylabel = 'Y (pixels)'
        x_coords = np.arange(pattern.shape[1])
        y_coords = np.arange(pattern.shape[0])

        if show_physical_scale and metadata and 'pixel_size_nm' in metadata:
            pixel_size = metadata['pixel_size_nm']
            image_size = metadata.get('image_size', pattern.shape[0])
            field_size = image_size * pixel_size
            extent = [0, field_size, 0, field_size]
            xlabel = 'X (nm)'
            ylabel = 'Y (nm)'
            x_coords = np.linspace(0, field_size, pattern.shape[1])
            y_coords = np.linspace(0, field_size, pattern.shape[0])

        # Main pattern image
        ax_main = fig.add_subplot(gs[1, 0])
        im = ax_main.imshow(pattern, cmap='gray', origin='lower', extent=extent)
        ax_main.set_xlabel(xlabel)
        ax_main.set_ylabel(ylabel)

        # Horizontal profile (top)
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        mid_row = pattern.shape[0] // 2
        ax_top.plot(x_coords, pattern[mid_row, :], 'b-', linewidth=1)
        ax_top.set_ylabel('Intensity')
        ax_top.set_title(title or self._generate_title_from_metadata(metadata))
        ax_top.grid(True, alpha=0.3)
        plt.setp(ax_top.get_xticklabels(), visible=False)

        # Vertical profile (right)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        mid_col = pattern.shape[1] // 2
        ax_right.plot(pattern[:, mid_col], y_coords, 'r-', linewidth=1)
        ax_right.set_xlabel('Intensity')
        ax_right.grid(True, alpha=0.3)
        plt.setp(ax_right.get_yticklabels(), visible=False)

        # Colorbar in top-right
        ax_cbar = fig.add_subplot(gs[0, 1])
        plt.colorbar(im, cax=ax_cbar)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def visualize_multiple_patterns(
        self,
        patterns: List[np.ndarray],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        titles: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        show: bool = True,
        show_physical_scale: bool = True
    ) -> None:
        """
        Visualize multiple patterns in a grid.

        Args:
            patterns: List of 2D pattern arrays
            metadatas: Optional list of metadata dictionaries
            titles: Optional list of titles
            save_path: Optional path to save figure
            show: Whether to display the figure
            show_physical_scale: If True and metadata available, show axes in nm
        """
        n_patterns = len(patterns)
        n_cols = min(4, n_patterns)
        n_rows = (n_patterns + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_patterns == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (pattern, ax) in enumerate(zip(patterns, axes)):
            # Determine extent and labels
            extent = None
            xlabel = 'X (pixels)'
            ylabel = 'Y (pixels)'

            if show_physical_scale and metadatas and idx < len(metadatas):
                metadata = metadatas[idx]
                if 'pixel_size_nm' in metadata:
                    pixel_size = metadata['pixel_size_nm']
                    image_size = metadata.get('image_size', pattern.shape[0])
                    field_size = image_size * pixel_size
                    extent = [0, field_size, 0, field_size]
                    xlabel = 'X (nm)'
                    ylabel = 'Y (nm)'

            im = ax.imshow(pattern, cmap='gray', origin='lower', extent=extent)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)

            # Set title
            if titles and idx < len(titles):
                ax.set_title(titles[idx])
            elif metadatas and idx < len(metadatas):
                title = self._generate_title_from_metadata(metadatas[idx])
                ax.set_title(title, fontsize=9)

            plt.colorbar(im, ax=ax, fraction=0.046)

        # Hide unused subplots
        for idx in range(n_patterns, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def visualize_statistics(
        self,
        pattern: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> None:
        """
        Visualize pattern statistics (histogram, FFT, etc.).

        Args:
            pattern: 2D pattern array
            metadata: Optional metadata dictionary
            save_path: Optional path to save figure
            show: Whether to display the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Pattern
        ax = axes[0, 0]
        im = ax.imshow(pattern, cmap='gray', origin='lower')
        ax.set_title('Pattern')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax)

        # Histogram
        ax = axes[0, 1]
        ax.hist(pattern.flatten(), bins=50, color='blue', alpha=0.7)
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Count')
        ax.set_title('Intensity Histogram')
        ax.grid(True, alpha=0.3)
        ax.axvline(pattern.mean(), color='r', linestyle='--', label=f'Mean: {pattern.mean():.3f}')
        ax.legend()

        # Power Spectral Density (FFT)
        ax = axes[1, 0]
        fft = np.fft.fftshift(np.fft.fft2(pattern))
        psd = np.abs(fft)**2
        psd_log = np.log10(psd + 1)  # Log scale with offset to avoid log(0)
        im = ax.imshow(psd_log, cmap='viridis', origin='lower')
        ax.set_title('Power Spectral Density (log scale)')
        ax.set_xlabel('Frequency X')
        ax.set_ylabel('Frequency Y')
        plt.colorbar(im, ax=ax, label='log10(PSD)')

        # Radial PSD profile
        ax = axes[1, 1]
        center = np.array(psd.shape) // 2
        y, x = np.ogrid[:psd.shape[0], :psd.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)

        max_radius = min(center)
        radial_profile = np.zeros(max_radius)
        counts = np.zeros(max_radius)

        for radius in range(max_radius):
            mask = (r == radius)
            if np.any(mask):
                radial_profile[radius] = psd[mask].mean()
                counts[radius] = mask.sum()

        # Only plot where we have data
        valid = counts > 0
        ax.plot(np.arange(max_radius)[valid], radial_profile[valid], 'b-', linewidth=2)
        ax.set_xlabel('Spatial Frequency (cycles/image)')
        ax.set_ylabel('Power')
        ax.set_title('Radial Power Spectrum')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        if metadata:
            fig.suptitle(self._generate_title_from_metadata(metadata), fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def _generate_title_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Generate informative title from metadata.

        Args:
            metadata: Metadata dictionary

        Returns:
            title: Formatted title string
        """
        pattern_type = metadata.get('pattern_type', 'Unknown')

        if pattern_type == 'grating':
            return (f"Grating: pitch={metadata.get('pitch_nm', 0):.1f}nm, "
                   f"DC={metadata.get('duty_cycle', 0):.2f}, "
                   f"θ={metadata.get('orientation_deg', 0):.0f}°")

        elif pattern_type == 'contact_holes':
            return (f"Contacts: {metadata.get('shape', 'N/A')}, "
                   f"D={metadata.get('diameter_nm', 0):.1f}nm, "
                   f"pitch={metadata.get('pitch_nm', 0):.1f}nm, "
                   f"{metadata.get('array_type', 'N/A')}")

        elif pattern_type == 'isolated_feature':
            return (f"Isolated {metadata.get('feature_type', 'N/A')}: "
                   f"W={metadata.get('width_nm', 0):.1f}nm, "
                   f"L={metadata.get('length_nm', 0):.1f}nm")

        else:
            return pattern_type


def quick_visualize(
    pattern: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    mode: str = 'simple'
) -> None:
    """
    Quick visualization helper function.

    Args:
        pattern: 2D pattern array
        metadata: Optional metadata dictionary
        mode: Visualization mode ('simple', 'profile', 'stats')
    """
    visualizer = PatternVisualizer()

    if mode == 'simple':
        visualizer.visualize_pattern(pattern, metadata)
    elif mode == 'profile':
        visualizer.visualize_pattern_with_profile(pattern, metadata)
    elif mode == 'stats':
        visualizer.visualize_statistics(pattern, metadata)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'simple', 'profile', or 'stats'.")
