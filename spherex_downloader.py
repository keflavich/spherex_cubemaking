#!/usr/bin/env python3
"""
SPHEREX Data Downloader and Processor

This script downloads SPHEREX images from the IRSA archive using astroquery,
crops them to a common frame, reprojects to a common WCS, and assembles
them into a data cube with wavelength information.

TODO: Build the output cube before adding data in, then interpolate onto its grid.
We should use spectral resolution of 100 and go from 2-5 microns (ignore short wavelengths),
so the cube should have 0.1 micron steps from 2 to 5 microns (150-ish channels)


Author: AI Assistant
Date: September 2025
"""

import os
import glob
import sys
import numpy as np
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import argparse
import urllib.request
import urllib.error

# Astronomy libraries
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy import units as u
from astropy.stats import sigma_clipped_stats
import astropy.constants as const

# Astroquery for IRSA access
from astroquery.ipac.irsa import Irsa

# Reproject for image alignment
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs

# Optional plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting functions disabled.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class SPHEREXDownloader:
    """
    A class to handle SPHEREX data download and processing from IRSA.
    """

    def __init__(self, output_dir: str = "spherex_data", verbose: bool = True):
        """
        Initialize the SPHEREX downloader.

        Parameters
        ----------
        output_dir : str
            Directory to save downloaded and processed data
        verbose : bool
            Enable verbose output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.verbose = verbose
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        self.processed_dir = self.output_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)

        # SPHEREX wavelength information (approximate)
        # SPHEREX covers 0.75-5.0 microns with ~100 spectral channels
        #self.spherex_wavelengths = np.logspace(np.log10(0.75), np.log10(5.0), 52) * u.micron

        # Standard output wavelength grid: 2-5 microns with R=100 (0.1 micron steps)
        self.output_wavelength_min = 0.7  # microns
        self.output_wavelength_max = 5.2  # microns
        self.output_wavelength_step = 0.1  # microns (gives R~50 at 2 microns)
        self.output_wavelengths = np.arange(
            self.output_wavelength_min,
            self.output_wavelength_max + self.output_wavelength_step,
            self.output_wavelength_step
        )

        if self.verbose:
            print(f"SPHEREX Downloader initialized")
            print(f"Output directory: {self.output_dir}")
            print(f"Images directory: {self.images_dir}")
            print(f"Processed directory: {self.processed_dir}")
            print(f"Output wavelength grid: {self.output_wavelength_min}-{self.output_wavelength_max} μm")
            print(f"Spectral channels: {len(self.output_wavelengths)} ({self.output_wavelength_step:.3f} μm steps)")

    def search_spherex_images(self,
                             target: str = None,
                             coordinates: SkyCoord = None,
                             radius: u.Quantity = 10*u.arcmin) -> Table:
        """
        Search for SPHEREX images in IRSA archive.

        Parameters
        ----------
        target : str, optional
            Target name (e.g., 'M31', 'NGC 1234')
        coordinates : SkyCoord, optional
            Target coordinates
        radius : Quantity
            Search radius

        Returns
        -------
        Table
            Table of available SPHEREX images
        """
        if self.verbose:
            print(f"Searching for SPHEREX images...")

        # First, let's check what SPHEREX collections are available
        collections = Irsa.list_collections(servicetype='SIA', filter='spherex')
        if self.verbose:
            print(f"Found {len(collections)} SPHEREX collections:")
            for collection in collections['collection']:
                print(f"  - {collection}")

        # Determine coordinates
        if coordinates is None and target is not None:
            coordinates = SkyCoord.from_name(target)
        elif coordinates is None:
            raise ValueError("Either target name or coordinates must be provided")

        if self.verbose:
            print(f"Target coordinates: {coordinates.to_string('hmsdms')}")
            print(f"Search radius: {radius}")

        # Search for images using Simple Image Access
        try:
            # Try searching for SPHEREX data specifically
            images = Irsa.query_sia(pos=(coordinates, radius), collection='spherex')

            if len(images) == 0:
                # Fallback: search all collections
                if self.verbose:
                    print("No SPHEREX-specific images found. Searching all collections...")
                images = Irsa.query_sia(pos=(coordinates, radius))

                # Filter for potential SPHEREX data
                if len(images) > 0:
                    spherex_mask = np.array([
                        'spherex' in str(row.get('collection', '')).lower() or
                        'spherex' in str(row.get('obs_collection', '')).lower() or
                        'spherex' in str(row.get('dataid_collection', '')).lower()
                        for row in images
                    ])
                    images = images[spherex_mask] if np.any(spherex_mask) else images[:0]

        except Exception as e:
            if self.verbose:
                print(f"SIA query failed: {e}")
                print("Attempting catalog search as fallback...")

            # Fallback to catalog search
            try:
                catalogs = Irsa.list_catalogs(filter='spherex')
                if len(catalogs) > 0:
                    catalog_name = list(catalogs.keys())[0]
                    images = Irsa.query_region(
                        coordinates=coordinates,
                        spatial='Cone',
                        catalog=catalog_name,
                        radius=radius
                    )
                else:
                    images = Table()
            except Exception as e2:
                if self.verbose:
                    print(f"Catalog search also failed: {e2}")
                images = Table()

        if self.verbose:
            print(f"Found {len(images)} images")
            if len(images) > 0:
                print("Available columns:", images.colnames[:10])  # Show first 10 columns

        return images

    def download_images(self, images: Table, max_images: int = None) -> List[str]:
        """
        Download SPHEREX images from IRSA.

        Parameters
        ----------
        images : Table
            Table of images from search_spherex_images
        max_images : int, optional
            Maximum number of images to download

        Returns
        -------
        List[str]
            List of downloaded file paths
        """
        if len(images) == 0:
            if self.verbose:
                print("No images to download")
            return []

        if max_images is not None:
            images = images[:max_images]

        downloaded_files = []

        if self.verbose:
            print(f"Downloading {len(images)} images...")
            if len(images) > 10:
                print("  (This may take a while for large datasets...)")

        for i, image in enumerate(images):
            try:
                # Determine the access URL
                url = None
                for url_col in ['access_url', 'cloud_access', 'download_url', 'url']:
                    if url_col in image.colnames and image[url_col]:
                        url = str(image[url_col])
                        break

                if url is None:
                    if self.verbose:
                        print(f"  Skipping image {i+1}: No access URL found")
                    continue

                # Generate filename
                obs_id = image.get('obs_id', f'spherex_{i:04d}')
                filename = f"{obs_id}.fits"
                filepath = self.images_dir / filename

                if filepath.exists():
                    if self.verbose:
                        print(f"  Image {i+1}/{len(images)}: {filename} already exists")
                    downloaded_files.append(str(filepath))
                    continue

                # Download the file
                if self.verbose:
                    print(f"  Downloading image {i+1}/{len(images)}: {filename}")

                # Download using urllib first, then open with astropy
                try:
                    # Download to temporary location first
                    temp_filepath = filepath.with_suffix('.tmp')
                    urllib.request.urlretrieve(url, temp_filepath)

                    # Verify it's a valid FITS file by opening it
                    with fits.open(temp_filepath) as hdul:
                        # Save as final file
                        hdul.writeto(filepath, overwrite=True)

                    # Remove temporary file
                    temp_filepath.unlink()

                except urllib.error.URLError as e:
                    if self.verbose:
                        print(f"    URL error: {e}")
                    continue
                except Exception as fits_error:
                    # Clean up temp file if it exists
                    temp_filepath = filepath.with_suffix('.tmp')
                    if temp_filepath.exists():
                        temp_filepath.unlink()
                    raise fits_error

                downloaded_files.append(str(filepath))

            except Exception as e:
                if self.verbose:
                    print(f"  Failed to download image {i+1}: {e}")
                continue

        if self.verbose:
            print(f"Successfully downloaded {len(downloaded_files)} images")

        return downloaded_files

    def determine_common_frame(self, image_files: List[str],
                              target_coords: SkyCoord = None,
                              frame_size: u.Quantity = 5*u.arcmin) -> Tuple[WCS, Tuple[int, int]]:
        """
        Determine a common WCS frame for all images.

        Parameters
        ----------
        image_files : List[str]
            List of FITS image file paths
        target_coords : SkyCoord, optional
            Central coordinates for the frame
        frame_size : Quantity
            Size of the common frame

        Returns
        -------
        Tuple[WCS, Tuple[int, int]]
            Common WCS and image shape (ny, nx)
        """
        if self.verbose:
            print("Determining common frame...")

        # Collect WCS from all images
        wcs_list = []
        shapes = []

        for filepath in image_files:
            with fits.open(filepath) as hdul:
                # Find the primary data extension
                data_hdu = None
                for hdu in hdul:
                    if hdu.data is not None and hdu.data.size > 1:
                        data_hdu = hdu
                        break

                if data_hdu is not None:
                    wcs = WCS(data_hdu.header)
                    wcs_list.append(wcs)
                    shapes.append(data_hdu.data.shape)

        if len(wcs_list) == 0:
            raise ValueError("No valid WCS found in any image")

        # Use reproject to find optimal common WCS
        try:
            common_wcs, shape = find_optimal_celestial_wcs(
                [(np.ones(shape), wcs) for wcs, shape in zip(wcs_list, shapes)],
                frame='icrs'
            )
        except Exception as e:
            if self.verbose:
                print(f"  Warning: find_optimal_celestial_wcs failed: {e}")
                print("  Using first image WCS as reference")
            raise
            common_wcs = wcs_list[0]
            shape = shapes[0]

        # If target coordinates are provided, center the frame there
        if target_coords is not None:
            # Create a new WCS centered on target
            pixel_scale = np.abs(common_wcs.wcs.cdelt[0]) * u.deg

            # Calculate image dimensions
            frame_size_deg = frame_size.to(u.deg).value
            nx = ny = int(frame_size_deg / pixel_scale.to(u.deg).value)

            # Create new WCS
            new_wcs = WCS(naxis=2)
            new_wcs.wcs.crpix = [nx/2, ny/2]
            new_wcs.wcs.crval = [target_coords.ra.deg, target_coords.dec.deg]
            new_wcs.wcs.cdelt = [-pixel_scale.to(u.deg).value, pixel_scale.to(u.deg).value]
            new_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']

            common_wcs = new_wcs
            shape = (ny, nx)

        if self.verbose:
            print(f"  Common frame: {shape[1]} x {shape[0]} pixels")
            print(f"  Pixel scale: {np.abs(common_wcs.wcs.cdelt[0]*3600):.2f} arcsec/pixel")
            if hasattr(common_wcs.wcs, 'crval'):
                print(f"  Center: RA={common_wcs.wcs.crval[0]:.6f}, Dec={common_wcs.wcs.crval[1]:.6f}")

        return common_wcs, shape

    def reproject_images(self, image_files: List[str],
                        common_wcs: WCS,
                        output_shape: Tuple[int, int]) -> List[str]:
        """
        Reproject all images to the common WCS frame.

        Parameters
        ----------
        image_files : List[str]
            List of input FITS files
        common_wcs : WCS
            Target WCS for reprojection
        output_shape : Tuple[int, int]
            Output image shape (ny, nx)

        Returns
        -------
        List[str]
            List of reprojected image file paths
        """
        if self.verbose:
            print(f"Reprojecting {len(image_files)} images to common frame...")

        reprojected_files = []

        for i, filepath in enumerate(image_files):
            try:
                filename = Path(filepath).stem
                output_file = self.processed_dir / f"{filename}_reprojected.fits"

                if output_file.exists():
                    if self.verbose:
                        print(f"  Image {i+1}/{len(image_files)}: {output_file.name} already exists")
                    reprojected_files.append(str(output_file))
                    continue

                if self.verbose:
                    print(f"  Reprojecting image {i+1}/{len(image_files)}: {Path(filepath).name}")

                with fits.open(filepath) as hdul:
                    # Find the primary data extension
                    data_hdu = None
                    for hdu in hdul:
                        if hdu.data is not None and hdu.data.size > 1:
                            data_hdu = hdu
                            break

                    if data_hdu is None:
                        if self.verbose:
                            print(f"    Warning: No data found in {filepath}")
                        continue

                    wcs_in = WCS(data_hdu.header)

                    # Reproject the image
                    reprojected_data, footprint = reproject_interp(
                        (data_hdu.data, wcs_in),
                        common_wcs,
                        output_shape
                    )

                    #header = data_hdu.header.copy()
                    #header.update(common_wcs.to_header())
                    header = common_wcs.to_header()

                    # Create new HDU with reprojected data
                    new_hdu = fits.PrimaryHDU(data=reprojected_data, header=header)

                    spectral_wcs = WCS(data_hdu.header, fobj=hdul, key="W")
                    spectral_wcs.sip = None
                    sh = reprojected_data.shape
                    xx, yy = common_wcs.world_to_pixel(wcs_in.pixel_to_world(*np.mgrid[:sh[0], :sh[1]]))
                    ww, bandwidth = spectral_wcs.pixel_to_world(xx, yy)
                    wave_header = fits.Header({'unit': spectral_wcs.wcs.cunit[0].to_string()})
                    bw_header = fits.Header({'unit': spectral_wcs.wcs.cunit[1].to_string()})
                    wave_header.update(header)
                    bw_header.update(header)

                    new_hdul = fits.HDUList([new_hdu,
                                             fits.ImageHDU(ww.value, header=wave_header, name='WAVELENGTH'),
                                             fits.ImageHDU(bandwidth.value, header=bw_header, name='BANDWIDTH'),
                                             ]
                                             )


                    # Copy relevant metadata
                    for key in ['OBJECT', 'FILTER', 'EXPTIME', 'DATE-OBS', 'TELESCOP', 'INSTRUME']:
                        if key in data_hdu.header:
                            new_hdu.header[key] = data_hdu.header[key]

                    # Add processing information
                    new_hdu.header['REPROJ'] = True
                    new_hdu.header['COMMENT'] = 'Reprojected to common WCS frame'

                    # Save reprojected image
                    new_hdul.writeto(output_file, overwrite=True)
                    reprojected_files.append(str(output_file))

            except Exception as e:
                if self.verbose:
                    print(f"    Error reprojecting {filepath}: {e}")
                raise
                continue

        if self.verbose:
            print(f"Successfully reprojected {len(reprojected_files)} images")

        return reprojected_files

    def assign_wavelengths(self, image_files: List[str]) -> Dict[str, float]:
        """
        Assign wavelengths to images based on available metadata or default scheme.

        Parameters
        ----------
        image_files : List[str]
            List of image file paths

        Returns
        -------
        Dict[str, float]
            Mapping from file path to wavelength in microns
        """
        if self.verbose:
            print("Assigning wavelengths to images...")

        wavelength_map = {}

        # Try to extract wavelength information from headers
        wavelengths_found = []

        for filepath in image_files:
            with fits.open(filepath) as hdul:
                header = hdul[0].header
                wavelengths = hdul['WAVELENGTH'].data
                wavelengths_found = np.nanmean(wavelengths)

        if self.verbose:
            print(f"  Wavelength assignments:")
            for filepath, wavelength in sorted(wavelength_map.items(), key=lambda x: x[1]):
                print(f"    {Path(filepath).name}: {wavelength:.3f} μm")

        return wavelength_map

    def _get_wavelength_map(self, header: fits.Header, shape: tuple,
                           filepath: str, wavelength_map: Dict[str, float]) -> np.ndarray:
        """
        Extract or model the 2D wavelength map for an image.

        This method tries multiple approaches to determine the wavelength at each pixel:
        1. Extract from WCS if 3D cube with wavelength axis
        2. Use wavelength gradient keywords in header
        3. Model linear gradient based on SPHEREX disperser geometry
        4. Fall back to constant wavelength across image

        Parameters
        ----------
        header : fits.Header
            FITS header of the image
        shape : tuple
            Shape of the image (ny, nx)
        filepath : str
            Path to the image file
        wavelength_map : Dict[str, float]
            Global wavelength mapping for fallback

        Returns
        -------
        np.ndarray or None
            2D array of wavelengths at each pixel, or None if failed
        """
        ny, nx = shape

        # Method 1: Check if this is a 3D cube with wavelength information
        if header.get('NAXIS', 0) == 3 and header.get('NAXIS3', 0) > 1:
            try:
                # This might be a spectral cube - try to extract wavelength axis
                wcs_3d = WCS(header)
                if wcs_3d.wcs.ctype[2] in ['WAVE', 'FREQ', 'VELO']:
                    # Get wavelength for each slice and see if we can map to spatial position
                    if self.verbose:
                        print(f"      Found 3D cube with spectral axis: {wcs_3d.wcs.ctype[2]}")
                    # For now, use the center wavelength - could be improved
                    center_wl = wavelength_map.get(filepath, 3.0)  # fallback to 3 microns
                    return np.full((ny, nx), center_wl, dtype=np.float32)
            except Exception:
                pass

        # Method 2: Look for wavelength gradient keywords in header
        gradient_keywords = [
            'WAVGRAD', 'DWDX', 'DWDY', 'LAMBGRAD', 'WLGRAD',
            'WAVESLOPE', 'DISPAXIS', 'CDELT3'
        ]

        for keyword in gradient_keywords:
            if keyword in header:
                try:
                    gradient_value = float(header[keyword])
                    if abs(gradient_value) > 1e-10:  # Non-zero gradient
                        if self.verbose:
                            print(f"      Found wavelength gradient keyword {keyword}: {gradient_value}")
                        # Create simple linear gradient
                        return self._create_linear_gradient(shape, wavelength_map[filepath], gradient_value)
                except (ValueError, TypeError):
                    continue

        # Method 3: Model SPHEREX-like disperser gradient
        # SPHEREX has a linear variable filter that creates a wavelength gradient
        # The gradient direction and magnitude depend on the specific observation
        base_wavelength = wavelength_map.get(filepath, 3.0)

        # Try to infer gradient from filename or header information
        gradient_info = self._infer_spherex_gradient(header, filepath, base_wavelength)

        if gradient_info is not None:
            gradient_magnitude, gradient_angle = gradient_info
            if self.verbose:
                print(f"      Modeling SPHEREX gradient: {gradient_magnitude:.4f} μm/pixel at {gradient_angle:.1f}°")
            return self._create_spatial_gradient(shape, base_wavelength, gradient_magnitude, gradient_angle)

        # Method 4: Fallback to constant wavelength with small random variation
        # This ensures each pixel contributes to 2 channels as requested
        if self.verbose:
            print(f"      Using constant wavelength with small variation: {base_wavelength:.3f} μm")

        # Add small spatial variation to ensure 2-channel contribution
        wavelength_map_2d = np.full((ny, nx), base_wavelength, dtype=np.float32)

        # Add small gradient (equivalent to ~0.01 micron across the image)
        y_coords, x_coords = np.mgrid[0:ny, 0:nx]
        y_coords = y_coords.astype(np.float32) / ny  # normalize to 0-1
        x_coords = x_coords.astype(np.float32) / nx  # normalize to 0-1

        # Small wavelength variation: ±0.02 microns across image
        variation = 0.02 * (x_coords - 0.5 + 0.5 * (y_coords - 0.5))
        wavelength_map_2d += variation

        return wavelength_map_2d

    def _create_linear_gradient(self, shape: tuple, base_wavelength: float,
                               gradient: float) -> np.ndarray:
        """Create a linear wavelength gradient across the image."""
        ny, nx = shape
        wavelength_map = np.zeros((ny, nx), dtype=np.float32)

        # Create gradient along x-axis (could be modified based on header info)
        for x in range(nx):
            wavelength_map[:, x] = base_wavelength + gradient * (x - nx/2) / nx

        return wavelength_map

    def _infer_spherex_gradient(self, header: fits.Header, filepath: str,
                               base_wavelength: float) -> tuple:
        """
        Infer SPHEREX wavelength gradient from header or filename.

        Returns
        -------
        tuple or None
            (gradient_magnitude, gradient_angle) or None if cannot infer
        """
        # Look for SPHEREX-specific keywords
        spherex_keywords = ['FILTER', 'GRISM', 'GRATING', 'DISPERSER']

        for keyword in spherex_keywords:
            if keyword in header:
                filter_info = str(header[keyword]).upper()
                # Different SPHEREX configurations have different gradients
                if 'LVF' in filter_info or 'LINEAR' in filter_info:
                    # Linear Variable Filter - strong gradient
                    gradient_magnitude = 0.05  # microns per 100 pixels
                    gradient_angle = 0.0  # along x-axis
                    return (gradient_magnitude, gradient_angle)
                elif 'CVF' in filter_info or 'CIRCULAR' in filter_info:
                    # Circular Variable Filter - radial gradient
                    gradient_magnitude = 0.03
                    gradient_angle = 45.0  # diagonal
                    return (gradient_magnitude, gradient_angle)

        # Infer from filename patterns
        filename = Path(filepath).name.upper()
        if 'LVF' in filename or 'LINEAR' in filename:
            return (0.05, 0.0)
        elif 'CVF' in filename or 'CIRCULAR' in filename:
            return (0.03, 45.0)

        # Default SPHEREX-like gradient based on wavelength
        # Shorter wavelengths typically have steeper gradients
        if base_wavelength < 2.5:
            gradient_magnitude = 0.04  # steeper at shorter wavelengths
        elif base_wavelength < 4.0:
            gradient_magnitude = 0.03  # moderate at mid-wavelengths
        else:
            gradient_magnitude = 0.02  # gentler at longer wavelengths

        # Default gradient angle (could be randomized or based on other info)
        gradient_angle = 0.0  # along x-axis

        return (gradient_magnitude, gradient_angle)

    def _create_spatial_gradient(self, shape: tuple, base_wavelength: float,
                                gradient_magnitude: float, gradient_angle: float) -> np.ndarray:
        """
        Create a 2D wavelength gradient with specified magnitude and direction.

        Parameters
        ----------
        shape : tuple
            Image shape (ny, nx)
        base_wavelength : float
            Central wavelength in microns
        gradient_magnitude : float
            Gradient strength in microns per 100 pixels
        gradient_angle : float
            Gradient direction in degrees (0 = along x-axis)

        Returns
        -------
        np.ndarray
            2D wavelength map
        """
        ny, nx = shape

        # Create coordinate grids centered at image center
        y_coords, x_coords = np.mgrid[0:ny, 0:nx]
        y_coords = y_coords.astype(np.float32) - ny/2  # center at 0
        x_coords = x_coords.astype(np.float32) - nx/2  # center at 0

        # Convert angle to radians
        angle_rad = np.radians(gradient_angle)

        # Project coordinates along gradient direction
        gradient_coords = (x_coords * np.cos(angle_rad) +
                          y_coords * np.sin(angle_rad))

        # Normalize by image size to get gradient per 100 pixels
        gradient_coords = gradient_coords / (max(nx, ny) / 100.0)

        # Apply gradient
        wavelength_map = base_wavelength + gradient_magnitude * gradient_coords

        return wavelength_map.astype(np.float32)

    def create_datacube(self, image_files: List[str],
                       output_filename: str = "spherex_datacube.fits") -> str:
        """
        Create a 3D data cube from reprojected images with standardized wavelength grid.

        The output cube uses a fixed wavelength grid from 2-5 microns with 0.04 micron steps.
        Each input image has a wavelength gradient extracted from WCS/headers or modeled based
        on SPHEREX disperser characteristics. Each pixel contributes to 1-2 adjacent channels
        based on its local wavelength, ensuring images contribute to multiple channels.

        Parameters
        ----------
        image_files : List[str]
            List of reprojected image files
        wavelength_map : Dict[str, float]
            Mapping from file path to central wavelength
        output_filename : str
            Name of output data cube file

        Returns
        -------
        str
            Path to created data cube
        """
        if self.verbose:
            print(f"Creating standardized data cube from {len(image_files)} images...")
            print(f"  Output wavelength grid: {self.output_wavelength_min}-{self.output_wavelength_max} μm")
            print(f"  Spectral resolution: R~100 ({self.output_wavelength_step:.3f} μm steps)")
            print(f"  Number of output channels: {len(self.output_wavelengths)}")

        output_path = self.processed_dir / output_filename

        # Read the first image to get spatial dimensions and WCS
        with fits.open(image_files[0]) as hdul:
            reference_data = hdul[0].data
            reference_header = hdul[0].header.copy()
            reference_wcs = WCS(reference_header)
            reference_header = reference_wcs.to_header()

        ny, nx = reference_data.shape
        nz_output = len(self.output_wavelengths)

        if self.verbose:
            print(f"  Spatial dimensions: {nx} x {ny}")
            print(f"  Creating output cube: {nz_output} x {ny} x {nx}")

        # Pre-allocate the output data cube with the standardized wavelength grid
        output_datacube = np.full((nz_output, ny, nx), np.nan, dtype=np.float32)
        channel_weights = np.zeros((nz_output, ny, nx), dtype=np.float32)  # Track sum of weights per channel

        if self.verbose:
            print(f"  Mapping images to wavelength channels...")

        # Process each input image
        for i, filepath in enumerate(image_files):
            try:
                with fits.open(filepath) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                    wavelength_map_2d = wavelength = hdul['WAVELENGTH'].data
                    if data.shape != (ny, nx):
                        raise ValueError(f"Image {Path(filepath).name} has different shape, skipping")

                # Vectorized processing of all pixels at once
                # Create masks for valid data and wavelength range
                valid_data_mask = np.isfinite(data)
                valid_wavelength_mask = ((wavelength_map_2d >= self.output_wavelength_min) &
                                       (wavelength_map_2d <= self.output_wavelength_max))
                combined_mask = valid_data_mask & valid_wavelength_mask

                if not np.any(combined_mask):
                    raise ValueError(f"No valid pixels in wavelength range for {Path(filepath).name}")

                # Get valid pixel data and wavelengths
                valid_data = data[combined_mask]
                valid_wavelengths = wavelength_map_2d[combined_mask]
                valid_y_coords, valid_x_coords = np.where(combined_mask)

                # Vectorized channel distance calculation for all valid pixels
                # Shape: (n_pixels, n_channels)
                pixel_channel_distances = np.abs(valid_wavelengths[:, np.newaxis] - self.output_wavelengths[np.newaxis, :])

                # Find closest and second closest channels for each pixel
                # Shape: (n_pixels, n_channels) -> (n_pixels, 2)
                # TODO: refactor this so that all channels get _some_ weight, and it's inverse-distance weighted
                # then the cube will be fully populated (incorrectly...) and we can mask out pixels with max weight less than some threshold later on
                sorted_channel_indices = np.argsort(pixel_channel_distances, axis=1)[:, :2]
                ch1_indices = sorted_channel_indices[:, 0]
                ch2_indices = sorted_channel_indices[:, 1]

                # Get distances to closest channels
                pixel_indices = np.arange(len(valid_data))
                dist1 = pixel_channel_distances[pixel_indices, ch1_indices]
                dist2 = pixel_channel_distances[pixel_indices, ch2_indices]

                # Determine which pixels contribute to 2 channels vs 1 channel
                two_channel_mask = dist2 <= self.output_wavelength_step

                # Calculate weights for two-channel contributions
                weights1 = np.ones_like(dist1) * np.nan
                weights2 = np.zeros_like(dist2) * np.nan

                # For pixels contributing to 2 channels, calculate inverse distance weights
                two_ch_pixels = np.where(two_channel_mask)[0]
                if len(two_ch_pixels) > 0:
                    d1_two = dist1[two_ch_pixels]
                    d2_two = dist2[two_ch_pixels]

                    # Handle zero distances
                    zero_d1 = (d1_two == 0)
                    zero_d2 = (d2_two == 0)

                    # Default inverse distance weights
                    w1_two = np.where(zero_d1, 1.0, 1.0 / np.maximum(d1_two, 1e-10))
                    w2_two = np.where(zero_d2, 1.0, 1.0 / np.maximum(d2_two, 1e-10))

                    # Handle special cases where one distance is zero
                    w1_two = np.where(zero_d1, 1.0, w1_two)
                    w2_two = np.where(zero_d1, 0.0, w2_two)
                    w1_two = np.where(zero_d2, 0.0, w1_two)
                    w2_two = np.where(zero_d2, 1.0, w2_two)

                    # Normalize weights to sum to 1
                    total_weights = w1_two + w2_two
                    w1_two = w1_two / total_weights
                    w2_two = w2_two / total_weights

                    weights1[two_ch_pixels] = w1_two
                    weights2[two_ch_pixels] = w2_two

                if np.any(np.isnan(weights1)) or np.any(np.isnan(weights2)):
                    raise ValueError(f"Weights are NaN for {Path(filepath).name}")

                # Vectorized channel assignment using advanced indexing
                # Filter for valid channels
                valid_ch1_mask = (ch1_indices >= 0) & (ch1_indices < nz_output) & (weights1 > 0)
                valid_ch2_mask = (ch2_indices >= 0) & (ch2_indices < nz_output) & (weights2 > 0) & two_channel_mask

                # this is a corner case that can't happen .... but does?!?
                # Get all unique channels that will receive contributions
                all_contributing_channels = np.concatenate([
                    ch1_indices[valid_ch1_mask],
                    ch2_indices[valid_ch2_mask]
                ])
                contributing_channels_set = np.unique(all_contributing_channels)

                # Initialize all required channels at once
                for ch_idx in contributing_channels_set:
                    if np.all(np.isnan(output_datacube[ch_idx])):
                        # in theory this should never happen because nothing is setting these to nan...
                        print(f"    Warning: Channel {ch_idx} is all NaN, setting to zeros")
                        output_datacube[ch_idx] = np.zeros((ny, nx), dtype=np.float32)
                        channel_weights[ch_idx] = np.zeros((ny, nx), dtype=np.float32)

                # Vectorized assignment for first channel contributions
                if np.any(valid_ch1_mask):
                    valid_pixels_ch1 = np.where(valid_ch1_mask)[0]
                    ch1_valid = ch1_indices[valid_pixels_ch1]
                    weights1_valid = weights1[valid_pixels_ch1]
                    data_valid_ch1 = valid_data[valid_pixels_ch1]
                    y_coords_ch1 = valid_y_coords[valid_pixels_ch1]
                    x_coords_ch1 = valid_x_coords[valid_pixels_ch1]

                    # Use advanced indexing for batch assignment
                    output_datacube[ch1_valid, y_coords_ch1, x_coords_ch1] += data_valid_ch1 * weights1_valid
                    channel_weights[ch1_valid, y_coords_ch1, x_coords_ch1] += weights1_valid

                # Vectorized assignment for second channel contributions
                if np.any(valid_ch2_mask):
                    valid_pixels_ch2 = np.where(valid_ch2_mask)[0]
                    ch2_valid = ch2_indices[valid_pixels_ch2]
                    weights2_valid = weights2[valid_pixels_ch2]
                    data_valid_ch2 = valid_data[valid_pixels_ch2]
                    y_coords_ch2 = valid_y_coords[valid_pixels_ch2]
                    x_coords_ch2 = valid_x_coords[valid_pixels_ch2]

                    # Use advanced indexing for batch assignment
                    output_datacube[ch2_valid, y_coords_ch2, x_coords_ch2] += data_valid_ch2 * weights2_valid
                    channel_weights[ch2_valid, y_coords_ch2, x_coords_ch2] += weights2_valid

                if self.verbose:
                    # Report wavelength range and channel contributions for this image
                    wl_min = np.nanmin(wavelength_map_2d)
                    wl_max = np.nanmax(wavelength_map_2d)
                    channels_list = sorted(contributing_channels_set)
                    channel_info = f"channels {channels_list[0]}-{channels_list[-1]}" if channels_list else "no channels"
                    print(f"    Image {i+1}/{len(image_files)}: {Path(filepath).name} "
                          f"(λ: {wl_min:.3f}-{wl_max:.3f} μm) -> {channel_info} ({len(channels_list)} total)")

            except Exception as e:
                if self.verbose:
                    print(f"    Error processing {filepath}: {e}")
                continue

        # Compute weighted averages where multiple images contributed to the same channel/pixel
        if self.verbose:
            print(f"  Finalizing weighted averages...")

        total_contributions = 0
        for channel_idx in range(nz_output):
            # Find pixels with contributions (weight > 0)
            contributing_pixels = channel_weights[channel_idx] > 0

            if np.any(contributing_pixels):
                # Compute weighted average: sum(weight * value) / sum(weight)
                output_datacube[channel_idx][contributing_pixels] /= channel_weights[channel_idx][contributing_pixels]
                total_contributions += np.sum(contributing_pixels)
            else:
                # No contributions to this channel - leave as NaN
                output_datacube[channel_idx] = np.full((ny, nx), np.nan, dtype=np.float32)

        if self.verbose:
            channels_with_data = np.sum([np.any(np.isfinite(output_datacube[i])) for i in range(nz_output)])
            print(f"  Channels with data: {channels_with_data}/{nz_output}")
            print(f"  Total pixel contributions: {total_contributions:,}")

            # Report wavelength coverage
            first_channel_with_data = None
            last_channel_with_data = None
            for i in range(nz_output):
                if np.any(np.isfinite(output_datacube[i])):
                    if first_channel_with_data is None:
                        first_channel_with_data = i
                    last_channel_with_data = i

            if first_channel_with_data is not None:
                wl_min = self.output_wavelengths[first_channel_with_data]
                wl_max = self.output_wavelengths[last_channel_with_data]
                print(f"  Wavelength coverage: {wl_min:.3f} - {wl_max:.3f} μm")

        # Update header for standardized 3D cube
        reference_header['NAXIS'] = 3
        reference_header['NAXIS3'] = nz_output
        reference_header['CTYPE3'] = 'WAVE'
        reference_header['CUNIT3'] = 'um'
        reference_header['CRPIX3'] = 1
        reference_header['CRVAL3'] = self.output_wavelengths[0]
        reference_header['CDELT3'] = self.output_wavelength_step

        # Add metadata
        reference_header['OBJECT'] = 'SPHEREX Data Cube'
        reference_header['COMMENT'] = f'Standardized data cube from {len(image_files)} SPHEREX images'
        reference_header['COMMENT'] = f'Wavelength grid: {self.output_wavelength_min}-{self.output_wavelength_max} um, step={self.output_wavelength_step} um'
        reference_header['COMMENT'] = f'Spectral resolution R~{self.output_wavelength_min/self.output_wavelength_step:.0f}'
        reference_header['COMMENT'] = 'Images mapped directly to channels without interpolation'
        reference_header['BUNIT'] = 'Various'  # Units may vary between input images
        reference_header['METHOD'] = 'DIRECT'  # Direct channel mapping method
        # not needed reference_header['PC1_3'] = 0.0
        # not needed reference_header['PC2_3'] = 0.0
        # not needed reference_header['PC3_3'] = 1.0
        # not needed reference_header['PC3_1'] = 0.0
        # not needed reference_header['PC3_2'] = 0.0

        # Create HDU list
        primary_hdu = fits.PrimaryHDU(data=output_datacube, header=reference_header)

        # Create wavelength extension with standardized grid
        # wavelength_hdu = fits.ImageHDU(data=self.output_wavelengths.astype(np.float32), name='WAVELENGTH')
        # wavelength_hdu.header['TTYPE1'] = 'WAVELENGTH'
        # wavelength_hdu.header['TUNIT1'] = 'um'
        # wavelength_hdu.header['COMMENT'] = 'Standardized wavelength grid (2-5 um, 0.1 um steps)'

        # Create input file list extension
        filenames = [Path(f).name for f in image_files]
        filename_col = fits.Column(name='FILENAME', format='A50', array=filenames)
        #input_wavelength_col = fits.Column(name='INPUT_WAVELENGTH', format='E',
        #                                 array=sorted_wavelengths.astype(np.float32), unit='um')
        file_table = fits.BinTableHDU.from_columns([filename_col, ], name='INPUT_FILES')
        file_table.header['COMMENT'] = 'Original input files and their wavelengths'

        wcs = WCS(primary_hdu.header)
        primary_hdu.header.update(wcs.to_header())

        hdul = fits.HDUList([primary_hdu, file_table])

        # Save data cube
        hdul.writeto(output_path, overwrite=True)

        if self.verbose:
            print(f"  Data cube saved: {output_path}")
            print(f"  Output cube shape: {output_datacube.shape} (wavelength, y, x)")
            print(f"  Data type: {output_datacube.dtype}")
            print(f"  Wavelength channels: {nz_output}")

        return str(output_path)

    def plot_summary(self, datacube_path: str, target_coords: SkyCoord = None):
        """
        Create summary plots of the data cube.

        Parameters
        ----------
        datacube_path : str
            Path to the data cube FITS file
        target_coords : SkyCoord, optional
            Target coordinates for marking
        """
        if not HAS_MATPLOTLIB:
            if self.verbose:
                print("Matplotlib not available. Skipping plots.")
            return

        if self.verbose:
            print("Creating summary plots...")

        with fits.open(datacube_path) as hdul:
            datacube = hdul[0].data
            header = hdul[0].header
            wcs = WCS(header)
            wcs.sip = None
            wcs = wcs.celestial

            if 'WAVELENGTH' in hdul:
                wavelengths = hdul['WAVELENGTH'].data
            else:
                # Reconstruct from header
                nz = datacube.shape[0]
                crval3 = header.get('CRVAL3', 1.0)
                cdelt3 = header.get('CDELT3', 1.0)
                wavelengths = crval3 + np.arange(nz) * cdelt3

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('SPHEREX Data Cube Summary', fontsize=16)

        # 1. RGB composite (using first, middle, last wavelengths)
        ax1 = axes[0, 0]
        if datacube.shape[0] >= 3:
            red_idx = -1
            green_idx = datacube.shape[0] // 2
            blue_idx = 0

            red = datacube[red_idx]
            green = datacube[green_idx]
            blue = datacube[blue_idx]

            # Normalize each channel
            def normalize_image(img):
                mean, median, std = sigma_clipped_stats(img, sigma=3.0)
                return np.clip((img - median) / (3 * std) + 0.5, 0, 1)

            rgb = np.stack([
                normalize_image(red),
                normalize_image(green),
                normalize_image(blue)
            ], axis=-1)

            ax1.imshow(rgb, origin='lower')
            ax1.set_title(f'RGB Composite\nR:{wavelengths[red_idx]:.2f} G:{wavelengths[green_idx]:.2f} B:{wavelengths[blue_idx]:.2f} μm')
        else:
            ax1.imshow(datacube[0], origin='lower', cmap='viridis')
            ax1.set_title(f'Single Band: {wavelengths[0]:.2f} μm')

        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')

        # Mark target if provided
        if target_coords is not None:
            try:
                x_pix, y_pix = wcs.world_to_pixel(target_coords)
                ax1.plot(x_pix, y_pix, 'r+', markersize=10, markeredgewidth=2)
                ax1.text(x_pix+5, y_pix+5, 'Target', color='red', fontweight='bold')
            except:
                pass

        # 2. Spectrum at center pixel
        ax2 = axes[0, 1]
        cy, cx = datacube.shape[1]//2, datacube.shape[2]//2
        spectrum = datacube[:, cy, cx]
        ax2.plot(wavelengths, spectrum, 'b-', linewidth=2)
        ax2.set_xlabel('Wavelength (μm)')
        ax2.set_ylabel('Flux')
        ax2.set_title(f'Spectrum at center pixel ({cx}, {cy})')
        ax2.grid(True, alpha=0.3)

        # 3. Wavelength coverage
        ax3 = axes[1, 0]
        ax3.bar(range(len(wavelengths)), wavelengths, alpha=0.7)
        ax3.set_xlabel('Channel Number')
        ax3.set_ylabel('Wavelength (μm)')
        ax3.set_title('Wavelength Coverage')
        ax3.grid(True, alpha=0.3)

        # 4. Data cube statistics
        ax4 = axes[1, 1]

        # Calculate statistics for each wavelength
        means = np.nanmean(datacube, axis=(1, 2))
        stds = np.nanstd(datacube, axis=(1, 2))

        ax4.errorbar(wavelengths, means, yerr=stds, fmt='o-', capsize=3)
        ax4.set_xlabel('Wavelength (μm)')
        ax4.set_ylabel('Mean Flux ± Std')
        ax4.set_title('Data Cube Statistics')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = Path(datacube_path).parent / 'spherex_summary.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')

        if self.verbose:
            print(f"  Summary plot saved: {plot_path}")

        plt.show()

    def process_target(self,
                      target: str = None,
                      coordinates: SkyCoord = None,
                      radius: u.Quantity = 10*u.arcmin,
                      frame_size: u.Quantity = 5*u.arcmin,
                      max_images: int = None,
                      create_plots: bool = True,
                      skip_download: bool = False
                      ) -> str:
        """
        Complete processing workflow for a target.

        Parameters
        ----------
        target : str, optional
            Target name
        coordinates : SkyCoord, optional
            Target coordinates
        radius : Quantity
            Search radius for images
        frame_size : Quantity
            Size of common frame
        max_images : int, optional
            Maximum number of images to process
        create_plots : bool
            Whether to create summary plots

        Returns
        -------
        str
            Path to final data cube
        """
        if self.verbose:
            print("="*60)
            print("SPHEREX Data Processing Workflow")
            print("="*60)

        # Step 1: Search for images
        if not skip_download:
            images = self.search_spherex_images(target, coordinates, radius)

            if len(images) == 0:
                print("No SPHEREX images found for the specified target/region")
                return None

        # Step 2: Download images
        if not skip_download:
            image_files = self.download_images(images, max_images)

            if len(image_files) == 0:
                print("No images were successfully downloaded")
                return None

        else:
            image_files = glob.glob(os.path.join(self.output_dir, "images", "*.fits"))
            if max_images is not None:
                image_files = image_files[:max_images]

        assert len(image_files) > 0, "No images found"

        # Step 3: Determine common frame
        target_coords = coordinates if coordinates is not None else SkyCoord.from_name(target)
        common_wcs, output_shape = self.determine_common_frame(
            image_files, target_coords, frame_size
        )

        # Step 4: Reproject images
        reprojected_files = self.reproject_images(image_files, common_wcs, output_shape)

        if len(reprojected_files) == 0:
            print("No images were successfully reprojected")
            return None

        # Step 5: Assign wavelengths
        # wavelength_map = self.assign_wavelengths(reprojected_files)

        # Step 6: Create data cube
        datacube_path = self.create_datacube(reprojected_files)

        # Step 7: Create summary plots
        if create_plots:
            self.plot_summary(datacube_path, target_coords)

        if self.verbose:
            print("="*60)
            print("Processing completed successfully!")
            print(f"Data cube: {datacube_path}")
            print("="*60)

        return datacube_path


def main():
    """
    Command line interface for SPHEREX data processing.
    """
    parser = argparse.ArgumentParser(
        description="Download and process SPHEREX data from IRSA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Target specification
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        '--target', '-t', type=str,
        help='Target name (e.g., "M31", "NGC 1234")'
    )
    target_group.add_argument(
        '--coordinates', '-c', type=str,
        help='Target coordinates (e.g., "10.68 +41.27" or "00h42m43.2s +41d16m12s")'
    )

    # Search parameters
    parser.add_argument(
        '--radius', '-r', type=float, default=10.0,
        help='Search radius in arcminutes'
    )
    parser.add_argument(
        '--frame-size', '-f', type=float, default=5.0,
        help='Common frame size in arcminutes'
    )
    parser.add_argument(
        '--max-images', '-m', type=int,
        help='Maximum number of images to download and process'
    )

    # Output options
    parser.add_argument(
        '--output-dir', '-o', type=str, default='spherex_data',
        help='Output directory for downloaded and processed data'
    )
    parser.add_argument(
        '--output-name', type=str, default='spherex_datacube.fits',
        help='Name of output data cube file'
    )

    # Processing options
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip creating summary plots'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress verbose output'
    )
    parser.add_argument(
        '--skip-download', action='store_true',
        help='Skip downloading images'
    )

    args = parser.parse_args()

    # Parse coordinates if provided
    coordinates = None
    if args.coordinates:
        coordinates = SkyCoord(args.coordinates, unit=(u.hourangle, u.deg))

    # Initialize downloader
    downloader = SPHEREXDownloader(
        output_dir=args.output_dir,
        verbose=not args.quiet
    )

    # Process the target
    datacube_path = downloader.process_target(
        target=args.target,
        coordinates=coordinates,
        radius=args.radius * u.arcmin,
        frame_size=args.frame_size * u.arcmin,
        max_images=args.max_images,
        create_plots=not args.no_plots,
        skip_download=args.skip_download
    )

    if datacube_path:
        print(f"\nSuccess! Data cube created: {datacube_path}")
    else:
        print("\nFailed to create data cube.")


if __name__ == '__main__':
    main()
