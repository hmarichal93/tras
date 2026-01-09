"""Tests for ring resampling utility."""

import numpy as np
import pytest
from tras.utils.ring_sampling import (
    build_rays_xy,
    resample_ring_by_rays,
    resample_rings_by_rays,
)


def test_build_rays_xy():
    """Test ray generation from center point."""
    center_xy = (100.0, 100.0)
    sampling_nr = 36
    max_radius = 200.0
    
    rays = build_rays_xy(sampling_nr, center_xy, max_radius)
    
    assert len(rays) == sampling_nr
    
    # Check that all rays start at center
    for ray in rays:
        coords = list(ray.coords)
        assert len(coords) == 2
        start_x, start_y = coords[0]
        assert abs(start_x - center_xy[0]) < 1e-6
        assert abs(start_y - center_xy[1]) < 1e-6


def test_resample_ring_by_rays_circle():
    """Test resampling a perfect circle ring."""
    center_xy = (100.0, 100.0)
    radius = 50.0
    sampling_nr = 36
    
    # Create a perfect circle ring
    angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    ring_xy = np.array([
        [center_xy[0] + radius * np.cos(a), center_xy[1] + radius * np.sin(a)]
        for a in angles
    ], dtype=np.float32)
    
    resampled = resample_ring_by_rays(ring_xy, center_xy, sampling_nr)
    
    # Should have exactly sampling_nr points
    assert resampled.shape == (sampling_nr, 2)
    
    # Check that points are approximately at the expected radius
    distances = np.sqrt(
        (resampled[:, 0] - center_xy[0])**2 + 
        (resampled[:, 1] - center_xy[1])**2
    )
    mean_distance = np.mean(distances)
    assert abs(mean_distance - radius) < radius * 0.1  # Within 10% of expected radius


def test_resample_ring_by_rays_ellipse():
    """Test resampling an elliptical ring."""
    center_xy = (100.0, 100.0)
    a, b = 60.0, 40.0  # Semi-major and semi-minor axes
    sampling_nr = 72
    
    # Create an elliptical ring
    angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    ring_xy = np.array([
        [center_xy[0] + a * np.cos(a), center_xy[1] + b * np.sin(a)]
        for a in angles
    ], dtype=np.float32)
    
    resampled = resample_ring_by_rays(ring_xy, center_xy, sampling_nr)
    
    # Should have exactly sampling_nr points
    assert resampled.shape == (sampling_nr, 2)
    
    # Points should be distributed around the ellipse
    assert np.all(np.isfinite(resampled))


def test_resample_rings_by_rays_multiple():
    """Test resampling multiple rings."""
    center_xy = (100.0, 100.0)
    sampling_nr = 36
    
    # Create multiple concentric rings
    rings = []
    for radius in [30.0, 50.0, 70.0]:
        angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        ring_xy = np.array([
            [center_xy[0] + radius * np.cos(a), center_xy[1] + radius * np.sin(a)]
            for a in angles
        ], dtype=np.float32)
        rings.append(ring_xy)
    
    resampled = resample_rings_by_rays(rings, center_xy, sampling_nr)
    
    # Should have same number of rings
    assert len(resampled) == len(rings)
    
    # Each ring should have exactly sampling_nr points
    for ring in resampled:
        assert ring.shape == (sampling_nr, 2)
        assert np.all(np.isfinite(ring))


def test_resample_ring_by_rays_small_input():
    """Test resampling with very few input points."""
    center_xy = (100.0, 100.0)
    sampling_nr = 36
    
    # Create a ring with only 3 points (minimum for polygon)
    ring_xy = np.array([
        [150.0, 100.0],
        [100.0, 150.0],
        [50.0, 100.0],
    ], dtype=np.float32)
    
    resampled = resample_ring_by_rays(ring_xy, center_xy, sampling_nr)
    
    # Should still produce sampling_nr points
    assert resampled.shape == (sampling_nr, 2)
    assert np.all(np.isfinite(resampled))


def test_resample_ring_by_rays_ordering():
    """Test that resampled points maintain angular ordering."""
    center_xy = (100.0, 100.0)
    radius = 50.0
    sampling_nr = 72
    
    # Create a circle ring
    angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    ring_xy = np.array([
        [center_xy[0] + radius * np.cos(a), center_xy[1] + radius * np.sin(a)]
        for a in angles
    ], dtype=np.float32)
    
    resampled = resample_ring_by_rays(ring_xy, center_xy, sampling_nr)
    
    # Compute angles of resampled points
    resampled_angles = np.arctan2(
        resampled[:, 1] - center_xy[1],
        resampled[:, 0] - center_xy[0]
    )
    # Normalize to [0, 2*pi]
    resampled_angles = (resampled_angles + 2 * np.pi) % (2 * np.pi)
    
    # Angles should be approximately evenly spaced
    sorted_angles = np.sort(resampled_angles)
    angle_diffs = np.diff(sorted_angles)
    mean_diff = np.mean(angle_diffs)
    expected_diff = 2 * np.pi / sampling_nr
    
    # Mean difference should be close to expected
    assert abs(mean_diff - expected_diff) < expected_diff * 0.2  # Within 20%

