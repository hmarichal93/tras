"""
Ring resampling utility for post-processing detected tree rings.

This module provides ray-based resampling of tree ring polygons to a fixed
number of points (sampling_nr) distributed evenly around the ring from the pith center.
"""

import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon, LineString, Point


def build_rays_xy(sampling_nr: int, center_xy: Tuple[float, float], max_radius: float) -> List[LineString]:
    """
    Build rays from center point outward in evenly spaced angles.
    
    Args:
        sampling_nr: Number of rays (radial samples)
        center_xy: Center point as (x, y) tuple
        max_radius: Maximum radius to extend rays (should be large enough to intersect all rings)
    
    Returns:
        List of LineString rays, each from center to edge at evenly spaced angles
    """
    cx, cy = center_xy
    angles_deg = np.arange(0, 360, 360 / sampling_nr)
    rays = []
    
    for angle_deg in angles_deg:
        angle_rad = np.deg2rad(angle_deg)
        # Calculate endpoint at max_radius distance
        end_x = cx + max_radius * np.cos(angle_rad)
        end_y = cy + max_radius * np.sin(angle_rad)
        ray = LineString([(cx, cy), (end_x, end_y)])
        rays.append(ray)
    
    return rays


def resample_ring_by_rays(ring_xy: np.ndarray, center_xy: Tuple[float, float], sampling_nr: int) -> np.ndarray:
    """
    Resample a ring polygon to exactly sampling_nr points using ray intersections.
    
    For each ray from the center, finds the intersection with the ring boundary
    and selects the nearest outward intersection point.
    
    Args:
        ring_xy: Ring polygon as Nx2 array of (x, y) points
        center_xy: Center point (pith) as (x, y) tuple
        sampling_nr: Number of points to resample to
    
    Returns:
        Resampled ring as sampling_nr x 2 array of (x, y) points, ordered by angle
    """
    if len(ring_xy) < 3:
        # Not enough points to form a polygon, return as-is
        return ring_xy
    
    # Create Shapely polygon from ring points
    # Note: ring_xy is already in (x, y) format
    ring_poly = Polygon(ring_xy)
    
    if not ring_poly.is_valid:
        # Try to fix invalid polygon (e.g., self-intersections)
        ring_poly = ring_poly.buffer(0)
    
    # Calculate max radius from center to furthest ring point
    cx, cy = center_xy
    distances = np.sqrt((ring_xy[:, 0] - cx)**2 + (ring_xy[:, 1] - cy)**2)
    max_radius = np.max(distances) * 1.5  # Add 50% margin
    
    # Build rays
    rays = build_rays_xy(sampling_nr, center_xy, max_radius)
    
    # Find intersections for each ray
    resampled_points = []
    for ray in rays:
        try:
            intersection = ray.intersection(ring_poly.boundary)
            
            if intersection.is_empty:
                # No intersection found - use closest point on ring boundary
                # This can happen if ray goes through a gap
                # Fallback: use the point on ring boundary closest to ray endpoint
                ray_end = Point(ray.coords[-1])
                closest_point = None
                min_dist = float('inf')
                for i in range(len(ring_xy)):
                    pt = Point(ring_xy[i])
                    dist = ray_end.distance(pt)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = (ring_xy[i][0], ring_xy[i][1])
                if closest_point:
                    resampled_points.append(closest_point)
                else:
                    # Last resort: use ray endpoint projected onto ring
                    resampled_points.append((ray.coords[-1][0], ray.coords[-1][1]))
            elif isinstance(intersection, Point):
                # Single intersection point
                resampled_points.append((intersection.x, intersection.y))
            elif isinstance(intersection, LineString):
                # Multiple points or line segment - use the furthest point from center
                coords = list(intersection.coords)
                if len(coords) > 0:
                    # Find point furthest from center (outward intersection)
                    furthest = max(coords, key=lambda p: np.sqrt((p[0] - cx)**2 + (p[1] - cy)**2))
                    resampled_points.append(furthest)
                else:
                    # Fallback to first coordinate
                    resampled_points.append((coords[0][0], coords[0][1]))
            else:
                # MultiPoint or other geometry - extract first point
                if hasattr(intersection, 'geoms'):
                    first_geom = intersection.geoms[0]
                    if isinstance(first_geom, Point):
                        resampled_points.append((first_geom.x, first_geom.y))
                    else:
                        # Fallback
                        coords = list(first_geom.coords) if hasattr(first_geom, 'coords') else []
                        if coords:
                            resampled_points.append((coords[0][0], coords[0][1]))
                        else:
                            # Use ray endpoint
                            resampled_points.append((ray.coords[-1][0], ray.coords[-1][1]))
                else:
                    # Fallback to ray endpoint
                    resampled_points.append((ray.coords[-1][0], ray.coords[-1][1]))
        except Exception:
            # If intersection fails, use closest point on ring
            ray_end = Point(ray.coords[-1])
            closest_point = None
            min_dist = float('inf')
            for i in range(len(ring_xy)):
                pt = Point(ring_xy[i])
                dist = ray_end.distance(pt)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = (ring_xy[i][0], ring_xy[i][1])
            if closest_point:
                resampled_points.append(closest_point)
            else:
                resampled_points.append((ray.coords[-1][0], ray.coords[-1][1]))
    
    # Ensure we have exactly sampling_nr points
    if len(resampled_points) != sampling_nr:
        # This shouldn't happen, but handle it gracefully
        if len(resampled_points) < sampling_nr:
            # Duplicate last point to fill
            while len(resampled_points) < sampling_nr:
                resampled_points.append(resampled_points[-1])
        else:
            # Truncate
            resampled_points = resampled_points[:sampling_nr]
    
    return np.array(resampled_points, dtype=np.float32)


def resample_rings_by_rays(
    rings: List[np.ndarray], 
    center_xy: Tuple[float, float], 
    sampling_nr: int
) -> List[np.ndarray]:
    """
    Resample multiple rings to exactly sampling_nr points each using ray intersections.
    
    Args:
        rings: List of ring polygons, each as Nx2 array of (x, y) points
        center_xy: Center point (pith) as (x, y) tuple
        sampling_nr: Number of points to resample each ring to
    
    Returns:
        List of resampled rings, each as sampling_nr x 2 array
    """
    resampled = []
    for ring in rings:
        resampled_ring = resample_ring_by_rays(ring, center_xy, sampling_nr)
        resampled.append(resampled_ring)
    return resampled

