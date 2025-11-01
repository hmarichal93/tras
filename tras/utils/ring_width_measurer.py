"""
Ring width measurement along a radial line
"""
import numpy as np
from shapely.geometry import LineString, Polygon, Point


def compute_ring_widths_along_line(rings, pith_xy, direction_xy):
    """
    Compute ring widths along a radial line from pith in specified direction.
    
    Args:
        rings: List of ring shapes (each has .points attribute with QPointF objects)
        pith_xy: Tuple (x, y) of pith coordinates
        direction_xy: Tuple (x, y) of a point defining the direction from pith
        
    Returns:
        dict: {ring_label: {'intersection_point': (x, y), 'radial_width': float}}
    """
    if not rings:
        return {}
    
    px, py = pith_xy
    dx, dy = direction_xy
    
    # Create a very long line from pith in the specified direction
    # Direction vector
    vx, vy = dx - px, dy - py
    length = np.sqrt(vx**2 + vy**2)
    if length < 1e-6:
        return {}
    
    # Normalize and extend to a very long line (10000 pixels)
    vx, vy = vx / length, vy / length
    end_x = px + vx * 10000
    end_y = py + vy * 10000
    
    # Create radial line from pith
    radial_line = LineString([(px, py), (end_x, end_y)])
    
    # Compute intersections for each ring
    results = {}
    intersections = []  # List of (distance_from_pith, ring_label, intersection_point)
    
    for shape in rings:
        if not shape.label or not shape.label.startswith("ring_"):
            continue
        
        # Convert ring points to polygon
        points = [(p.x(), p.y()) for p in shape.points]
        if len(points) < 3:
            continue
        
        try:
            ring_polygon = Polygon(points)
            
            # Find intersection with radial line
            intersection = ring_polygon.intersection(radial_line)
            
            if intersection.is_empty:
                continue
            
            # Get the intersection point(s)
            # Usually a ring will intersect the radial line at 2 points (entry and exit)
            # We want the point farthest from pith (outer intersection)
            intersection_points = []
            
            if intersection.geom_type == 'Point':
                intersection_points = [intersection]
            elif intersection.geom_type == 'MultiPoint':
                intersection_points = list(intersection.geoms)
            elif intersection.geom_type == 'LineString':
                # If intersection is a line segment, take the farthest point
                coords = list(intersection.coords)
                intersection_points = [Point(coords[0]), Point(coords[-1])]
            elif intersection.geom_type == 'MultiLineString':
                # Multiple line segments, collect all endpoints
                for line in intersection.geoms:
                    coords = list(line.coords)
                    intersection_points.extend([Point(coords[0]), Point(coords[-1])])
            
            if not intersection_points:
                continue
            
            # Find the farthest intersection point from pith (outer boundary)
            max_dist = -1
            farthest_point = None
            for pt in intersection_points:
                dist = np.sqrt((pt.x - px)**2 + (pt.y - py)**2)
                if dist > max_dist:
                    max_dist = dist
                    farthest_point = pt
            
            if farthest_point:
                intersections.append((max_dist, shape.label, (farthest_point.x, farthest_point.y)))
        
        except Exception as e:
            print(f"Warning: Could not compute intersection for {shape.label}: {e}")
            continue
    
    # Sort intersections by distance from pith (innermost to outermost)
    intersections.sort(key=lambda x: x[0])
    
    # Compute ring widths (distance between consecutive intersections)
    for i, (dist, label, point) in enumerate(intersections):
        if i == 0:
            # First (innermost) ring: width is distance from pith to first intersection
            radial_width = dist
        else:
            # Width is distance from previous ring
            prev_dist = intersections[i-1][0]
            radial_width = dist - prev_dist
        
        results[label] = {
            'intersection_point': point,
            'distance_from_pith': dist,
            'radial_width': radial_width
        }
    
    return results


def extract_ring_number(label):
    """Extract numeric part from ring label (ring_123 or ring_2020)"""
    try:
        parts = label.split('_')
        if len(parts) >= 2:
            return int(parts[1])
        return 0
    except (IndexError, ValueError):
        return 0

