#!/usr/bin/env python3
"""
Generate default racing track PNG for TrackLearnerAI
Creates a simple oval track with clear boundaries for collision detection
"""

from PIL import Image, ImageDraw
import os
import math

def create_oval_track(width=1200, height=700, filename="assets/tracks/track1.png"):
    """
    Create a simple, clean racing track:
    - Oval loop with smooth curves
    - Black boundaries on white background
    - Checkered finish line

    The white area is drivable, black is walls
    """
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img, 'RGBA')

    center_x, center_y = width // 2, height // 2

    # Track parameters
    outer_a = 350  # Horizontal semi-axis (outer)
    outer_b = 200  # Vertical semi-axis (outer)
    inner_a = 250  # Horizontal semi-axis (inner)
    inner_b = 100  # Vertical semi-axis (inner)

    # Generate outer ellipse points
    outer_points = []
    for i in range(360):
        angle = math.radians(i)
        x = center_x + outer_a * math.cos(angle)
        y = center_y + outer_b * math.sin(angle)
        outer_points.append((x, y))

    # Generate inner ellipse points
    inner_points = []
    for i in range(360):
        angle = math.radians(i)
        x = center_x + inner_a * math.cos(angle)
        y = center_y + inner_b * math.sin(angle)
        inner_points.append((x, y))

    # Draw outer boundary (thick black line)
    draw.line(outer_points + [outer_points[0]], fill='black', width=12)

    # Draw inner boundary (thick black line)
    draw.line(inner_points + [inner_points[0]], fill='black', width=12)

    # Draw finish line area (checkered pattern at top)
    finish_y = center_y - outer_b - 40
    finish_left = center_x - 80
    finish_right = center_x + 80
    square_size = 15

    for i in range(-100, 100, square_size):
        for j in range(4):
            x = finish_left + i
            y = finish_y + j * square_size
            if (i // square_size + j) % 2 == 0:
                draw.rectangle(
                    [x, y, x + square_size, y + square_size],
                    fill='black'
                )

    # Draw dashed center line
    for i in range(0, 360, 10):
        angle1 = math.radians(i)
        angle2 = math.radians(i + 5)

        center_a = (outer_a + inner_a) / 2
        center_b = (outer_b + inner_b) / 2

        x1 = center_x + center_a * math.cos(angle1)
        y1 = center_y + center_b * math.sin(angle1)
        x2 = center_x + center_a * math.cos(angle2)
        y2 = center_y + center_b * math.sin(angle2)

        draw.line([(x1, y1), (x2, y2)], fill=(100, 100, 100), width=2)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save
    img.save(filename)
    print(f"✓ Track image created: {filename}")
    print(f"  Dimensions: {width}x{height}")
    print(f"  Track type: Oval (smooth elliptical loop)")
    print(f"  Drivable area: White interior between black boundaries")

if __name__ == "__main__":
    create_oval_track()
