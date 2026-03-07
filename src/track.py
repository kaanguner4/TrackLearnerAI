"""
Track management module
Handles track loading, collision masks, and metadata
"""

import pygame
import os
from pathlib import Path


class Track:
    """Represents a racing track with collision detection"""

    def __init__(self, track_path="assets/tracks/track1.png", sensor_range=500):
        """
        Initialize track from PNG file

        Args:
            track_path: Path to track PNG image
            sensor_range: Maximum sensor range in pixels (for AI)
        """
        self.track_path = track_path
        self.sensor_range = sensor_range

        # Load track image
        if not os.path.exists(track_path):
            raise FileNotFoundError(f"Track file not found: {track_path}")

        self.image = pygame.image.load(track_path)
        # Use convert_alpha() for better compatibility, falls back to plain load if needed
        try:
            self.image = self.image.convert_alpha()
        except pygame.error:
            pass  # Display mode not set yet, will be converted after display init

        self.width, self.height = self.image.get_size()
        print(f"✓ Track loaded: {track_path} ({self.width}x{self.height})")

        # Create collision mask (black pixels = walls/track boundaries)
        self.mask = pygame.mask.from_surface(self.image)

        # Default start position (top-center of track, on the white area)
        # This should be on the drivable area (white), not on the black boundaries
        self.start_x = self.width // 2
        self.start_y = self.height // 4  # Upper portion of track
        self.start_heading = 0  # Radians, 0 = facing right

        # Verify start position is valid (on white area)
        if not self.is_on_track(self.start_x, self.start_y):
            print(
                f"⚠ Start position ({self.start_x}, {self.start_y}) is off track, adjusting..."
            )
            self._find_valid_start_position()

    def _find_valid_start_position(self):
        """Find a valid starting position on the track"""
        # Search for a white pixel to start from (drivable area)
        # Check middle-left area of the track
        for y in range(self.height // 3, 2 * self.height // 3):
            for x in range(self.width // 3, 2 * self.width // 3):
                if self.is_on_track(x, y):
                    self.start_x = x
                    self.start_y = y
                    print(f"✓ Valid start position found: ({x}, {y})")
                    return

        # Fallback: search entire image
        print("⚠ Searching entire track for valid start position...")
        for y in range(0, self.height, 10):
            for x in range(0, self.width, 10):
                if self.is_on_track(x, y):
                    self.start_x = x
                    self.start_y = y
                    print(f"✓ Valid start position found at: ({x}, {y})")
                    return

        print("⚠ Warning: No valid start position found. Using checkpoint area.")
        self.start_x = self.width // 2
        self.start_y = self.height // 2

    def is_on_track(self, x, y):
        """
        Check if a position is on the track (white/drivable area)

        Args:
            x, y: Pixel coordinates

        Returns:
            True if on white area (drivable), False if on black (wall) or out of bounds
        """
        # Bounds check
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False

        # Check mask: False = drivable (white), True = wall (black)
        # The black pixels create the mask, so inverted logic
        try:
            return not self.mask.get_at((int(x), int(y)))
        except (IndexError, ValueError):
            return False

    def get_start_position(self):
        """Return (x, y) starting position"""
        return (self.start_x, self.start_y)

    def get_start_heading(self):
        """Return starting heading in radians"""
        return self.start_heading

    def get_size(self):
        """Return (width, height) of track"""
        return (self.width, self.height)

    def get_sensor_range(self):
        """Return maximum sensor range in pixels"""
        return self.sensor_range

    def render(self, surface):
        """
        Draw track on given surface

        Args:
            surface: Pygame surface to draw on
        """
        surface.blit(self.image, (0, 0))

    def get_surface(self):
        """Return the track image surface (for rendering)"""
        return self.image

    def get_mask(self):
        """Return the collision mask"""
        return self.mask
