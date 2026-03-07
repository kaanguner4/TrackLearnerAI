"""
Track management module
Handles track loading, collision masks, and metadata
"""

import math
import os
import pygame
import numpy as np


class Track:
    """Represents a racing track with collision detection"""

    WALL_COLOR = (0, 0, 0, 255)
    WALL_TOLERANCE = (70, 70, 70, 255)
    CHECKPOINT_COLOR = (56, 182, 255, 255)
    CHECKPOINT_TOLERANCE = (120, 90, 80, 255)
    START_FINISH_COLOR = (255, 49, 49, 255)
    START_FINISH_TOLERANCE = (60, 120, 120, 255)

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

        # Build wall mask from dark pixels. Cars must avoid black boundary lines.
        self.mask = pygame.mask.from_threshold(
            self.image,
            self.WALL_COLOR,
            self.WALL_TOLERANCE,
        )

        self.start_finish_mask = None
        self.start_finish_center = None
        self.start_finish_rect = None
        self.checkpoint_centers = []
        self.checkpoint_index_map = np.full(
            (self.height, self.width), -1, dtype=np.int16
        )

        self.clockwise_vector = (1.0, 0.0)

        # Parse visual guides from track image (red start/finish and blue checkpoints).
        self._extract_colored_guides()

        # Default start setup (fallback).
        self.start_x = self.width // 2
        self.start_y = self.height // 4
        self.start_heading = 0.0
        self._configure_start_position()

        # Verify start position is valid (on white area)
        if not self.is_on_track(self.start_x, self.start_y):
            print(
                f"⚠ Start position ({self.start_x}, {self.start_y}) is off track, adjusting..."
            )
            self._find_valid_start_position()

    def _extract_colored_guides(self):
        """Extract red start/finish and ordered blue checkpoints from track image."""
        self.start_finish_mask = pygame.mask.from_threshold(
            self.image,
            self.START_FINISH_COLOR,
            self.START_FINISH_TOLERANCE,
        )
        self.checkpoint_mask = pygame.mask.from_threshold(
            self.image,
            self.CHECKPOINT_COLOR,
            self.CHECKPOINT_TOLERANCE,
        )

        red_components = self.start_finish_mask.connected_components(40)
        if red_components:
            red_component = max(red_components, key=lambda component: component.count())
            self.start_finish_center = red_component.centroid()
            bounding_rects = red_component.get_bounding_rects()
            if bounding_rects:
                self.start_finish_rect = bounding_rects[0]
            print(
                "✓ Start/finish strip detected at "
                f"({self.start_finish_center[0]}, {self.start_finish_center[1]})"
            )
        else:
            print("⚠ Red start/finish strip not found, using fallback start placement.")

        blue_components = self.checkpoint_mask.connected_components(80)
        if not blue_components:
            print("⚠ Blue checkpoints not found on track image.")
            return

        raw_centers = [component.centroid() for component in blue_components]
        if self.start_finish_center is None:
            start_reference = (self.width // 2, self.height // 4)
        else:
            start_reference = self.start_finish_center

        heading_hint = self._guess_clockwise_vector_from_start(start_reference)
        ordered_indices = self._build_checkpoint_order(
            raw_centers, start_reference, heading_hint
        )

        ordered_components = [blue_components[i] for i in ordered_indices]
        self.checkpoint_centers = [raw_centers[i] for i in ordered_indices]
        self._build_checkpoint_index_map(ordered_components)
        print(f"✓ Blue checkpoints parsed: {len(self.checkpoint_centers)}")

    def _build_checkpoint_order(self, centers, start_reference, heading_hint):
        """Order checkpoint line components so index progression is clockwise."""
        if len(centers) <= 1:
            return list(range(len(centers)))

        start_idx = min(
            range(len(centers)),
            key=lambda idx: self._distance(centers[idx], start_reference),
        )

        order = [start_idx]
        remaining = set(range(len(centers)))
        remaining.remove(start_idx)
        previous_direction = heading_hint
        current = start_idx

        while remaining:
            nearest_candidates = sorted(
                remaining,
                key=lambda idx: self._distance(centers[current], centers[idx]),
            )[:8]

            next_idx = max(
                nearest_candidates,
                key=lambda idx: self._checkpoint_transition_score(
                    centers[current], centers[idx], previous_direction
                ),
            )

            move_vector = (
                centers[next_idx][0] - centers[current][0],
                centers[next_idx][1] - centers[current][1],
            )
            move_norm = math.hypot(move_vector[0], move_vector[1])
            if move_norm > 1e-6:
                previous_direction = (
                    move_vector[0] / move_norm,
                    move_vector[1] / move_norm,
                )

            order.append(next_idx)
            remaining.remove(next_idx)
            current = next_idx

        order = self._orient_order_with_heading(order, centers, heading_hint)
        if not self._is_checkpoint_order_reasonable(order, centers):
            order = self._fallback_angle_order(centers, start_idx, heading_hint)

        return order

    def _checkpoint_transition_score(self, from_point, to_point, previous_direction):
        """Score transition between checkpoints: close and forward-aligned is best."""
        dx = to_point[0] - from_point[0]
        dy = to_point[1] - from_point[1]
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            return -1e9

        direction = (dx / distance, dy / distance)
        alignment = (
            direction[0] * previous_direction[0] + direction[1] * previous_direction[1]
        )
        return alignment * 120.0 - distance

    def _orient_order_with_heading(self, order, centers, heading_hint):
        """Choose checkpoint order direction that matches clockwise heading hint."""
        if len(order) < 2:
            return order

        reversed_order = [order[0]] + list(reversed(order[1:]))
        forward_alignment = self._first_step_alignment(order, centers, heading_hint)
        reverse_alignment = self._first_step_alignment(
            reversed_order, centers, heading_hint
        )
        return order if forward_alignment >= reverse_alignment else reversed_order

    def _first_step_alignment(self, order, centers, heading_hint):
        """Compute alignment of first checkpoint transition with desired heading."""
        if len(order) < 2:
            return -1.0

        start_point = centers[order[0]]
        next_point = centers[order[1]]
        dx = next_point[0] - start_point[0]
        dy = next_point[1] - start_point[1]
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            return -1.0

        step_direction = (dx / distance, dy / distance)
        return step_direction[0] * heading_hint[0] + step_direction[1] * heading_hint[1]

    def _is_checkpoint_order_reasonable(self, order, centers):
        """Basic sanity check for checkpoint traversal order."""
        if len(order) < 3:
            return True

        segment_lengths = []
        for index in range(len(order)):
            current_idx = order[index]
            next_idx = order[(index + 1) % len(order)]
            segment_lengths.append(
                self._distance(centers[current_idx], centers[next_idx])
            )

        median_length = float(np.median(segment_lengths))
        if median_length < 1e-6:
            return False

        return max(segment_lengths) <= median_length * 4.0

    def _fallback_angle_order(self, centers, start_idx, heading_hint):
        """Fallback ordering based on global angle around checkpoint centroid."""
        center_x = sum(point[0] for point in centers) / len(centers)
        center_y = sum(point[1] for point in centers) / len(centers)

        indexed = []
        for idx, (x, y) in enumerate(centers):
            angle = math.atan2(y - center_y, x - center_x)
            indexed.append((idx, angle))

        clockwise = sorted(indexed, key=lambda item: item[1], reverse=True)
        order = [idx for idx, _ in clockwise]

        if start_idx in order:
            pivot = order.index(start_idx)
            order = order[pivot:] + order[:pivot]

        return self._orient_order_with_heading(order, centers, heading_hint)

    def _build_checkpoint_index_map(self, ordered_components):
        """Build fast lookup map from pixel position to checkpoint index."""
        self.checkpoint_index_map.fill(-1)

        for checkpoint_idx, component in enumerate(ordered_components):
            for rect in component.get_bounding_rects():
                for y in range(rect.y, rect.y + rect.h):
                    for x in range(rect.x, rect.x + rect.w):
                        if component.get_at((x, y)):
                            self.checkpoint_index_map[y, x] = checkpoint_idx

    def _configure_start_position(self):
        """Set start position from red strip and heading to clockwise direction."""
        if self.start_finish_center is None:
            self._configure_clockwise_start()
            return

        self.start_x, self.start_y = self.start_finish_center
        self.clockwise_vector = self._guess_clockwise_vector_from_start(
            (self.start_x, self.start_y)
        )

        # Refine heading with first checkpoint transition direction.
        if len(self.checkpoint_centers) >= 2:
            checkpoint_a = self.checkpoint_centers[0]
            checkpoint_b = self.checkpoint_centers[1]
            dx = checkpoint_b[0] - checkpoint_a[0]
            dy = checkpoint_b[1] - checkpoint_a[1]
            norm = math.hypot(dx, dy)
            if norm > 1e-6:
                candidate = (dx / norm, dy / norm)
                alignment = (
                    candidate[0] * self.clockwise_vector[0]
                    + candidate[1] * self.clockwise_vector[1]
                )
                if alignment < 0:
                    candidate = (-candidate[0], -candidate[1])
                self.clockwise_vector = candidate

        self.start_heading = math.atan2(
            self.clockwise_vector[1], self.clockwise_vector[0]
        )
        print(
            f"✓ Start configured from red strip: ({self.start_x}, {self.start_y}), "
            "heading: clockwise"
        )

    def _clockwise_vector_at(self, x, y):
        """Return clockwise tangent vector around track centroid at a point."""
        center_x = self.width * 0.5
        center_y = self.height * 0.5
        radial_x = x - center_x
        radial_y = y - center_y

        tangent_x = -radial_y
        tangent_y = radial_x
        norm = math.hypot(tangent_x, tangent_y)
        if norm < 1e-6:
            return (1.0, 0.0)

        return (tangent_x / norm, tangent_y / norm)

    def _guess_clockwise_vector_from_start(self, start_point):
        """Estimate clockwise forward vector from start/finish strip geometry."""
        start_x, start_y = start_point
        if self.start_finish_rect is None:
            return self._clockwise_vector_at(start_x, start_y)

        if self.start_finish_rect.h >= self.start_finish_rect.w:
            # Vertical strip => travel horizontally.
            return (1.0, 0.0) if start_y <= (self.height / 2) else (-1.0, 0.0)

        # Horizontal strip => travel vertically.
        return (0.0, -1.0) if start_x <= (self.width / 2) else (0.0, 1.0)

    def _distance(self, point_a, point_b):
        """Euclidean distance helper."""
        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])

    def _find_valid_start_position(self):
        """Find a valid starting position on the track"""
        # Search for a white pixel to start from (drivable area)
        # Check middle-left area of the track
        for y in range(self.height // 3, 2 * self.height // 3):
            for x in range(self.width // 3, 2 * self.width // 3):
                if self.is_on_track(x, y):
                    self.start_x = x
                    self.start_y = y
                    self.clockwise_vector = self._clockwise_vector_at(x, y)
                    self.start_heading = math.atan2(
                        self.clockwise_vector[1], self.clockwise_vector[0]
                    )
                    print(f"✓ Valid start position found: ({x}, {y})")
                    return

        # Fallback: search entire image
        print("⚠ Searching entire track for valid start position...")
        for y in range(0, self.height, 10):
            for x in range(0, self.width, 10):
                if self.is_on_track(x, y):
                    self.start_x = x
                    self.start_y = y
                    self.clockwise_vector = self._clockwise_vector_at(x, y)
                    self.start_heading = math.atan2(
                        self.clockwise_vector[1], self.clockwise_vector[0]
                    )
                    print(f"✓ Valid start position found at: ({x}, {y})")
                    return

        print("⚠ Warning: No valid start position found. Using checkpoint area.")
        self.start_x = self.width // 2
        self.start_y = self.height // 2
        self.clockwise_vector = self._clockwise_vector_at(self.start_x, self.start_y)
        self.start_heading = math.atan2(self.clockwise_vector[1], self.clockwise_vector[0])

    def _configure_clockwise_start(self):
        """Find a lane center near the top of the map and set clockwise heading."""
        # Prefer the upper straight segment and keep heading to the right (clockwise).
        top_search_limit = max(20, int(self.height * 0.45))
        min_lane_gap = max(10, int(self.height * 0.02))
        x_candidates = [
            int(self.width * r) for r in (0.5, 0.55, 0.45, 0.6, 0.4, 0.65, 0.35)
        ]

        for x in x_candidates:
            wall_ys = [
                y for y in range(top_search_limit) if self.mask.get_at((x, y))
            ]
            if len(wall_ys) < 2:
                continue

            # Group consecutive y values into wall segments.
            segments = []
            seg_start = wall_ys[0]
            prev = wall_ys[0]
            for y in wall_ys[1:]:
                if y == prev + 1:
                    prev = y
                    continue
                segments.append((seg_start, prev))
                seg_start = y
                prev = y
            segments.append((seg_start, prev))

            if len(segments) < 2:
                continue

            outer_segment = segments[0]
            inner_segment = None
            for segment in segments[1:]:
                if segment[0] - outer_segment[1] >= min_lane_gap:
                    inner_segment = segment
                    break

            if inner_segment is None:
                continue

            start_y = (outer_segment[1] + inner_segment[0]) // 2
            if self.is_on_track(x, start_y):
                self.start_x = x
                self.start_y = start_y
                self.clockwise_vector = self._clockwise_vector_at(x, start_y)
                self.start_heading = math.atan2(
                    self.clockwise_vector[1], self.clockwise_vector[0]
                )
                print(
                    f"✓ Start configured: ({self.start_x}, {self.start_y}), "
                    "heading: clockwise"
                )
                return

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

    def get_clockwise_vector(self):
        """Return unit vector representing clockwise forward direction."""
        return self.clockwise_vector

    def get_checkpoint_count(self):
        """Return number of parsed blue checkpoints."""
        return len(self.checkpoint_centers)

    def get_checkpoint_centers(self):
        """Return ordered checkpoint center coordinates."""
        return self.checkpoint_centers.copy()

    def get_checkpoint_center(self, checkpoint_idx):
        """Return center of checkpoint by index, or None if invalid."""
        if checkpoint_idx < 0 or checkpoint_idx >= len(self.checkpoint_centers):
            return None
        return self.checkpoint_centers[checkpoint_idx]

    def get_checkpoint_index_at(self, x, y):
        """Return checkpoint index at given pixel, or -1 if none."""
        x = int(x)
        y = int(y)
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return -1
        return int(self.checkpoint_index_map[y, x])

    def get_crossed_checkpoints(self, start_pos, end_pos, step_size=2):
        """
        Return checkpoint indices crossed by segment from start_pos to end_pos.
        """
        if len(self.checkpoint_centers) == 0:
            return []

        start_x, start_y = start_pos
        end_x, end_y = end_pos
        dx = end_x - start_x
        dy = end_y - start_y
        distance = math.hypot(dx, dy)
        steps = max(1, int(distance / max(1, step_size)))

        crossed = []
        last_idx = -1

        for step in range(steps + 1):
            t = step / steps
            sample_x = start_x + dx * t
            sample_y = start_y + dy * t
            checkpoint_idx = self.get_checkpoint_index_at(sample_x, sample_y)
            if checkpoint_idx >= 0 and checkpoint_idx != last_idx:
                crossed.append(checkpoint_idx)
                last_idx = checkpoint_idx
            elif checkpoint_idx < 0:
                last_idx = -1

        return crossed

    def is_on_start_finish(self, x, y):
        """Return whether a position lies on the red start/finish strip."""
        if self.start_finish_mask is None:
            return False
        x = int(x)
        y = int(y)
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False
        return bool(self.start_finish_mask.get_at((x, y)))

    def segment_crosses_start_finish(
        self, start_pos, end_pos, step_size=2, require_clockwise=False
    ):
        """Return whether a movement segment crosses red start/finish strip."""
        if self.start_finish_mask is None:
            return False

        start_x, start_y = start_pos
        end_x, end_y = end_pos
        move_x = end_x - start_x
        move_y = end_y - start_y
        distance = math.hypot(move_x, move_y)
        if distance < 1e-6:
            return False

        if require_clockwise:
            move_direction = (move_x / distance, move_y / distance)
            alignment = (
                move_direction[0] * self.clockwise_vector[0]
                + move_direction[1] * self.clockwise_vector[1]
            )
            if alignment <= 0:
                return False

        steps = max(1, int(distance / max(1, step_size)))
        for step in range(steps + 1):
            t = step / steps
            sample_x = start_x + move_x * t
            sample_y = start_y + move_y * t
            if self.is_on_start_finish(sample_x, sample_y):
                return True

        return False

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
