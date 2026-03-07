"""
Rendering module for visualization, HUD, and debug views
"""

import pygame
import math
from src.utils import Vector2


class Renderer:
    """Handles all rendering for the simulation"""

    def __init__(self, screen_width=1200, screen_height=700):
        """Initialize renderer with screen dimensions"""
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("TrackLearnerAI - Training")

        # Font for HUD
        self.small_font = pygame.font.Font(None, 24)
        self.medium_font = pygame.font.Font(None, 32)
        self.large_font = pygame.font.Font(None, 48)

        self.show_sensors = True  # Debug mode toggle
        self.space_hit_count = 0

    def render_frame(self, track, cars, generation=0, max_fitness=0, alive_count=0, best_lap_time=None):
        """
        Render complete frame

        Args:
            track: Track object
            cars: List of Car objects
            generation: Current generation number
            max_fitness: Best fitness so far
            alive_count: Number of alive cars
            best_lap_time: Best lap time if available
        """
        # Clear screen with white background
        self.screen.fill((255, 255, 255))

        # Draw track
        track_surface = track.get_surface()
        self.screen.blit(track_surface, (0, 0))

        # Draw cars
        for car in cars:
            self._draw_car(car)

            # Draw sensors if debug mode enabled
            if self.show_sensors and car.is_alive():
                self._draw_sensors(car)

        # Draw HUD
        self._draw_hud(generation, max_fitness, alive_count, best_lap_time)

        # Update display
        pygame.display.flip()

    def _draw_car(self, car):
        """Draw a single car on screen"""
        x, y = car.get_position()
        heading = car.get_heading()

        # Car size (pixels from center)
        size = 8

        # Calculate car corners (rotated rectangle)
        corners = [
            Vector2(-size, -size),
            Vector2(size, -size),
            Vector2(size, size),
            Vector2(-size, size),
        ]

        # Rotate corners by heading
        rotated_corners = []
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        for corner in corners:
            rx = corner.x * cos_h - corner.y * sin_h
            ry = corner.x * sin_h + corner.y * cos_h
            rotated_corners.append((x + rx, y + ry))

        # Colors
        if car.is_alive():
            color = (0, 100, 255)  # Blue for alive
        else:
            color = (200, 0, 0)  # Red for dead

        # Draw filled polygon
        pygame.draw.polygon(self.screen, color, rotated_corners)

        # Draw front indicator (line from center in heading direction)
        front_x = x + size * math.cos(heading)
        front_y = y + size * math.sin(heading)
        pygame.draw.line(self.screen, (0, 0, 0), (x, y), (front_x, front_y), 2)

    def _draw_sensors(self, car):
        """Draw raycasting sensor rays as debug visualization"""
        x, y = car.get_position()
        heading = car.get_heading()
        sensor_range = 500  # Match track sensor range

        sensor_angles = car.get_sensor_angles()
        sensor_values = car.get_sensor_readings()

        for i, angle_offset in enumerate(sensor_angles):
            ray_direction = heading + angle_offset
            distance = sensor_values[i] * sensor_range

            # Ray endpoint
            end_x = x + math.cos(ray_direction) * distance
            end_y = y + math.sin(ray_direction) * distance

            # Color: gradient based on distance (green = far, red = close)
            distance_ratio = sensor_values[i]
            r = int(255 * distance_ratio)
            g = int(255 * (1 - distance_ratio))
            color = (r, g, 0)

            # Draw ray (thin line)
            pygame.draw.line(self.screen, color, (x, y), (end_x, end_y), 1)

            # Draw endpoint circle
            pygame.draw.circle(self.screen, color, (int(end_x), int(end_y)), 2)

    def _draw_hud(self, generation, max_fitness, alive_count, best_lap_time):
        """Draw heads-up display with training info"""
        hud_bg_color = (50, 50, 50)
        hud_text_color = (255, 255, 255)
        hud_y_offset = 10

        # HUD background (semi-transparent rectangle at top)
        hud_height = 110
        pygame.draw.rect(
            self.screen, hud_bg_color, (0, 0, self.screen_width, hud_height)
        )

        # HUD text
        lines = [
            f"Generation: {generation}",
            f"Alive: {alive_count}",
            f"Max Fitness: {max_fitness:.1f}",
        ]

        if best_lap_time is not None:
            lines.append(f"Best Lap: {best_lap_time:.2f}s")

        lines.append("[SPACE] Toggle Sensors | [R] Reset | [P] Pause")

        for i, line in enumerate(lines):
            text_surface = self.small_font.render(line, True, hud_text_color)
            self.screen.blit(text_surface, (15, hud_y_offset + i * 20))

    def toggle_sensors(self):
        """Toggle sensor debug visualization"""
        self.show_sensors = not self.show_sensors
        return self.show_sensors

    def handle_events(self):
        """
        Handle pygame events

        Returns:
            dict with keys: 'quit', 'paused', 'reset', 'sensors_toggled'
        """
        result = {
            "quit": False,
            "paused": False,
            "reset": False,
            "sensors_toggled": False,
            "speed_multiplier": 1,
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                result["quit"] = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    result["sensors_toggled"] = True
                elif event.key == pygame.K_p:
                    result["paused"] = True
                elif event.key == pygame.K_r:
                    result["reset"] = True
                elif event.key == pygame.K_UP:
                    result["speed_multiplier"] = 2  # Speed up simulation
                elif event.key == pygame.K_DOWN:
                    result["speed_multiplier"] = 0.5  # Slow down

        return result

    def set_caption(self, title):
        """Set window caption"""
        pygame.display.set_caption(title)

    def get_screen(self):
        """Get pygame screen surface"""
        return self.screen
