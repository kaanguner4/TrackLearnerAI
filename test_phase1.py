#!/usr/bin/env python3
"""
Phase 1 Test: Single vehicle with debug rendering
Tests: track loading, physics, sensors, collisions, rendering
"""

import pygame
import sys
import math

from src.track import Track
from src.car import Car
from src.render import Renderer


def main():
    """Phase 1 test: manual control of single vehicle"""

    # Initialize Pygame
    pygame.init()
    pygame.display.init()
    pygame.font.init()

    print("=" * 60)
    print("TrackLearnerAI - Phase 1 Test: Single Vehicle")
    print("=" * 60)
    print()
    print("Controls:")
    print("  Arrow Keys: Steer (LEFT/RIGHT) and Throttle (UP/DOWN)")
    print("  SPACE: Toggle sensor visualization")
    print("  P: Pause")
    print("  R: Reset car")
    print("  Q: Quit")
    print()

    # Load track
    try:
        track = Track("assets/tracks/track1.png", sensor_range=500)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Get track info
    width, height = track.get_size()
    start_x, start_y = track.get_start_position()
    start_heading = track.get_start_heading()

    print(f"Track loaded: {width}x{height} pixels")
    print(f"Start position: ({start_x}, {start_y}), heading: {start_heading}")
    print()

    # Create renderer
    renderer = Renderer(width, height)
    renderer.set_caption("TrackLearnerAI - Phase 1: Manual Test")

    # Create single test vehicle
    test_car = Car(0, track, start_x, start_y, start_heading)
    print(f"✓ Test car initialized at ({start_x}, {start_y})")
    print(f"✓ Initial sensors: {[f'{s:.2f}' for s in test_car.get_sensor_readings()]}")
    print()

    # Test state
    clock = pygame.time.Clock()
    fps_target = 30
    paused = False
    running = True
    steps = 0

    # Manual input states
    steering = 0
    throttle = 0

    print("Starting simulation... (press Q to quit)")
    print()

    while running:
        dt = 1.0  # Fixed timestep

        # Handle events
        events = renderer.handle_events()

        if events["quit"]:
            print("\nQuitting...")
            running = False
        if events["sensors_toggled"]:
            state = renderer.toggle_sensors()
            print(f"Sensor visualization: {'ON' if state else 'OFF'}")
        if events["reset"]:
            test_car.reset(start_x, start_y, start_heading)
            print("Car reset to start position")

        # Handle keyboard input
        keys = pygame.key.get_pressed()
        steering = 0
        throttle = 0

        if keys[pygame.K_LEFT]:
            steering = -1
        elif keys[pygame.K_RIGHT]:
            steering = 1

        if keys[pygame.K_UP]:
            throttle = 1
        elif keys[pygame.K_DOWN]:
            throttle = -0.5

        if keys[pygame.K_q]:
            running = False

        if keys[pygame.K_p]:
            paused = not paused

        # Update car if not paused
        if not paused:
            test_car.update(steering, throttle, dt)
            steps += 1

        # Render
        alive_count = 1 if test_car.is_alive() else 0
        max_fitness = test_car.get_average_speed() * 0.5

        # Print stats every 30 frames
        if steps % 30 == 0 and steps > 0:
            pos = test_car.get_position()
            vel = test_car.get_velocity()
            sensors = test_car.get_sensor_readings()
            print(f"Step {steps:4d} | Pos: ({pos[0]:6.1f}, {pos[1]:6.1f}) | "
                  f"Vel: {vel:6.1f} | Avg Speed: {test_car.get_average_speed():6.1f} | "
                  f"Alive: {alive_count} | Sensors: {[f'{s:.2f}' for s in sensors]}")

        renderer.render_frame(
            track,
            [test_car],
            generation=0,
            max_fitness=max_fitness,
            alive_count=alive_count,
            best_lap_time=steps / fps_target,
        )

        clock.tick(fps_target)

    print()
    print("=" * 60)
    print(f"Test completed. Ran for {steps} steps")
    print(f"Final position: {test_car.get_position()}")
    print(f"Final velocity: {test_car.get_velocity():.2f}")
    print(f"Average speed: {test_car.get_average_speed():.2f}")
    print(f"Car alive: {test_car.is_alive()}")
    print("=" * 60)

    pygame.quit()


if __name__ == "__main__":
    main()
