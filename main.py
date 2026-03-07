#!/usr/bin/env python3
"""
TrackLearnerAI - Main entry point
AI agents learning to race using NEAT algorithm
"""

import pygame
import sys
import argparse
from src.core import SimulationEngine


def main():
    """Main entry point for TrackLearnerAI"""

    parser = argparse.ArgumentParser(description="TrackLearnerAI - Learn racing with NEAT")
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations to train (default: 50)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=30,
        help="Population size (default: 30)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Max steps per generation (default: 10000)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS (default: 30)",
    )
    parser.add_argument(
        "--track",
        default="assets/tracks/track1.png",
        help="Path to track PNG (default: assets/tracks/track1.png)",
    )

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  TrackLearnerAI v1.0 - AI Racing with NEAT")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Generations: {args.generations}")
    print(f"  Population: {args.population}")
    print(f"  Max steps/gen: {args.steps}")
    print(f"  Target FPS: {args.fps}")
    print(f"  Track: {args.track}")
    print()
    print("Controls:")
    print("  SPACE       - Toggle sensor visualization")
    print("  P           - Pause/Resume")
    print("  R           - Reset (in best lap mode)")
    print("  UP/DOWN     - Speed up/down simulation")
    print("  Q or close  - Quit")
    print()
    print("=" * 70)
    print()

    # Initialize Pygame
    pygame.init()

    try:
        # Create simulation engine
        engine = SimulationEngine(
            track_path=args.track,
            neat_config="config/config-feedforward.txt",
            pop_size=args.population,
            max_generations=args.generations,
            max_steps_per_generation=args.steps,
        )

        # Main simulation loop
        clock = pygame.time.Clock()
        frame_count = 0
        speed_multiplier = 1

        print("Starting simulation...")
        print()

        while engine.is_running():
            # Handle events
            if not engine.handle_events():
                break

            # Update simulation
            for _ in range(int(speed_multiplier)):
                engine.update()

            # Render
            engine.render()

            # Frame rate control
            clock.tick(args.fps)
            frame_count += 1

            # Debug info every 30 frames
            if frame_count % 30 == 0:
                state = engine.get_state()
                gen = engine.get_generation()
                fitness = engine.get_best_fitness()
                if state == "training":
                    sys.stdout.write(
                        f"\rGeneration: {gen:3d} | Best Fitness: {fitness:8.1f}"
                    )
                    sys.stdout.flush()

        print("\n")
        print("=" * 70)
        print("Simulation ended")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        pygame.quit()
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
