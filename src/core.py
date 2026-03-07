"""
Core simulation engine
Main game loop, state management, and training coordination
"""

import pygame
from src.track import Track
from src.car import Car
from src.ai_manager import AIManager, CarBrain
from src.render import Renderer
from src.utils import ProgressTracker, calculate_fitness
import neat


class SimulationState:
    """Simulation state machine"""

    TRAINING = "training"
    BEST_LAP = "best_lap"
    PAUSED = "paused"


class SimulationEngine:
    """
    Main simulation engine
    Coordinates track, cars, AI, and rendering
    """

    def __init__(
        self,
        track_path="assets/tracks/track1.png",
        neat_config="config/config-feedforward.txt",
        pop_size=30,
        max_generations=100,
        max_steps_per_generation=10000,
    ):
        """
        Initialize simulation engine

        Args:
            track_path: Path to track PNG
            neat_config: Path to NEAT config file
            pop_size: Population size
            max_generations: Training iterations
            max_steps_per_generation: Max steps per generation
        """
        # Create renderer first to initialize pygame display
        import pygame as pg
        temp_img = pg.image.load(track_path)
        temp_width, temp_height = temp_img.get_size()

        self.renderer = Renderer(temp_width, temp_height)

        # Now load track with display mode set
        self.track = Track(track_path, sensor_range=500)

        # AI and training
        self.ai_manager = AIManager(neat_config, pop_size)
        self.config = self.ai_manager.get_config()

        # Progress tracking
        self.progress_tracker = ProgressTracker(
            self.track.width, self.track.height, num_checkpoints=50
        )

        # Training parameters
        self.max_generations = max_generations
        self.max_steps_per_generation = max_steps_per_generation
        self.pop_size = pop_size

        # State
        self.state = SimulationState.TRAINING
        self.generation = 0
        self.step = 0
        self.paused = False
        self.running = True

        # Best lap tracking
        self.best_lap_time = None
        self.best_cars_data = {}  # Store best lap car trajectory

        # Initialize population
        self._init_population()

        print(f"✓ Simulation engine initialized")
        print(f"  Track: {self.track.width}x{self.track.height}")
        print(f"  Population: {self.pop_size}")
        print(f"  Max generations: {self.max_generations}")

    def _init_population(self):
        """Initialize car population from NEAT genomes"""
        start_x, start_y = self.track.get_start_position()
        start_heading = self.track.get_start_heading()

        self.cars = []
        for i in range(self.pop_size):
            car = Car(i, self.track, start_x, start_y, start_heading)
            self.cars.append(car)

    def update(self, dt=1.0):
        """
        Update simulation by one frame

        Args:
            dt: Delta time (default 1.0 frame)
        """
        if self.state == SimulationState.TRAINING and not self.paused:
            self._update_training(dt)
        elif self.state == SimulationState.BEST_LAP:
            self._update_best_lap(dt)

    def _update_training(self, dt):
        """Update training step"""
        # Get current genomes from NEAT population
        genomes = list(self.ai_manager.population.population.items())

        if not genomes or len(self.cars) == 0:
            return

        # Update each car with its neural network
        alive_count = 0

        for (genome_id, genome), car in zip(genomes, self.cars):
            if not car.is_alive():
                continue

            alive_count += 1

            # Create neural network for this genome
            net = neat.nn.FeedForwardNetwork.create(genome, self.config)
            brain = CarBrain(net)

            # Get sensor readings
            sensors = car.get_sensor_readings()

            # Compute network output (steering, throttle)
            steering, throttle = brain.forward(sensors)

            # Update car physics
            car.update(steering, throttle, dt)

            # Calculate fitness
            progress = self.progress_tracker.calculate_progress(car.car_id, car.get_position())
            avg_speed = car.get_average_speed()
            is_dead = not car.is_alive()

            fitness = calculate_fitness(progress, avg_speed, is_dead)
            genome.fitness = fitness

        self.step += 1

        # Check if generation is over
        if self.step >= self.max_steps_per_generation or alive_count == 0:
            self._end_generation()

    def _end_generation(self):
        """End current generation and advance to next"""
        # Record metrics
        genomes = list(self.ai_manager.population.population.items())

        if genomes:
            best_gen_fitness = max(genome.fitness for _, genome in genomes)
            self.ai_manager.best_fitness = max(
                self.ai_manager.best_fitness, best_gen_fitness
            )

            # Find best genome this generation
            for genome_id, genome in genomes:
                if genome.fitness == best_gen_fitness:
                    self.ai_manager.best_genome = genome
                    break

        # Advance NEAT generation
        self.generation += 1
        self.step = 0

        # Reset cars
        start_x, start_y = self.track.get_start_position()
        start_heading = self.track.get_start_heading()
        for car in self.cars:
            car.reset(start_x, start_y, start_heading)

        # Check if training is done
        if self.generation >= self.max_generations:
            print(f"\n✓ Training complete after {self.generation} generations")
            print(f"  Best fitness: {self.ai_manager.best_fitness:.2f}")

            # Save best genome
            self.ai_manager.save_best_genome()

            # Switch to best lap mode
            self._init_best_lap_mode()

    def _init_best_lap_mode(self):
        """Initialize best lap replay mode"""
        print("\n" + "=" * 60)
        print("BEST LAP MODE - Replaying best genome")
        print("=" * 60)

        best_brain = self.ai_manager.get_best_brain()

        if best_brain is None:
            print("Error: No best genome available")
            self.state = SimulationState.PAUSED
            return

        # Create single car for replay
        start_x, start_y = self.track.get_start_position()
        start_heading = self.track.get_start_heading()

        best_car = Car(999, self.track, start_x, start_y, start_heading)
        self.cars = [best_car]
        self.best_lap_brain = best_brain

        self.state = SimulationState.BEST_LAP
        self.step = 0
        self.best_lap_time = None

    def _update_best_lap(self, dt):
        """Update best lap replay"""
        if len(self.cars) == 0:
            return

        car = self.cars[0]

        if not car.is_alive():
            if self.best_lap_time is None:
                self.best_lap_time = self.step
                print(f"\n✓ Best lap completed: {self.best_lap_time} steps")
            return

        # Run best brain
        sensors = car.get_sensor_readings()
        steering, throttle = self.best_lap_brain.forward(sensors)

        car.update(steering, throttle, dt)
        self.step += 1

    def render(self):
        """Render current frame"""
        alive_count = sum(1 for car in self.cars if car.is_alive())

        self.renderer.render_frame(
            self.track,
            self.cars,
            generation=self.generation,
            max_fitness=self.ai_manager.best_fitness,
            alive_count=alive_count,
            best_lap_time=self.best_lap_time,
        )

    def handle_events(self):
        """
        Handle input events

        Returns:
            True if should continue, False if quit
        """
        events = self.renderer.handle_events()

        if events["quit"]:
            self.running = False
            return False

        if events["sensors_toggled"]:
            self.renderer.toggle_sensors()

        if events["paused"]:
            self.paused = not self.paused

        if events["reset"]:
            self._init_best_lap_mode()

        return True

    def is_running(self):
        """Check if simulation should continue"""
        return self.running

    def get_state(self):
        """Return current state"""
        return self.state

    def get_generation(self):
        """Return current generation"""
        return self.generation

    def get_best_fitness(self):
        """Return best fitness"""
        return self.ai_manager.best_fitness

    def step_training(self):
        """Run single training step"""
        self.update()
        self.render()
        return self.is_running()
