"""
AI Manager - NEAT population and training
Handles genome evaluation, fitness calculation, and evolution
"""

import pickle
import neat
import os
from src.utils import calculate_fitness, ProgressTracker


class CarBrain:
    """
    Neural network brain for a car
    Wraps a NEAT genome/network
    """

    def __init__(self, network):
        """
        Initialize car brain

        Args:
            network: neat.nn.FeedForwardNetwork
        """
        self.network = network

    def forward(self, sensor_inputs):
        """
        Get steering and throttle commands from sensor inputs

        Args:
            sensor_inputs: list of 5 normalized sensor values (0..1)

        Returns:
            (steering, throttle) tuple; steering in -1..1, throttle in 0.1..1
        """
        outputs = self.network.activate(sensor_inputs)
        steering = None
        throttle = None

        if len(outputs) >= 2:
            steering = max(-1.0, min(1.0, outputs[0]))
            # Keep cars moving forward so training consistently progresses clockwise.
            throttle = max(0.1, min(1.0, (outputs[1] + 1.0) / 2.0))
        else:
            steering = 0
            throttle = 0

        return steering, throttle


class AIManager:
    """
    Manages NEAT population for training
    Handles evaluation, fitness calculation, and model persistence
    """

    def __init__(self, config_path="config/config-feedforward.txt", pop_size=None):
        """
        Initialize AI manager with NEAT config

        Args:
            config_path: Path to NEAT config file
            pop_size: Population size (if None, uses config file value)
        """
        self.config_path = config_path

        # Load NEAT configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"NEAT config not found: {config_path}")

        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )

        # Override population size if specified
        if pop_size is not None:
            config.pop_size = pop_size

        self.pop_size = config.pop_size

        # Create population
        self.population = neat.Population(config)

        # Add reporters for progress tracking
        self.stats = neat.StatisticsReporter()
        self.population.add_reporter(self.stats)
        self.population.add_reporter(neat.StdOutReporter(False))

        # Training state
        self.generation = 0
        self.best_genome = None
        self.best_fitness = -float("inf")
        self.best_brain = None

        print(f"✓ NEAT population initialized (size: {self.pop_size})")

    def create_car_brains(self, cars):
        """
        Create and assign neural network brains to cars

        Args:
            cars: list of Car objects
            genomes: list of NEAT genomes for cars
            config: NEAT config object
        """
        genomes = list(self.population.population.values())

        brains = []
        for i, (genome_id, genome) in enumerate(genomes):
            if i < len(cars):
                net = neat.nn.FeedForwardNetwork.create(genome, self.population.config)
                brain = CarBrain(net)
                brains.append((cars[i], genome, brain))

        return brains

    def evaluate_genomes(self, genomes, cars, track, progress_tracker, max_steps=10000):
        """
        Evaluate fitness for all genomes

        Args:
            genomes: list of (genome_id, genome) tuples
            cars: list of Car objects
            track: Track object
            progress_tracker: ProgressTracker for monotonic progress
            max_steps: maximum simulation steps per car
        """
        for genome_id, genome in genomes:
            genome.fitness = 0.0

        # Run simulation for each car
        for step in range(max_steps):
            # Get alive cars and their corresponding genomes/brains
            alive_pairs = []

            for (gid, genome), car in zip(genomes, cars):
                if car.is_alive():
                    net = neat.nn.FeedForwardNetwork.create(
                        genome, self.population.config
                    )
                    brain = CarBrain(net)

                    # Get sensor readings and compute brain output
                    sensors = car.get_sensor_readings()
                    steering, throttle = brain.forward(sensors)

                    # Update car physics
                    previous_pos = car.get_position()
                    car.update(steering, throttle, dt=1.0)

                    # Update progress
                    progress = progress_tracker.calculate_progress(
                        car.car_id, car.get_position(), previous_pos
                    )

                    # Calculate fitness incrementally
                    avg_speed = car.get_average_speed()
                    is_dead = not car.is_alive()

                    fitness = calculate_fitness(progress, avg_speed, is_dead)
                    genome.fitness = fitness

            # Check if all cars are dead
            all_dead = all(not car.is_alive() for car in cars)
            if all_dead:
                break

        # Update best genome
        for genome_id, genome in genomes:
            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                self.best_genome = genome

    def advance_generation(self):
        """
        Run one generation of NEAT evolution

        Args:
            genomes: evaluation function that fills genome.fitness values

        Returns:
            leader_genome (best in this generation)
        """
        # This should be called by the population.run() method
        self.generation += 1

    def get_best_genome(self):
        """Return the best genome found so far"""
        return self.best_genome

    def get_best_brain(self):
        """Return network for best genome"""
        if self.best_genome is None:
            return None

        net = neat.nn.FeedForwardNetwork.create(self.best_genome, self.population.config)
        return CarBrain(net)

    def get_generation(self):
        """Return current generation number"""
        return self.generation

    def get_best_fitness(self):
        """Return best fitness achieved"""
        return self.best_fitness

    def save_best_genome(self, filepath="data/best_brains/best_genome.pkl"):
        """
        Save best genome to file

        Args:
            filepath: path to save to
        """
        if self.best_genome is None:
            print("No best genome to save")
            return False

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        try:
            with open(filepath, "wb") as f:
                pickle.dump(self.best_genome, f)
            print(f"✓ Best genome saved: {filepath}")
            return True
        except Exception as e:
            print(f"✗ Error saving genome: {e}")
            return False

    def load_best_genome(self, filepath="data/best_brains/best_genome.pkl"):
        """
        Load best genome from file

        Args:
            filepath: path to load from

        Returns:
            loaded genome or None if error
        """
        if not os.path.exists(filepath):
            print(f"Genome file not found: {filepath}")
            return None

        try:
            with open(filepath, "rb") as f:
                genome = pickle.load(f)
            print(f"✓ Genome loaded: {filepath}")
            self.best_genome = genome
            return genome
        except Exception as e:
            print(f"✗ Error loading genome: {e}")
            return None

    def get_config(self):
        """Return NEAT config object"""
        return self.population.config
