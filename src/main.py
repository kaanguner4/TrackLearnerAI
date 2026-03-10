import pygame
import sys
import os
import neat
from track_env import TrackEnvironment
from agent import Agent

# Renkler
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
PRUPLE = (128, 0, 128)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
BLUE = (0, 0, 255)


# GLOBAL DEĞİŞKENLER
env = None
screen = None
virtual_surface = None
clock = None
window_width = 0
window_height = 0
track_image = None
hud_font = None
checkpoint_font = None
crash_font = None
generation = 0


def evol_genomes(genomes, config):
    global env, screen, virtual_surface, clock, window_width, window_height, track_image, hud_font, checkpoint_font, crash_font, generation
    generation += 1

    nets = []
    agents = []
    ge = []

    # Her bir genom (DNA) için bir sinir ağı ve bir Ajan (Araba) oluşturuyoruz
    start_position = (env.finish_line['center'][0], env.finish_line['center'][1])
    for genome_id, genome in genomes:
        genome.fitness = 0  # Başlangıçta her genomun fitness'ını sıfırla
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        agents.append(Agent(start_position, env))
        ge.append(genome)


    # Ajanların sonsuza kadar kendi etrafında dönmesini engellemek için zamanlayıcılar
    frames_since_last_checkpoint = [0] * len(agents)

    running = True
    while running and len(agents) > 0:
        clock.tick(60)  # FPS sınırla

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Ekrandaki aktif ajan sayısını bul
        alive_agents = 0

        for i, agent in enumerate(agents):
            if not agent.is_alive:
                continue

            alive_agents += 1
            frames_since_last_checkpoint[i] += 1

            # 1. Sensör verilerini al ve Sinir Ağına ver
            inputs = agent.get_data()
            output = nets[i].activate(inputs)

            # 2. Sinir Ağı çıktısına göre hareket et (0.5 eşik değeri)
            # output[0]: Gaz, output[1]: Fren, output[2]: Sol, output[3]: Sağ
            if output[0] > 0.5:
                agent.speed += agent.acceleration
            elif output[1] > 0.5:
                agent.speed -= agent.acceleration
            else:
                if agent.speed > 0:
                    agent.speed -= 0.1
                elif agent.speed < 0:
                    agent.speed += 0.1

            agent.speed = max(-agent.max_speed / 2, min(agent.speed, agent.max_speed))

            if abs(agent.speed) > 0.1:
                if output[2] > 0.5:
                    agent.angle -= agent.turn_speed
                if output[3] > 0.5:
                    agent.angle += agent.turn_speed

            # 3. Ajanı güncelle
            agent.update()

            # 4. Checkpoint kontrolü ve fitness güncellemesi (look-ahead ile)
            look_ahead = 5
            max_check = min(agent.current_checkpoint + look_ahead, len(env.checkpoints))
            for cp_idx in range(agent.current_checkpoint, max_check):
                if env.is_checkpoint_passed(agent.prev_x, agent.prev_y, agent.x, agent.y, cp_idx):
                    # Atlanan checkpoint'ler için de puan ver
                    checkpoints_passed = cp_idx - agent.current_checkpoint + 1
                    ge[i].fitness += 100 * checkpoints_passed
                    agent.current_checkpoint = cp_idx + 1
                    frames_since_last_checkpoint[i] = 0
                    break

            # 5. Mesafe bazlı sürekli fitness ödülü
            if agent.speed > 0:
                ge[i].fitness += agent.speed * 0.1

            # 6. Zaman aşımı kontrolü: Eğer ajan uzun süre checkpoint geçemezse öldür
            if frames_since_last_checkpoint[i] > 300:  # ~5 saniye (60 FPS)
                agent.is_alive = False
                ge[i].fitness -= 50

        # Eğer yaşayan hiç ajan kalmadıysa bu jenerasyonu bitir
        if alive_agents == 0:
            break

        # --- Çizim Aşaması (frame başına bir kez) ---
        virtual_surface.blit(track_image, (0, 0))

        for idx, cp in enumerate(env.checkpoints):
            pygame.draw.line(virtual_surface, GREEN, cp['p1'], cp['p2'], 2)
            num_text = checkpoint_font.render(str(idx), True, BLACK)
            virtual_surface.blit(num_text, (cp['center'][0] + 6, cp['center'][1] - 8))

        for agent in agents:
            if agent.is_alive:
                agent.draw(virtual_surface)

        # HUD (Head-Up Display) çizimi
        hud_bg = pygame.Surface((380, 200))
        hud_bg.set_alpha(180)
        hud_bg.fill(BLACK)
        virtual_surface.blit(hud_bg, (10, 10))

        hud_texts = [
            f"FPS: {int(clock.get_fps())}",
            f"Jenerasyon: {generation}",
            f"En İyi Fitness: {max(ge, key=lambda g: g.fitness).fitness:.2f}",
            f"Ortalama Fitness: {sum(g.fitness for g in ge) / len(ge):.2f}",
            f"Hayatta Kalan Ajanlar: {alive_agents} / {len(agents)}"
        ]

        for idx, text in enumerate(hud_texts):
            rendered_text = hud_font.render(text, True, WHITE)
            virtual_surface.blit(rendered_text, (20, 20 + (idx * 40)))

        # Ekrana Yansıtma
        scaled_surface = pygame.transform.smoothscale(virtual_surface, (window_width, window_height))
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

def run_neat(config_path):
    # NEAT algoritması için konfigürasyon dosyasını yükle
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # NEAT algoritması için bir popülasyon oluştur
    p = neat.Population(config)

    # Terminale NEAT ilerlemesini göstermek için raporlayıcılar ekle
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Evrim sürecini başlat
    print("Evrim başlıyor...")
    winner = p.run(evol_genomes, 150)  # 150 jenerasyon

    print(f"Evrim tamamlandı! En iyi genomun ve fitness değeri:\n {winner} / {winner.fitness:.2f}")

def setup_pygame_and_environment():
    global env, screen, virtual_surface, clock, window_width, window_height, track_image, hud_font, checkpoint_font
    
    pygame.init()
    
    track_path = "assets/tracks/track1.png"
    env = TrackEnvironment(track_path)
    
    infoObject = pygame.display.Info()
    monitor_w, monitor_h = infoObject.current_w, infoObject.current_h
    max_window_w, max_window_h = int(monitor_w * 0.85), int(monitor_h * 0.85)
    
    aspect_ratio = env.width / env.height
    if env.width > max_window_w or env.height > max_window_h:
        if (max_window_w / aspect_ratio) <= max_window_h:
            window_width = max_window_w
            window_height = int(max_window_w / aspect_ratio)
        else:
            window_height = max_window_h
            window_width = int(max_window_h * aspect_ratio)
    else:
        window_width = env.width
        window_height = env.height
        
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("TrackLearnerAI - NeuroEvolution (NEAT)")
    
    virtual_surface = pygame.Surface((env.width, env.height))
    clock = pygame.time.Clock()
    track_image = pygame.image.load(track_path).convert()

    hud_font = pygame.font.SysFont(None, 36)
    checkpoint_font = pygame.font.SysFont(None, 20)

if __name__ == "__main__":
    # 1. Pygame ve Çevreyi Hazırla
    setup_pygame_and_environment()

    # 2. config dosyasının yolunu bul (Ana dizinde olduğunu varsayıyoruz)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, '..', 'config-feedforward.txt')

    # 3. NEAT algoritmasını çalıştır
    run_neat(config_path)