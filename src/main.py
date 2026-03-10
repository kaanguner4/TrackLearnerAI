import pygame
import sys
import os
import glob
import math
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


def show_track_selector():
    pygame.init()

    # Pist dosyalarini bul ve sirala
    track_dir = os.path.join(os.path.dirname(__file__), '..', 'assets', 'tracks')
    track_files = sorted(glob.glob(os.path.join(track_dir, 'track*.png')))

    if not track_files:
        print("Hata: assets/tracks/ klasorunde pist bulunamadi!")
        sys.exit(1)

    # Pencere ayarlari
    padding = 30
    cols = 2
    rows = (len(track_files) + cols - 1) // cols
    thumb_w, thumb_h = 380, 280
    win_w = cols * thumb_w + (cols + 1) * padding
    win_h = rows * thumb_h + (rows + 1) * padding + 60

    selector_screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("TrackLearnerAI - Pist Sec")

    title_font = pygame.font.SysFont(None, 48)
    label_font = pygame.font.SysFont(None, 28)

    # Thumbnail'leri yukle
    thumbnails = []
    for path in track_files:
        img = pygame.image.load(path).convert()
        img = pygame.transform.smoothscale(img, (thumb_w, thumb_h))
        name = os.path.splitext(os.path.basename(path))[0].capitalize()
        thumbnails.append((img, name, path))

    selected_path = None
    while selected_path is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                for idx, (_, _, path) in enumerate(thumbnails):
                    col = idx % cols
                    row = idx // cols
                    x = padding + col * (thumb_w + padding)
                    y = 60 + padding + row * (thumb_h + padding)
                    if x <= mx <= x + thumb_w and y <= my <= y + thumb_h:
                        selected_path = path

        selector_screen.fill((30, 30, 30))

        title = title_font.render("Pist Sec", True, (255, 255, 255))
        selector_screen.blit(title, (win_w // 2 - title.get_width() // 2, 15))

        mx, my = pygame.mouse.get_pos()
        for idx, (img, name, path) in enumerate(thumbnails):
            col = idx % cols
            row = idx // cols
            x = padding + col * (thumb_w + padding)
            y = 60 + padding + row * (thumb_h + padding)

            hovered = x <= mx <= x + thumb_w and y <= my <= y + thumb_h
            border_color = (0, 200, 0) if hovered else (100, 100, 100)
            border_width = 3 if hovered else 1

            pygame.draw.rect(selector_screen, border_color, (x - border_width, y - border_width, thumb_w + border_width * 2, thumb_h + border_width * 2), border_width)
            selector_screen.blit(img, (x, y))

            label = label_font.render(name, True, (255, 255, 255))
            selector_screen.blit(label, (x + thumb_w // 2 - label.get_width() // 2, y + thumb_h + 5))

        pygame.display.flip()

    pygame.quit()
    return selected_path


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
    # Ajan ilk olarak son checkpoint'ten geçerse tek seferlik ceza vermek için
    wrong_way_penalized = [False] * len(agents)
    num_checkpoints = len(env.checkpoints)

    # Her ajan icin sonraki checkpoint'e olan baslangic mesafesini hesapla
    def _dist_to_checkpoint(ax, ay, cp_idx):
        if cp_idx < num_checkpoints:
            cx, cy = env.checkpoints[cp_idx]['center']
        else:
            cx, cy = env.finish_line['center']
        return math.hypot(cx - ax, cy - ay)

    prev_dist_to_cp = [_dist_to_checkpoint(agent.x, agent.y, 0) for agent in agents]

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
            max_check = min(agent.current_checkpoint + look_ahead, num_checkpoints)
            passed_checkpoint = False
            for cp_idx in range(agent.current_checkpoint, max_check):
                if env.is_checkpoint_passed(agent.prev_x, agent.prev_y, agent.x, agent.y, cp_idx):
                    # Atlanan checkpoint'ler için de puan ver
                    checkpoints_passed = cp_idx - agent.current_checkpoint + 1
                    ge[i].fitness += 100 * checkpoints_passed

                    # Hizli gecis bonusu: checkpoint'i ne kadar az frame'de gecersen o kadar bonus
                    frames_used = frames_since_last_checkpoint[i]
                    if frames_used < 150:
                        ge[i].fitness += (150 - frames_used) * 0.2  # max +30 bonus

                    agent.current_checkpoint = cp_idx + 1
                    frames_since_last_checkpoint[i] = 0
                    passed_checkpoint = True
                    # Yeni checkpoint icin mesafe referansini guncelle
                    prev_dist_to_cp[i] = _dist_to_checkpoint(agent.x, agent.y, agent.current_checkpoint)
                    break

            # 4.1 İlk checkpoint yerine son checkpoint'ten geçerse eksi puan ver (ters yön teşhisi)
            if (not passed_checkpoint and not wrong_way_penalized[i]
                and agent.current_checkpoint == 0 and num_checkpoints > 1):
                last_cp_idx = num_checkpoints - 1
                if env.is_checkpoint_passed(agent.prev_x, agent.prev_y, agent.x, agent.y, last_cp_idx):
                    ge[i].fitness -= 100
                    wrong_way_penalized[i] = True

            # 5. Sonraki checkpoint'e yakinlik odulu (hiz odulunun yerine)
            curr_dist = _dist_to_checkpoint(agent.x, agent.y, agent.current_checkpoint)
            if curr_dist < prev_dist_to_cp[i]:
                ge[i].fitness += (prev_dist_to_cp[i] - curr_dist) * 0.3
            prev_dist_to_cp[i] = curr_dist

            # 6. Zaman aşımı kontrolü: Eğer ajan uzun süre checkpoint geçemezse öldür
            if frames_since_last_checkpoint[i] > 300:  # ~5 saniye (60 FPS)
                agent.is_alive = False
                ge[i].fitness -= 50

            # 7. Çarpışma kontrolü
            if env.check_collision(agent.x, agent.y):
                agent.is_alive = False
                ge[i].fitness -= 25

            # 8. Tur tamamlama kontrolu: Tum checkpoint'ler gecildiyse ve bitis cizgisini gectiyse
            if agent.current_checkpoint >= num_checkpoints:
                if env.is_finish_line_crossed(agent.prev_x, agent.prev_y, agent.x, agent.y):
                    ge[i].fitness += 500  # Tur tamamlama bonusu
                    agent.is_alive = False  # Turu bitirdi, vakit kaybettirmesin

            # 9. Eğer ajan ters yönde hareket ediyorsa
            

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

def setup_pygame_and_environment(track_path):
    global env, screen, virtual_surface, clock, window_width, window_height, track_image, hud_font, checkpoint_font

    pygame.init()

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
    # 1. Pist Secim Ekranini Goster
    selected_track = show_track_selector()

    # 2. Pygame ve Cevreyi Hazirla
    setup_pygame_and_environment(selected_track)

    # 3. config dosyasinin yolunu bul (Ana dizinde oldugunu varsayiyoruz)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, '..', 'config-feedforward.txt')

    # 4. NEAT algoritmasini calistir
    run_neat(config_path)
