import pygame
import math

class Agent:
    def __init__(self, start_position, track_env):
        self.track_env = track_env
        self.x, self.y = start_position
        self.prev_x, self.prev_y = start_position # Önceki konumu tutacak
        self.angle = 0 
        self.speed = 0
        self.max_speed = 5
        self.acceleration = 0.2
        self.turn_speed = 5 

        self.is_alive = True
        self.radars = [] 
        self.current_checkpoint = 0 
        self.distance_driven = 0 

        self.width = 20
        self.height = 10
        self.color = (0, 0, 255) # MAVİ YAPTIK
        
        # AJAN DOĞDUĞU AN İLK SENSÖR VERİSİNİ ALSIN DİYE EKLENEN SATIR:
        self._update_radars()

    def update(self):
        if not self.is_alive:
            return
        
        # Harekete başlamadan önce şimdiki konumu "eski konum" olarak kaydet
        self.prev_x = self.x
        self.prev_y = self.y
        
        # 1. Pozisyonu güncelle 
        radians = math.radians(self.angle)
        self.x += math.cos(radians) * self.speed
        self.y += math.sin(radians) * self.speed
        self.distance_driven += abs(self.speed)

        # 2. Çarpışma kontrolü
        if self.track_env.check_collision(self.x, self.y):
            self.is_alive = False
            return

        # 3. Radar verilerini güncelle
        self._update_radars()

    def _update_radars(self):
        self.radars.clear()
        radar_angles = [-90, -45, 0, 45, 90]
        for radar_angle in radar_angles:
            self._calculate_single_radar(radar_angle)
        
    def _calculate_single_radar(self, radar_angle):
        length = 0
        radar_x = int(self.x)
        radar_y = int(self.y)
        angle = (self.angle + radar_angle) % 360

        while not self.track_env.check_collision(radar_x, radar_y) and length < 200:
            length += 1
            radar_x = int(self.x + math.cos(math.radians(angle)) * length)
            radar_y = int(self.y + math.sin(math.radians(angle)) * length)

        self.radars.append((radar_x, radar_y, length))

    def get_data(self):
        # Radar mesafelerini 0-1 arasına normalize et (max 200 piksel)
        radar_distances = [radar[2] / 200.0 for radar in self.radars]
        # Hızı -1 ile 1 arasına normalize et
        normalized_speed = self.speed / self.max_speed
        return radar_distances + [normalized_speed]
    
    def draw(self, screen):
        if self.is_alive:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), 8)
            for radar in self.radars:
                radar_pos = (radar[0], radar[1])
                pygame.draw.line(screen, (255, 0, 255), (int(self.x), int(self.y)), radar_pos, 1) # Mor radar çizgisi