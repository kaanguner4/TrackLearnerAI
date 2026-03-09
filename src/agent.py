import pygame
import math

class Agent:
    def __init__(self, start_position, track_env):
        self.track_env = track_env
        self.x, self.y = start_position
        self.angle = 0 # Derece cinsinden, 0 doğuya bakar
        self.speed = 0
        self.max_speed = 5
        self.acceleration = 0.2
        self.turn_speed = 5 # Derece cinsinden dönüş hızı

        self.is_alive = True
        self.radars = [] # Radar verilerini tutacak liste
        self.current_checkpoint = 0 # Hangi checkpoint'in hedef olduğunu takip eder
        self.distance_driven = 0 # fitness hesaplaması için

        # Göreselleştirme (daha sonra assets'den resim yüklenebilir)
        self.width = 20
        self.height = 10
        self.color = (0, 255, 0) # Yeşil renk

    def update(self):
        if not self.is_alive:
            return
        
        # 1. Pozisyonu güncelle (trigonometri kullanarak)
        radians = math.radians(self.angle)
        self.x += math.cos(radians) * self.speed
        self.y += math.sin(radians) * self.speed
        self.distance_driven += self.speed

        # 2. Çarpışma kontrolü (track_env üzerinden)
        if self.track_env.check_collision(self.x, self.y):
            self.is_alive = False
            return
        
        # 3. Checkpoint kontrolü
        if self.track_env.is_checkpoint_passed(self.x, self.y, self.current_checkpoint):
            self.current_checkpoint += 1
            # Eğer tüm checkpointler biterse tur tamamlandı demektir (Bunu main loop'ta da yönetebiliriz)

        # 4. Radar verilerini güncelle
        self._update_radars()

    def _update_radars(self):
        self.radars.clear()
        # 5 radar açısı (-90, -45, 0, 45, 90) ve her biri için mesafe ölçümü
        radar_angles = [-90, -45, 0, 45, 90]
        for radar_angle in radar_angles:
            self._calculate_single_radar(radar_angle)
        
    def _calculate_single_radar(self, radar_angle):
        length = 0
        radar_x = int(self.x)
        radar_y = int(self.y)

        # Radar açısını = agent açısı + radarın kendi açısı
        angle = (self.angle + radar_angle) % 360

        # Radar siyah duvara çarpana kadar veya max uzunluğa (örn: 200px) ulaşana kadar çizgiyi uzat
        while not self.track_env.check_collision(radar_x, radar_y) and length < 200:
            length += 1
            radar_x = int(self.x + math.cos(math.radians(angle)) * length)
            radar_y = int(self.y + math.sin(math.radians(angle)) * length)

        self.radars.append((radar_x, radar_y, length))

    def get_data(self):
        # Yapay sinir ağına (NEAT) verilecek girdiler (Inputs)
        # Radarların mesafelerini ve aracın anlık hızını döndürürüz
        radar_distances = [radar[2] for radar in self.radars]
        return radar_distances + [self.speed]
    
    def draw(self, screen):
        # Aracı çiz (daha sonra resimle değiştirilebilir)
        if self.is_alive:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), 5)
            # Radar çizgilerini de görselleştirelim
            for radar in self.radars:
                radar_pos = (radar[0], radar[1])
                pygame.draw.line(screen, (0, 255, 255), (int(self.x), int(self.y)), radar_pos, 1)
    


if __name__ == "__main__":
    # Test için track_env dosyasındaki sınıfı buraya import etmemiz gerekir
    import track_env 
    
    print("Sınıflar test ediliyor...")
    
    # 1. Önce çevreyi (Environment) oluşturuyoruz
    track_env.TrackEnvironment("assets/tracks/track1.png") 
    
    # 2. Ajanı (Agent) oluştururken bu çevre nesnesini veriyoruz
    agent = Agent((100, 100), track_env.TrackEnvironment("assets/tracks/track1.png")) # Başlangıç pozisyonu (100, 100) olarak belirlendi
    
    # 3. Ajanın sensörlerini bir kereliğine manuel tetikliyoruz (veya update çağırıyoruz)
    agent._update_radars()
    
    # 4. Veriyi kontrol ediyoruz
    print("Agent sensör verileri ve hız:", agent.get_data())