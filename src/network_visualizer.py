import pygame
import math


class NetworkVisualizer:
    """NEAT sinir aginin anlik aktivasyon degerlerini gorsellestirir."""

    # Input ve output etiketleri
    INPUT_LABELS = ["Radar L", "Radar FL", "Radar F", "Radar FR", "Radar R", "Speed"]
    OUTPUT_LABELS = ["Gas", "Brake", "Left", "Right"]

    def __init__(self, width=340, height=260, padding=30):
        self.width = width
        self.height = height
        self.padding = padding
        self.font = None
        self.surface = None

    def _ensure_init(self):
        if self.font is None:
            self.font = pygame.font.SysFont(None, 18)
            self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

    # --- Renk yardimcilari ---
    @staticmethod
    def _activation_color(value):
        """Aktivasyon degeri: -1 mavi, 0 beyaz, +1 kirmizi."""
        t = max(-1.0, min(1.0, value))
        if t >= 0:
            r = 255
            g = int(255 * (1 - t))
            b = int(255 * (1 - t))
        else:
            r = int(255 * (1 + t))
            g = int(255 * (1 + t))
            b = 255
        return (r, g, b)

    @staticmethod
    def _weight_color(weight):
        """Pozitif agirlik yesil, negatif kirmizi."""
        if weight >= 0:
            return (0, 200, 0)
        return (200, 0, 0)

    # --- Layout hesaplama ---
    def _compute_layout(self, net):
        """Node pozisyonlarini katmanli olarak hesapla (sol: input, sag: output, orta: hidden)."""
        input_keys = list(net.input_nodes)
        output_keys = list(net.output_nodes)

        hidden_keys = []
        for node_key, _, _, _, _, _ in net.node_evals:
            if node_key not in output_keys:
                hidden_keys.append(node_key)

        usable_w = self.width - 2 * self.padding
        usable_h = self.height - 2 * self.padding

        has_hidden = len(hidden_keys) > 0
        num_layers = 3 if has_hidden else 2

        positions = {}
        layer_x = []
        if num_layers == 2:
            layer_x = [self.padding, self.padding + usable_w]
        else:
            layer_x = [
                self.padding,
                self.padding + usable_w // 2,
                self.padding + usable_w,
            ]

        # input dugumler (sol sutun)
        for i, key in enumerate(input_keys):
            n = len(input_keys)
            y = self.padding + (i + 0.5) * usable_h / n
            positions[key] = (layer_x[0], int(y))

        # hidden dugumler (orta sutun)
        if has_hidden:
            for i, key in enumerate(hidden_keys):
                n = len(hidden_keys)
                y = self.padding + (i + 0.5) * usable_h / n
                positions[key] = (layer_x[1], int(y))

        # output dugumler (sag sutun)
        out_x = layer_x[-1]
        for i, key in enumerate(output_keys):
            n = len(output_keys)
            y = self.padding + (i + 0.5) * usable_h / n
            positions[key] = (out_x, int(y))

        return positions, input_keys, output_keys, hidden_keys

    # --- Ana cizim ---
    def draw(self, target_surface, net, genome, x, y):
        """
        Neural network'u target_surface uzerine (x, y) konumunda cizer.

        net   : neat.nn.FeedForwardNetwork (activate() cagrilmis olmali)
        genome: neat.DefaultGenome
        """
        self._ensure_init()
        self.surface.fill((0, 0, 0, 0))

        # Yari saydam arka plan
        bg = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 180))
        self.surface.blit(bg, (0, 0))

        # Baslik
        title = self.font.render("Neural Network", True, (255, 255, 255))
        self.surface.blit(title, (self.width // 2 - title.get_width() // 2, 4))

        positions, input_keys, output_keys, hidden_keys = self._compute_layout(net)

        activations = getattr(net, 'values', {})

        # --- Baglantilari ciz ---
        for cg in genome.connections.values():
            if not cg.enabled:
                continue
            in_key, out_key = cg.key
            if in_key not in positions or out_key not in positions:
                continue
            color = self._weight_color(cg.weight)
            thickness = max(1, min(4, int(abs(cg.weight))))
            alpha = max(60, min(255, int(abs(cg.weight) * 80)))
            line_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.line(
                line_surf, (*color, alpha),
                positions[in_key], positions[out_key], thickness
            )
            self.surface.blit(line_surf, (0, 0))

        # --- Dugumleri ciz ---
        node_radius = 8

        # Input dugumler
        for i, key in enumerate(input_keys):
            val = activations.get(key, 0.0)
            color = self._activation_color(val)
            pos = positions[key]
            pygame.draw.circle(self.surface, color, pos, node_radius)
            pygame.draw.circle(self.surface, (255, 255, 255), pos, node_radius, 1)
            label = self.INPUT_LABELS[i] if i < len(self.INPUT_LABELS) else str(key)
            lbl_surf = self.font.render(label, True, (200, 200, 200))
            self.surface.blit(lbl_surf, (pos[0] - lbl_surf.get_width() - 6, pos[1] - 7))

        # Hidden dugumler
        for key in hidden_keys:
            val = activations.get(key, 0.0)
            color = self._activation_color(val)
            pos = positions[key]
            pygame.draw.circle(self.surface, color, pos, node_radius)
            pygame.draw.circle(self.surface, (255, 255, 255), pos, node_radius, 1)

        # Output dugumler
        for i, key in enumerate(output_keys):
            val = activations.get(key, 0.0)
            color = self._activation_color(val)
            pos = positions[key]
            pygame.draw.circle(self.surface, color, pos, node_radius)
            pygame.draw.circle(self.surface, (255, 255, 255), pos, node_radius, 1)
            label = self.OUTPUT_LABELS[i] if i < len(self.OUTPUT_LABELS) else str(key)
            lbl_surf = self.font.render(label, True, (200, 200, 200))
            self.surface.blit(lbl_surf, (pos[0] + node_radius + 4, pos[1] - 7))

        target_surface.blit(self.surface, (x, y))
