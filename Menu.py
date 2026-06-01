import pygame


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHTGRAY = (211, 211, 211)
GRAY = (160, 160, 160)
DARKGRAY = (90, 90, 90)
BLUE = (60, 110, 200)
LIGHTBLUE = (130, 170, 230)
GREEN = (60, 160, 90)


class Button:
    def __init__(self, rect, label, color=BLUE, hover=LIGHTBLUE):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.color = color
        self.hover = hover

    def draw(self, surf, font, mouse):
        c = self.hover if self.rect.collidepoint(mouse) else self.color
        pygame.draw.rect(surf, c, self.rect, border_radius=8)
        pygame.draw.rect(surf, BLACK, self.rect, width=2, border_radius=8)
        txt = font.render(self.label, True, WHITE)
        rect = txt.get_rect(center=self.rect.center)
        surf.blit(txt, rect)

    def hit(self, pos):
        return self.rect.collidepoint(pos)


class Slider:
    def __init__(self, rect, label, lo, hi, value, step=1, fmt='{}'):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.lo = lo
        self.hi = hi
        self.value = value
        self.step = step
        self.fmt = fmt
        self.dragging = False
        self.minus = pygame.Rect(self.rect.x - 50, self.rect.y, 30, self.rect.height)
        self.plus = pygame.Rect(self.rect.right + 20, self.rect.y, 30, self.rect.height)

    def draw(self, surf, font, mouse):
        pygame.draw.rect(surf, GRAY, self.minus, border_radius=4)
        pygame.draw.rect(surf, BLACK, self.minus, width=1, border_radius=4)
        m = font.render('-', True, WHITE)
        surf.blit(m, m.get_rect(center=self.minus.center))

        pygame.draw.rect(surf, GRAY, self.plus, border_radius=4)
        pygame.draw.rect(surf, BLACK, self.plus, width=1, border_radius=4)
        p = font.render('+', True, WHITE)
        surf.blit(p, p.get_rect(center=self.plus.center))

        pygame.draw.rect(surf, WHITE, self.rect, border_radius=6)
        pygame.draw.rect(surf, BLACK, self.rect, width=2, border_radius=6)
        t = (self.value - self.lo) / (self.hi - self.lo) if self.hi > self.lo else 0
        knob_x = self.rect.x + int(t * (self.rect.width - 16))
        pygame.draw.rect(surf, BLUE, (knob_x, self.rect.y, 16, self.rect.height), border_radius=4)

        lbl = font.render(f'{self.label}: {self.fmt.format(self.value)}', True, BLACK)
        surf.blit(lbl, (self.rect.x - 50, self.rect.y - 28))

    def event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.minus.collidepoint(event.pos):
                self._step(-1)
            elif self.plus.collidepoint(event.pos):
                self._step(1)
            elif self.rect.collidepoint(event.pos):
                self.dragging = True
                self._set_from_x(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(event.pos[0])

    def _set_from_x(self, x):
        t = max(0.0, min(1.0, (x - self.rect.x) / max(1, self.rect.width)))
        raw = self.lo + t * (self.hi - self.lo)
        self.value = self._snap(raw)

    def _step(self, direction):
        self.value = self._snap(self.value + direction * self.step)

    def _snap(self, v):
        v = max(self.lo, min(self.hi, v))
        steps = round((v - self.lo) / self.step)
        v = self.lo + steps * self.step
        if isinstance(self.step, int):
            return int(v)
        return round(v, 3)


class MenuScreen:
    """Standalone menu window. Used before grid Graphics exists."""
    WIDTH = 480
    HEIGHT = 560

    def __init__(self):
        self.screen = None
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont('Arial', 36, bold=True)
        self.font_btn = pygame.font.SysFont('Arial', 22, bold=True)
        self.font_lbl = pygame.font.SysFont('Arial', 16, bold=True)

    def show(self):
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('GridWorld RL')

    def main_menu(self):
        """Returns: 'start' | 'settings' | 'quit'."""
        self.show()
        btns = {
            'start': Button((140, 220, 200, 50), 'Start Training', GREEN, (110, 200, 130)),
            'settings': Button((140, 290, 200, 50), 'Settings'),
            'quit': Button((140, 360, 200, 50), 'Quit', DARKGRAY, GRAY),
        }
        return self._run_menu('GridWorld RL', btns)

    def post_run_menu(self):
        """Returns: 'again' | 'settings' | 'quit'."""
        self.show()
        btns = {
            'again': Button((140, 220, 200, 50), 'Train Again', GREEN, (110, 200, 130)),
            'settings': Button((140, 290, 200, 50), 'New Settings'),
            'quit': Button((140, 360, 200, 50), 'Quit', DARKGRAY, GRAY),
        }
        return self._run_menu('Done!', btns)

    def settings_menu(self, rows, cols, gamma, step_cost, slip):
        """Returns (rows, cols, gamma, step_cost, slip) or None if quit."""
        self.show()
        sliders = [
            Slider((150, 130, 200, 22), 'Rows', 2, 50, rows, step=1, fmt='{}'),
            Slider((150, 200, 200, 22), 'Cols', 2, 50, cols, step=1, fmt='{}'),
            Slider((150, 270, 200, 22), 'Gamma', 0.0, 1.0, gamma, step=0.05, fmt='{:.2f}'),
            Slider((150, 340, 200, 22), 'StepCost', 0.0, 0.2, step_cost, step=0.01, fmt='{:.2f}'),
            Slider((150, 410, 200, 22), 'Slip', 0.0, 0.5, slip, step=0.05, fmt='{:.2f}'),
        ]
        back = Button((140, 470, 200, 50), 'Back', BLUE, LIGHTBLUE)
        while True:
            mouse = pygame.mouse.get_pos()
            self.screen.fill(LIGHTGRAY)
            t = self.font_title.render('Settings', True, BLACK)
            self.screen.blit(t, t.get_rect(center=(self.WIDTH // 2, 60)))
            for s in sliders:
                s.draw(self.screen, self.font_lbl, mouse)
            back.draw(self.screen, self.font_btn, mouse)
            pygame.display.update()
            self.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                for s in sliders:
                    s.event(event)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if back.hit(event.pos):
                        return tuple(s.value for s in sliders)

    def _run_menu(self, title, btns):
        while True:
            mouse = pygame.mouse.get_pos()
            self.screen.fill(LIGHTGRAY)
            t = self.font_title.render(title, True, BLACK)
            self.screen.blit(t, t.get_rect(center=(self.WIDTH // 2, 70)))
            for b in btns.values():
                b.draw(self.screen, self.font_btn, mouse)
            pygame.display.update()
            self.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for key, b in btns.items():
                        if b.hit(event.pos):
                            return key
