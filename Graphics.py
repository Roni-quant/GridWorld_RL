import pygame


FPS = 60
LINE_WIDTH = 2
PADDING = 2
HEADER = 92
FRAME = 2
MAX_WINDOW = 800
MIN_SQUARE = 12
TEXT_THRESHOLD = 32
ARROW_THRESHOLD = 24

ARROWS = {0: '^', 1: '>', 2: '<', 3: 'v'}  # UP, RIGHT, LEFT, DOWN

RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
LIGHTGRAY = (211, 211, 211)
GREEN = (0, 128, 0)
LIGHTGREEN = (180, 230, 180)
LIGHTRED = (240, 180, 180)
YELLOWHILITE = (255, 245, 150)


class Graphics:
    def __init__(self, env) -> None:
        self.env = env
        self.rows = env.rows
        self.cols = env.cols
        self.square = max(MIN_SQUARE, min(MAX_WINDOW // max(self.rows, self.cols), 140))
        self.width = self.cols * self.square
        self.height = self.rows * self.square + HEADER
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('GridWorld RL')

        val_size = max(14, self.square // 6)
        arr_size = max(20, self.square // 3)
        term_size = max(20, self.square // 3)
        self.font_header = pygame.font.SysFont('Arial', 18, bold=True)
        self.font_header_sub = pygame.font.SysFont('Arial', 14)
        self.font_val = pygame.font.SysFont('Arial', val_size, bold=True)
        self.font_arrow = pygame.font.SysFont('Arial', arr_size, bold=True)
        self.font_term = pygame.font.SysFont('Arial', term_size, bold=True)
        self.load_img()

    def load_img(self):
        robot = pygame.image.load('Img/Robot_1.png')
        size = max(20, self.square - 40)
        self.robot = pygame.transform.scale(robot, (size, size))

    def draw(self, state, header_main='Trained agent running', header_sub='', paused=False):
        self.screen.fill(LIGHTGRAY)
        self.draw_header(header_main, header_sub)
        self.draw_lines()
        self.draw_end_squares()
        self.draw_start_marker()
        self.draw_img(state, self.robot)
        self.draw_run_toolbar(paused)
        pygame.display.update()

    def draw_run_toolbar(self, paused=False):
        btn_h = 28
        y = HEADER - btn_h - 6
        labels = [
            ('pause', 'Resume' if paused else 'Pause'),
            ('reset', 'Restart'),
            ('back', 'Back'),
        ]
        btn_w = min(90, (self.width - 20) // len(labels))
        colors = {'pause': (200, 130, 60), 'reset': (160, 160, 160), 'back': (90, 90, 90)}
        rects = {}
        for i, (key, lbl) in enumerate(labels):
            x = 8 + i * (btn_w + 4)
            rect = pygame.Rect(x, y, btn_w, btn_h)
            pygame.draw.rect(self.screen, colors[key], rect, border_radius=4)
            pygame.draw.rect(self.screen, BLACK, rect, width=1, border_radius=4)
            txt = self.font_header_sub.render(lbl, True, WHITE)
            self.screen.blit(txt, txt.get_rect(center=rect.center))
            rects[key] = rect
        self._toolbar_rects = rects

    def draw_training(self, V, Policy, iteration, delta, sub=''):
        self.screen.fill(LIGHTGRAY)
        self.draw_header(f'VI iter={iteration}  delta={delta:.4f}', sub)
        self.shade_values(V)
        self.draw_end_squares()
        self.draw_start_marker()
        self.draw_values(V, Policy)
        self.draw_lines()
        self.draw_train_toolbar()
        pygame.display.update()

    def draw_train_toolbar(self):
        btn_h = 28
        y = HEADER - btn_h - 6
        labels = [('step', 'Step'), ('auto', 'Auto'), ('skip', 'Skip'), ('back', 'Back')]
        btn_w = min(90, (self.width - 20) // len(labels))
        colors = {'step': (60, 110, 200), 'auto': (60, 160, 90),
                  'skip': (200, 130, 60), 'back': (90, 90, 90)}
        rects = {}
        for i, (key, lbl) in enumerate(labels):
            x = 8 + i * (btn_w + 4)
            rect = pygame.Rect(x, y, btn_w, btn_h)
            pygame.draw.rect(self.screen, colors[key], rect, border_radius=4)
            pygame.draw.rect(self.screen, BLACK, rect, width=1, border_radius=4)
            txt = self.font_header_sub.render(lbl, True, WHITE)
            self.screen.blit(txt, txt.get_rect(center=rect.center))
            rects[key] = rect
        self._toolbar_rects = rects
        return rects

    def draw_setup(self, msg):
        self.screen.fill(LIGHTGRAY)
        self.draw_header(msg)
        self.draw_end_squares()
        self.draw_start_marker()
        self.draw_lines()
        self.draw_setup_toolbar()
        pygame.display.update()

    def draw_setup_toolbar(self):
        """Buttons row in header area. Returns dict of name -> Rect for hit-testing."""
        btn_h = 28
        y = HEADER - btn_h - 6
        labels = [('random', 'Random'), ('clear', 'Clear'), ('train', 'Train'), ('back', 'Back')]
        btn_w = min(90, (self.width - 20) // len(labels))
        colors = {'random': (60, 160, 90), 'clear': (160, 160, 160),
                  'train': (60, 110, 200), 'back': (90, 90, 90)}
        rects = {}
        for i, (key, lbl) in enumerate(labels):
            x = 8 + i * (btn_w + 4)
            rect = pygame.Rect(x, y, btn_w, btn_h)
            pygame.draw.rect(self.screen, colors[key], rect, border_radius=4)
            pygame.draw.rect(self.screen, BLACK, rect, width=1, border_radius=4)
            txt = self.font_header_sub.render(lbl, True, WHITE)
            self.screen.blit(txt, txt.get_rect(center=rect.center))
            rects[key] = rect
        self._toolbar_rects = rects
        self._toolbar_h = btn_h + 8
        return rects

    def toolbar_hit(self, pos):
        for key, rect in getattr(self, '_toolbar_rects', {}).items():
            if rect.collidepoint(pos):
                return key
        return None

    def draw_start_marker(self):
        x, y = self.calc_pos(self.env.start)
        pygame.draw.rect(self.screen, BLUE, (x, y, self.square - PADDING, self.square - PADDING), width=3)

    def draw_header(self, txt, sub=''):
        pygame.draw.rect(self.screen, WHITE, (0, 0, self.width, HEADER))
        surf = self.font_header.render(txt, True, BLACK)
        self.screen.blit(surf, (10, 6))
        if sub:
            surf2 = self.font_header_sub.render(sub, True, BLACK)
            self.screen.blit(surf2, (10, 32))

    def draw_lines(self):
        if self.square < 8:
            return
        for i in range(self.rows + 1):
            y = HEADER + i * self.square
            pygame.draw.line(self.screen, BLACK, (0, y), (self.width, y), width=LINE_WIDTH)
        for i in range(self.cols + 1):
            x = i * self.square
            pygame.draw.line(self.screen, BLACK, (x, HEADER), (x, self.height), width=LINE_WIDTH)

    def shade_values(self, V):
        vmax = max(abs(V.max()), abs(V.min()), 1e-6)
        for row in range(self.rows):
            for col in range(self.cols):
                if self.env.board[row, col] != 0:
                    continue
                v = V[row, col]
                t = min(abs(v) / vmax, 1.0)
                if v > 0:
                    color = pygame.Color(*LIGHTGRAY).lerp(pygame.Color(*LIGHTGREEN), t)
                elif v < 0:
                    color = pygame.Color(*LIGHTGRAY).lerp(pygame.Color(*LIGHTRED), t)
                else:
                    color = LIGHTGRAY
                self.draw_square((row, col), color)

    def draw_end_squares(self):
        for row in range(self.rows):
            for col in range(self.cols):
                v = self.env.board[row, col]
                if v == -1:
                    self.draw_square((row, col), RED)
                    self.draw_center_txt((row, col), "-1", self.font_term, WHITE)
                elif v == 1:
                    self.draw_square((row, col), GREEN)
                    self.draw_center_txt((row, col), "+1", self.font_term, WHITE)

    def draw_values(self, V, Policy):
        if self.square < ARROW_THRESHOLD:
            return  # too small: color shading only
        show_text = self.square >= TEXT_THRESHOLD
        for row in range(self.rows):
            for col in range(self.cols):
                if self.env.board[row, col] != 0:
                    continue
                x, y = self.calc_pos((row, col))
                if show_text:
                    v_txt = self.font_val.render(f'{V[row,col]:+.2f}', True, BLACK)
                    self.screen.blit(v_txt, (x + 4, y + 4))
                arrow = ARROWS.get(int(Policy[row, col]), '?')
                a_txt = self.font_arrow.render(arrow, True, BLUE)
                rect = a_txt.get_rect(center=(x + self.square // 2, y + self.square // 2 + self.square // 10))
                self.screen.blit(a_txt, rect)

    def draw_square(self, row_col, color):
        pos = self.calc_pos(row_col)
        pygame.draw.rect(self.screen, color, (*pos, self.square - PADDING, self.square - PADDING))

    def draw_center_txt(self, row_col, txt, font, color):
        x, y = self.calc_pos(row_col)
        surf = font.render(txt, True, color)
        rect = surf.get_rect(center=(x + self.square // 2, y + self.square // 2))
        self.screen.blit(surf, rect)

    def draw_img(self, row_col, img):
        x, y = self.calc_pos(row_col)
        off = (self.square - img.get_width()) // 2
        self.screen.blit(img, (x + off, y + off))

    def calc_pos(self, row_col):
        row, col = row_col
        y = HEADER + row * self.square + FRAME
        x = col * self.square + FRAME
        return x, y

    def pixel_to_cell(self, pos):
        px, py = pos
        if py < HEADER:
            return None
        col = px // self.square
        row = (py - HEADER) // self.square
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return (int(row), int(col))
        return None

    def __call__(self, state=(0, 0)):
        self.draw(state)
