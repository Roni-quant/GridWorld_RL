import random
import pygame
from Graphics import Graphics, ARROWS, FPS
from Environement import Environement
from Agent import AI_Agent
from Menu import MenuScreen


TRAP_DENSITY = 0.15  # fraction of empty cells turned to -1 on random fill


RUN_STEP_MS = 250


def print_grid(label, arr, fmt='{:+.3f}'):
    print(f'{label}:')
    for row in arr:
        print('  ' + '  '.join(fmt.format(v) for v in row))


def print_policy(env, Policy):
    print('Policy* (arrows):')
    for r in range(env.rows):
        cells = []
        for c in range(env.cols):
            if env.board[r, c] == 1:
                cells.append(' G ')
            elif env.board[r, c] == -1:
                cells.append(' X ')
            else:
                cells.append(' ' + ARROWS[int(Policy[r, c])] + ' ')
        print(''.join(cells))


def random_fill(env):
    env.board[:] = 0
    cells = [(r, c) for r in range(env.rows) for c in range(env.cols) if (r, c) != env.start]
    random.shuffle(cells)
    if not cells:
        return
    goal = cells.pop()
    env.set_reward(goal, 1)
    n_traps = max(1, int(len(cells) * TRAP_DENSITY))
    for cell in cells[:n_traps]:
        env.set_reward(cell, -1)


def setup_rewards(env, graphics):
    msg = ('L-click: +1/-1/clear  R-click: start  '
           'R=random  C=clear  ENTER=train  ESC=back')
    graphics.draw_setup(msg)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if (env.board == 1).any():
                        return 'go'
                    graphics.draw_setup('Need at least one +1 cell. Press R for random.')
                    continue
                if event.key == pygame.K_ESCAPE:
                    return 'back'
                if event.key == pygame.K_r:
                    random_fill(env)
                    graphics.draw_setup(msg)
                if event.key == pygame.K_c:
                    env.board[:] = 0
                    graphics.draw_setup(msg)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    hit = graphics.toolbar_hit(event.pos)
                    if hit == 'random':
                        random_fill(env)
                        graphics.draw_setup(msg)
                        continue
                    if hit == 'clear':
                        env.board[:] = 0
                        graphics.draw_setup(msg)
                        continue
                    if hit == 'train':
                        if (env.board == 1).any():
                            return 'go'
                        graphics.draw_setup('Need at least one +1 cell. Click Random.')
                        continue
                    if hit == 'back':
                        return 'back'
                cell = graphics.pixel_to_cell(event.pos)
                if cell is None:
                    continue
                if event.button == 1:
                    if cell == env.start:
                        continue
                    cur = env.board[cell]
                    nxt = 1 if cur == 0 else (-1 if cur == 1 else 0)
                    env.set_reward(cell, nxt)
                    graphics.draw_setup(msg)
                elif event.button == 3:
                    if env.board[cell] != 0:
                        continue
                    env.start = cell
                    env.state = cell
                    graphics.draw_setup(msg)
        pygame.time.wait(20)


def train(agent, graphics):
    n_states = agent.rows * agent.cols
    render_every = 1 if n_states <= 400 else (5 if n_states <= 2500 else 20)
    auto_ms = 400 if n_states <= 400 else (80 if n_states <= 2500 else 20)
    state = {'mode': 'manual', 'skip': False, 'quit': False, 'last_auto': pygame.time.get_ticks()}
    hint = '[SPACE]=step  [ENTER]=auto  [ESC]=skip'
    graphics.draw_training(agent.Value, agent.Policy, 0, 0.0, f'gamma={agent.gamma} cost={agent.env.step_cost} slip={agent.env.slip}  {hint}')

    def poll():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state['quit'] = True
                state['skip'] = True
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return True
                if event.key == pygame.K_RETURN:
                    state['mode'] = 'auto' if state['mode'] == 'manual' else 'manual'
                if event.key == pygame.K_ESCAPE:
                    state['skip'] = True
                    return True
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                hit = graphics.toolbar_hit(event.pos)
                if hit == 'step':
                    return True
                if hit == 'auto':
                    state['mode'] = 'auto' if state['mode'] == 'manual' else 'manual'
                if hit == 'skip':
                    state['skip'] = True
                    return True
                if hit == 'back':
                    state['quit'] = True
                    state['skip'] = True
                    return True
        return False

    def wait_for_input():
        if state['skip']:
            return
        if state['mode'] == 'auto':
            while pygame.time.get_ticks() - state['last_auto'] < auto_ms:
                if poll():
                    return
                pygame.time.wait(10)
            state['last_auto'] = pygame.time.get_ticks()
            return
        while True:
            advance = poll()
            if state['skip']:
                return
            if state['mode'] == 'auto':
                state['last_auto'] = pygame.time.get_ticks()
                return
            if advance:
                return
            pygame.time.wait(20)

    def on_sweep(iteration, delta, V, Policy):
        mode_txt = '[SKIP]' if state['skip'] else f'[{state["mode"].upper()}]'
        sub = f'gamma={agent.gamma} cost={agent.env.step_cost} slip={agent.env.slip}  {mode_txt}  {hint}'
        if iteration % render_every == 0 or state['mode'] == 'manual':
            graphics.draw_training(V, Policy, iteration, delta, sub)
        if iteration % render_every == 0:
            print(f'[iter {iteration:3d}]  delta = {delta:.6f}  mode={state["mode"]}')
        if not state['skip']:
            wait_for_input()
        return False

    agent.Value_Iteration(callback=on_sweep)
    return 'quit' if state['quit'] else 'done'


def run_agent(env, agent, graphics):
    clock = pygame.time.Clock()
    v_start = agent.Value[env.start]
    stats = {'episode': 1, 'steps_in_ep': 0, 'ep_reward': 0.0,
             'wins': 0, 'losses': 0, 'total_ep_reward': 0.0, 'paused': False}

    def reset_stats():
        env.reset()
        agent.Reward = 0
        stats.update(episode=1, steps_in_ep=0, ep_reward=0.0,
                     wins=0, losses=0, total_ep_reward=0.0)

    def render():
        eps_done = stats['wins'] + stats['losses']
        avg = (stats['total_ep_reward'] / eps_done) if eps_done else 0.0
        succ = (100.0 * stats['wins'] / eps_done) if eps_done else 0.0
        main = f'Run  ep={stats["episode"]} step={stats["steps_in_ep"]} ep_r={stats["ep_reward"]:+.2f}'
        sub = (f'gamma={agent.gamma} cost={env.step_cost} slip={env.slip}  |  '
               f'V*(start)={v_start:+.2f}  avg={avg:+.2f}  succ={succ:.0f}%')
        graphics.draw(env.state, main, sub, stats['paused'])

    print('\n--- Running trained agent ---')
    print(f'V*(start) = {v_start:+.4f}  (Bellman prediction of episode return)')
    render()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return 'done'
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                hit = graphics.toolbar_hit(event.pos)
                if hit == 'back':
                    return 'done'
                if hit == 'pause':
                    stats['paused'] = not stats['paused']
                    render()
                if hit == 'reset':
                    reset_stats()
                    render()

        if stats['paused']:
            pygame.time.wait(50)
            clock.tick(FPS)
            continue

        action = agent(env.state)
        env.state, reward = env.move(env.state, action)
        agent.add_reward(reward)
        stats['steps_in_ep'] += 1
        stats['ep_reward'] += reward

        if env.end_of_game(env.state):
            if env.board[env.state] > 0:
                stats['wins'] += 1
                result = 'WIN'
            else:
                stats['losses'] += 1
                result = 'LOSS'
            stats['total_ep_reward'] += stats['ep_reward']
            print(f'[ep {stats["episode"]:3d}] {result}  steps={stats["steps_in_ep"]:3d}  '
                  f'ep_r={stats["ep_reward"]:+.2f}  '
                  f'avg={stats["total_ep_reward"]/(stats["wins"]+stats["losses"]):+.2f}  '
                  f'succ={100*stats["wins"]/(stats["wins"]+stats["losses"]):.0f}%')
            render()
            pygame.time.wait(600)
            env.reset()
            stats['episode'] += 1
            stats['steps_in_ep'] = 0
            stats['ep_reward'] = 0.0

        render()
        pygame.time.wait(RUN_STEP_MS)
        clock.tick(FPS)


def run_session(rows, cols, gamma, step_cost, slip):
    """One full pipeline: setup -> train -> run. Always fresh env."""
    env = Environement(rows=rows, cols=cols, step_cost=step_cost, slip=slip)
    graphics = Graphics(env)

    choice = setup_rewards(env, graphics)
    if choice == 'quit':
        return 'quit'
    if choice == 'back':
        return 'back'

    agent = AI_Agent(env)
    agent.gamma = gamma

    print(f'\n--- Training: VI  gamma={gamma}  step_cost={step_cost}  slip={slip} ---')
    result = train(agent, graphics)
    if result == 'quit':
        return 'quit'

    print('\n--- Converged ---')
    print_grid('Value*', agent.Value)
    print_policy(env, agent.Policy)

    pygame.time.wait(800)
    return run_agent(env, agent, graphics)


def main():
    pygame.init()
    rows, cols, gamma = 4, 4, 0.9
    step_cost, slip = 0.0, 0.0
    completed = False
    menu = MenuScreen()

    while True:
        choice = menu.post_run_menu() if completed else menu.main_menu()

        if choice == 'quit':
            break
        if choice == 'settings':
            result = menu.settings_menu(rows, cols, gamma, step_cost, slip)
            if result is None:
                break
            rows, cols, gamma, step_cost, slip = result
            completed = False
            continue
        if choice in ('start', 'again'):
            res = run_session(rows, cols, gamma, step_cost, slip)
            if res == 'quit':
                break
            completed = (res == 'done')
            continue

    pygame.quit()


if __name__ == '__main__':
    main()
