# CLAUDE.md — GridWorld_2

## Purpose

Personal RL learning project. User studies Dynamic Programming methods (Policy Evaluation, Policy Iteration, Value Iteration) on a configurable GridWorld MDP with pygame visualization.

**Teach, don't just code.** Prefer explanations + small guided edits over writing full solutions. When user asks for an algorithm, walk through the math (Bellman equation, update rule) before/with the code.

## Project shape

- Small pygame + numpy project. No tests, no package layout.
- Single directory. Run via `python Game.py`.
- Files: `Game.py`, `Menu.py`, `Environement.py` (sic — keep typo, it's the actual module name), `Agent.py`, `Action.py`, `Graphics.py`.

## Architecture

**State machine in `Game.py::main`:**
```
[main_menu] -> start -> run_session(rows, cols, gamma) -> [post_run_menu] -> again/settings/quit
            -> settings -> settings_menu -> [main_menu]
            -> quit
```

**`run_session` pipeline** (always fresh `Environement`):
```
setup_rewards (click placement)
  -> train (Value_Iteration with callback)
  -> run_agent (pygame loop, ESC to exit)
```

**Menus** in `Menu.py`:
- `Button`, `Slider` widgets
- `MenuScreen` instantiated once in `main`, persisted across loop
- `main_menu()` / `post_run_menu()` / `settings_menu()` all call `self.show()` to reset display to 480×420

## MDP facts

- `Environement(rows, cols)` — dynamic size, default 4×4
- Start: `env.start` (default `(0,0)`, settable by right-click in setup)
- States: `(row, col)` tuples
- Actions: `Action` enum — `UP=0, RIGHT=1, LEFT=2, DOWN=3`
- Transitions: deterministic. `env.move(state, action)` clips at walls (no-op on edge)
- Rewards: `board[new_state]` (0 / +1 / -1)
- Terminals: `env.end_of_game(s)` true when `board[s] != 0`
- Discount `gamma`: set on agent after construction (`agent.gamma = gamma`)
- `env.start` + `env.set_reward(cell, value)` are mutation hooks for setup phase

## Agent

- `Policy: np.ndarray (rows, cols)` — int action ids. Init = 3 (DOWN)
- `Value: np.ndarray (rows, cols)` — float. Init = 0
- Action lookup: `Action(self.Policy[state])`
- `_sweep(update_fn, callback)` — shared loop for `policy_eval` + `Value_Iteration`
- `q_value(s, a)` — handles terminal (V=0 absorbing)
- `best_action_value(s)` — returns `(Action, q)` argmax

## Implemented algorithms

- `policy_eval(callback=None)`
- `Policy_improv()` — returns stable bool
- `Policy_Iteration(callback=None)`
- `Value_Iteration(callback=None)` — used by Game.py

Callback signature: `(iteration, delta, V, Policy) -> bool`. Returning truthy stops the sweep loop early.

## Visualization (Graphics.py)

- Dynamic window size — square size auto-fits via `MAX_WINDOW=720`, `MIN_SQUARE=60`
- `draw_training(V, Policy, iter, delta, sub)` — main learning view
- `shade_values(V)` — green/red proportional to |V|
- Header 2 lines: top = iter+delta, bottom = gamma/mode/hint
- `pixel_to_cell(pos)` — for click-to-place reward in setup
- Fonts cached in `__init__`

## User interactions (Game.py)

**Setup (`setup_rewards`):**
- Left-click empty cell: cycle `+1 → -1 → 0`
- Right-click empty cell: move start
- ENTER: train (need ≥1 goal)
- ESC: back to menu

**Training (`train`):**
- SPACE: one sweep
- ENTER: toggle auto (AUTO_STEP_MS=400)
- ESC: skip to convergence

**Run (`run_agent`):**
- ESC: back to menu

## Style

- Match existing style: PascalCase methods (`Policy_Iteration`), capitalised attrs (`Policy`, `Value`, `Reward`). Don't refactor names — user wrote them.
- Keep `Environement` spelling. Don't rename.
- No new dependencies beyond `pygame-ce`, `numpy`.
- No tests requested. Don't add unless asked.

## Don't

- Don't rewrite working code "for cleanliness".
- Don't add type hints everywhere — match local style.
- Don't add logging frameworks, configs, abstractions.
- Don't create extra files unless user asks.
- Don't persist board state between Train Again sessions — user explicitly wants fresh grid each time.
