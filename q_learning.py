import numpy as np


def get_default_path():
    return {
        1: (1, 10),
        2: [1, 7, 9],
        3: [i for i in range(1, 8)] + [9],
        4: [3, 7],
        5: (0, 11),
        6: [5],
        7: (1, 10),
        8: [3, 7],
        9: (0, 11),
    }


def create_grid(m, n, terminal_reward, step_reward, terminal_states=((0, 5),)):
    grid = np.full((m, n), -terminal_reward)
    for terminal in terminal_states:
        grid[terminal] = terminal_reward
    for i, item in get_default_path().items():
        if isinstance(item, list):
            grid[i, item] = step_reward
        else:
            grid[i, item[0] : item[1]] = step_reward
    return grid


def get_start_loc(grid):
    rows, cols = np.where(grid == -1)
    idx = np.random.choice(len(rows))
    return rows[idx], cols[idx]


def get_action(epsilon, q_values, row, col, actions):
    if np.random.random() < epsilon:
        return np.argmax(q_values[row, col])
    else:
        return np.random.randint(actions)


def update_location(x, y, action, grid):
    if action == 0 and x > 0:
        return x - 1, y
    if action == 1 and y < grid.shape[1] - 1:
        return x, y + 1
    if action == 2 and x < grid.shape[0] - 1:
        return x + 1, y
    if action == 3 and y > 0:
        return x, y - 1
    return x, y


def process_ep(
    grid,
    terminal_reward,
    epsilon,
    q_values,
    actions,
    discount_factor,
    learning_rate,
    get_path=False,
    start_point=None,
):
    x1, y1 = start_point or get_start_loc(grid)
    path = [(x1, y1)]
    while grid[x1, y1] != terminal_reward:
        action = get_action(epsilon, q_values, x1, y1, actions)
        x0, y0 = x1, y1
        x1, y1 = update_location(x1, y1, action, grid)
        if get_path:
            path.append((x1, y1))
        reward = grid[x1, y1]
        q0 = q_values[x0, y0, action]
        td = reward + (discount_factor * np.max(q_values[x1, y1])) - q0
        q1 = q0 + learning_rate * td
        q_values[x0, y0, action] = q1
    if path:
        return path


def train(
    episodes, grid, terminal_reward, epsilon, actions, discount_factor, learning_rate
):
    q_values = np.zeros((*grid.shape, actions))
    for _ in range(episodes):
        process_ep(
            grid,
            terminal_reward,
            epsilon,
            q_values,
            actions,
            discount_factor,
            learning_rate,
        )
    return q_values


if __name__ == '__main__':
    g = create_grid(11, 11, 100, -1)
    q = train(1000, g, 100, 0.9, 4, 0.9, 0.9)
    print(process_ep(g, 100, 1.0, q, 4, 0.9, 0.9, True, (5, 0)))
