from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.solve.ShortestPaths import ShortestPaths
import random 
import matplotlib.pyplot as plt
from tqdm import tqdm
import json


SIZE = 7    # we use only square mazes


def sample_loc(size=7, exclude_boundary=True):
    if exclude_boundary:    
        return (random.randrange(1, size), random.randrange(1, size))
    else:
        return(random.randrange(size), random.randrange(size))
    

def create_maze(size, start=None, end=None, seed=None):
    if start == end and start != None:
        raise ValueError("start and end can't be same")

    mz = Maze(seed)
    mz.generator = Prims(h=size, w=size)
    mz.generate()

    if start == None and end == None:
        mz.generate_entrances(start_outer=False, end_outer=False)

    if start is not None:
        mz.start = start
    if end is not None:
        mz.end = end
    
    return mz


def solve_maze(maze):
    maze.solver = ShortestPaths()
    maze.solve()

    solns = []
    for soln in maze.solutions:
        soln.insert(0, maze.start)
        soln.append(maze.end)
        solns.append(soln)

    maze.solutions = solns


# visualize the maze 
def visualize_maze(maze):
    solve_path = maze.solutions
    grid = maze.grid
    start = maze.start
    end = maze.end
    _, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(grid, cmap="gray_r")

    # plt start and end
    ax.scatter(start[1], start[0], c="green", s=100)
    ax.scatter(end[1], end[0], c="red", s=100)

    # plt soln
    rows, cols = zip(*solve_path[0])
    ax.plot(cols, rows, color="blue", linewidth=10)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


def create_dataset(size_min=5, size_max=10, num=10000):
    training_data = []
    for i in tqdm(range(num), desc="Creating mazes"):
        curr_size = random.randrange(size_min, size_max)
        maze = create_maze(size=curr_size)
        grid = str(maze)
        solve_maze(maze)
        paths = maze.solutions
        
        shortest_length = min(len(path) for path in paths) if paths else 0
        complexity_ratio = shortest_length / (curr_size * curr_size)

        if complexity_ratio < 0.3:
            difficulty = "easy"
        elif complexity_ratio < 0.6:
            difficulty = "medium"
        else:
            difficulty = "hard"
        
        training_data.append({
            'maze_id': f'maze_{i:06d}',
            'maze': grid,
            'grid': maze.grid.tolist(),
            'start': maze.start,
            'end': maze.end,
            'size': [maze.grid.shape[0], maze.grid.shape[1]],
            'paths': paths,
            'shortest_path_length': shortest_length,
            'num_solutions': len(paths),
            'difficulty': difficulty
        })
    
    return training_data


def check_valid_path(maze, path):
    grid = maze['grid']
    start = maze['start']
    end = maze['end']
    R, C = len(grid), len(grid[0])

    if path[0] != start or path[-1] != end:
        return False

    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return False
        if not (0 <= r2 < R and 0 <= c2 < C):
            return False
        if grid[r2][c2] == 1:
            return False

    return True


def save_to_file(dataset, filename='./maze_data.json'):
    with open(filename, 'w') as f:
        json.dump(dataset, f)

    
def read_from_file(filename='./maze_data.json'):
    with open(filename, 'r') as f:
        dataset = json.load(f)
    return dataset


def test():
    mz = create_maze(seed=420)
    solve_maze(mz)
    visualize_maze(mz)

