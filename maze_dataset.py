from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.solve.Collision import Collision
import random 
import matplotlib.pyplot as plt
from tqdm import tqdm

SIZE = 7    # we use only square mazes


def sample_loc(size=7, exclude_boundary=True):
    if exclude_boundary:    
        return (random.randrange(1, size), random.randrange(1, size))
    else:
        return(random.randrange(size), random.randrange(size))
    

def create_maze(size=7, start=None, end=None, seed=None):
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
    maze.solver = Collision()
    maze.solve()


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
    for _ in tqdm(range(num), desc="Creating mazes: "):
        curr_size = random.randrange(size_min, size_max)
        maze = create_maze(size=curr_size)
        grid = str(maze)
        solve_maze(maze)
        soln = maze.solutions[0]
        training_data.append({
            'maze': grid,
            'path': soln
        })

    return training_data


def test():
    mz = create_maze(seed=42)
    solve_maze(mz)
    visualize_maze(mz)
