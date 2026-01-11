from maze_dataset import create_dataset


def format_path(path):
    directions = []
    for i in range(len(path) - 1):
        curr_pos = path[i]
        next_pos = path[i+1]

        row_diff = next_pos[0] - curr_pos[0]
        col_diff = next_pos[1] - curr_pos[1]

        if row_diff == 1 and col_diff == 0:
            directions.append("down")
        elif row_diff == -1 and col_diff == 0:
            directions.append("up")
        elif row_diff == 0 and col_diff == 1:
            directions.append("right")
        elif row_diff == 0 and col_diff == -1:
            directions.append("left")
        else:
            raise ValueError(f"Invalid move from {curr_pos} to {next_pos}")
        
    return directions


def format_dataset(maze_data):
    dataset = []
    for maze in maze_data:
        path = " ".join(format_path(maze['path']))
        res = f"""Input format:\n{maze['maze']}\n\nOutput:\n{path}"""
        dataset.append(res)

    return dataset


training_data = create_dataset(num=1000)
training_data = format_dataset(training_data)
print(training_data[0])