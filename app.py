import streamlit as st
import matplotlib.pyplot as plt
from maze_dataset import create_maze

def visualize_maze(maze):
    start = maze.start
    end = maze.end
    grid = maze.grid

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(grid, cmap="gray_r")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

maze = create_maze()
st.pyplot(visualize_maze(maze))

