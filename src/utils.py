
import os
import matplotlib.pyplot as plt

def plot_loss(loss_values, save_path='reporting/loss', filename='loss_plot.png', title='Training Loss', xlabel='Iterations', ylabel='Loss'):
    """
    Plot and save a graph of loss values over iterations.
    
    Args:
        loss_values (list): List of loss values to plot
        save_path (str): Directory to save the plot
        filename (str): Name of the output file
        title (str): Title of the plot
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    # Save the plot
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path)
    plt.close()
    
    print(f"Loss plot saved to {full_path}")
