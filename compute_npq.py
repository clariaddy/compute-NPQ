###Importing Libraries and Modules:

import os
import numpy as np
import scipy as sp
import scipy.interpolate
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

###Setting up Directories:

# Get the current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define input and output directories
data_dir = os.path.join(script_dir, 'data')
output_dir = os.path.join(script_dir, 'out')

# Create the output directory if it doesn't exist
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

###Creating the Main Figure and Axes:

# Create the main figure and axes for the polar plot
main_fig, main_axe = plt.subplots(ncols=1, nrows=1, figsize=(8, 8), subplot_kw={'projection': 'polar'})
main_axe.set_title('Mean intensity for all inputs')

###Image Processing Loop:

# Loop over a range of 10 iterations
for i in range(1):
    # Create an empty dataframe for each iteration
    cell_dataframe = pd.DataFrame()

    # Loop over two different types of images
    for j in range(2):
        # Set the value of 't' based on the inner loop index
        t = 1 if j == 0 else 15

        # Construct the image filename based on the loop indices
        img_name = f'CT{t}{i:02d}.png'

        # Get the full image file path
        img_path = os.path.join(data_dir, img_name)

        # Check if the image file exists
        if not os.path.isfile(img_path):
            raise RuntimeError(f"Could not find image '{img_path}'")

        print(f"Handling image '{img_path}'...")

        ###Image Loading and Processing:

        # Load the image and convert it to a float32 numpy array
        img = Image.open(img_path)
        img = np.asarray(img).astype(np.float32)

        # Calculate the pixel coordinates
        Ny, Nx = img.shape
        y = np.arange(Ny).astype(np.float32)
        x = np.arange(Nx).astype(np.float32)

        # Determine the approximate center coordinates
        cx = (img / img.sum() * x[None, :]).sum()
        cy = (img / img.sum() * y[:, None]).sum()

        # Determine the approximate radius
        sx = np.cumsum(img.sum(axis=0))
        sy = np.cumsum(img.sum(axis=1))
        dx = x[np.where(sx > 0.99 * sx[-1])[0][0]] - x[np.where(sx > 0.01 * sx[-1])[0][0]]
        dy = y[np.where(sy > 0.99 * sy[-1])[0][0]] - y[np.where(sy > 0.01 * sy[-1])[0][0]]
        r0 = (dx + dy) / 4

        # Use the first approximation to determine the final circle
        Mx = (x < cx - 1.2 * r0) | (x > cx + 1.2 * r0)
        My = (y < cy - 1.2 * r0) | (y > cy + 1.2 * r0)
        I = img.copy()
        I[:, Mx] = 0.0
        I[My, :] = 0.0
        cx = (I / I.sum() * x[None, :]).sum()
        cy = (I / I.sum() * y[:, None]).sum()
        sx = np.cumsum(I.sum(axis=0))
        sy = np.cumsum(I.sum(axis=1))
        dx = x[np.where(sx > 0.99 * sx[-1])[0][0]] - x[np.where(sx > 0.01 * sx[-1])[0][0]]
        dy = y[np.where(sy > 0.99 * sy[-1])[0][0]] - y[np.where(sy > 0.01 * sy[-1])[0][0]]
        r0 = (dx + dy) / 4

        # Interpolate values and compute intensity
        interpolator = sp.interpolate.RectBivariateSpline(y, x, img)
        Nr = 2 * int(np.sqrt(Nx ** 2 + Ny ** 2))
        Nθ = 1024
        r = np.linspace(0, r0, Nr, endpoint=True)
        θ = np.linspace(0, 2 * np.pi, Nθ, endpoint=True)
        X = cx + r[:, None] * np.cos(θ[None, :])
        Y = cy + r[:, None] * np.sin(θ[None, :])
        polar = interpolator(Y.ravel(), X.ravel(), grid=False).T.reshape(X.shape)
        intensity = polar.mean(axis=0)

        ###Plotting and Saving:

        # Create a new figure and axes for plotting
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(14, 8))

        ax = axes[0, 0]
        ax.imshow(img, origin='lower', cmap='gist_heat')
        ax.scatter([cx], [cy], zorder=1, c='w')
        ax.add_patch(plt.Circle((cx, cy), r0, edgecolor='w', facecolor='none'))
        ax.set_title('Approximate fit with circle')
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')

        axes[0, 1].remove()
        axes[0, 1] = fig.add_subplot(2, 2, 2, projection='polar')
        axes[0, 1].set_title('Mean intensity')
        axes[0, 1].plot(θ, intensity)

        ax = axes[1, 0]
        ax.imshow(polar, origin='lower', extent=[0, 360, 0, r0], cmap='gist_heat')
        ax.set_xlabel('θ (°)')
        ax.set_ylabel('r (pixels)')
        ax.set_ylim(0, r0)
        ax.set_xlim(0, 360)

        ax = axes[1, 1]
        for k in range(4):
            ax.plot(r, polar[:, int(k * Nθ / 4)], label=f'θ={k * 360 / 4}°')
        ax.set_xlabel('distance to center (pixels)')
        ax.set_ylabel('intensity')
        ax.set_xlim(0, r0)
        ax.legend(ncol=2)

        # Plot the intensity values in the main axes
        if j == 0:
            main_axe.plot(θ, intensity, c='b', label='T=0' if i == 0 else None)
        else:
            main_axe.plot(θ, intensity, c='r', label='T=15' if i == 0 else None)

        ###Finalizing and Saving:

        main_axe.legend()

        # Save the figure as an SVG file
        fig.savefig(os.path.join(output_dir, f'CT{t}{i:02d}.svg'), dpi=600)
        plt.close(fig)

        # Store the intensity values and angles in the dataframe
        if j == 0:
            cell_dataframe.insert(0, 'theta', θ)
        cell_dataframe.insert(1 + j, f'T={t}', intensity)

    # Save the dataframe as a CSV file
    cell_dataframe.to_csv(os.path.join(output_dir, f'./cell_{i}.csv'))

# Save the main figure
main_fig.savefig(os.path.join(output_dir, 'cell.svg'), dpi=300)
