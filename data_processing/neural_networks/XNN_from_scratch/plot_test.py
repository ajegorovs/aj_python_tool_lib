import matplotlib.pyplot as plt
import numpy as np, time
import asyncio

# Function to update the plot
def update_plot(x_data, y_data, plt_range):
    plt.clf()  # Clear the previous plot
    plt.plot(x_data, y_data, label='Updated Data')
    plt.xlim(*plt_range)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.draw()
    plt.pause(0.001)  # Pause to allow the plot to update

# Sample loop with plot updates
x_values = np.linspace(0, 10, 100)
y_values = np.sin(x_values)

plt.ion()  # Turn on interactive mode
time.sleep(1)
for i in range(len(x_values)):
    update_plot(x_values[:i], y_values[:i],(0,10))
    time.sleep(0.2)

plt.ioff()  # Turn off interactive mode to keep the plot window open
plt.show()
