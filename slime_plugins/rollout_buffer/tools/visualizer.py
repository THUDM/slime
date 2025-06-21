import os
import threading
import time

import matplotlib.pyplot as plt


class BufferStatsVisualizer:
    def __init__(self, time_window=60):
        """
        Initialize the buffer statistics visualizer

        Args:
            time_window (int): Time window in seconds for each data point (default: 60s)
        """
        self.time_window = time_window
        self.data_points = []  # List to store data points
        self.timestamps = []  # List to store timestamps
        self.start_time = time.time()
        self.last_window_start = self.start_time
        self.window_count = 0  # Counter for current window
        self.args_dict = None  # Store args for filename

        # Initialize the plot
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        (self.line,) = self.ax.plot([], [], "b-", label="Data Points per Window")

        # Set up the plot
        self.ax.set_xlabel("Time (minutes)")
        self.ax.set_ylabel("Data Points per 60s Window")
        self.ax.set_title("Buffer Statistics - Data Points per 60s Window")
        self.ax.grid(True)
        self.ax.legend()

        # Start the update thread
        self.running = True
        self.update_thread = threading.Thread(target=self._update_plot)
        self.update_thread.daemon = True
        self.update_thread.start()

    def set_args(self, args_dict):
        """Set the args dictionary for filename generation"""
        self.args_dict = args_dict

    def add_data_point(self, _):
        """Add a new data point to the statistics"""
        current_time = time.time()
        self.window_count += 1  # Increment counter for current window

        # Check if we've reached the end of a time window
        if current_time - self.last_window_start >= self.time_window:
            # Calculate the time in minutes since start
            time_in_minutes = (current_time - self.start_time) / 60

            # Add the data point and timestamp
            self.data_points.append(self.window_count)
            self.timestamps.append(time_in_minutes)

            # Save the plot after adding new data point
            self.save_plot()

            # Reset for next window
            self.last_window_start = current_time
            self.window_count = 0

    def _update_plot(self):
        """Update the plot periodically"""
        while self.running:
            if self.data_points:  # Only update if we have data
                self.line.set_data(self.timestamps, self.data_points)
                self.ax.relim()
                self.ax.autoscale_view()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            time.sleep(1)  # Update every second

    def save_plot(self):
        """Save the current plot to a file"""
        timestamp = self.start_time

        # Create filename based on args and timestamp
        if self.args_dict:
            # Extract key parameters from args
            key_params = []
            for key in ["task_type", "num_repeat_per_sample", "group_size"]:
                if key in self.args_dict:
                    key_params.append(f"{key}_{self.args_dict[key]}")

            filename = f"buffer_stats_{'_'.join(key_params)}_{timestamp}.png"
        else:
            filename = f"buffer_stats_{timestamp}.png"

        # Create directory if it doesn't exist
        os.makedirs("buffer_stats", exist_ok=True)
        filepath = os.path.join("buffer_stats", filename)

        # Save the plot
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {filepath}")

    def close(self):
        """Close the visualizer and clean up"""
        self.running = False
        if self.update_thread.is_alive():
            self.update_thread.join()
        plt.close(self.fig)


# Example usage:
if __name__ == "__main__":
    visualizer = BufferStatsVisualizer(time_window=60)
    visualizer.set_args({"task_type": "test", "num_repeat_per_sample": 16})

    # Simulate some data
    try:
        for i in range(1000):
            visualizer.add_data_point(1)  # Just increment the counter
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.close()
