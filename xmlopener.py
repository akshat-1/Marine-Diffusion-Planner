import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from commonocean.common.file_reader import CommonOceanFileReader
from commonocean.visualization.draw_dispatch_cr import draw_object
from matplotlib.widgets import Button

# Load the scenario file
file_path = "/run/media/akshat/Akshat_USB/generated_scenarios4/scenario_0101.xml"
scenario, planning_problem_set = CommonOceanFileReader(file_path).open()

# Automatically find the last timestep available
max_timestep = max([obs.prediction.final_time_step for obs in scenario.dynamic_obstacles])
print(f"Loaded scenario. Total timesteps: 0 to {max_timestep}")

class InteractiveScenarioViewer:
    def __init__(self, scenario, planning_problem_set, max_timestep):
        self.scenario = scenario
        self.planning_problem_set = planning_problem_set
        self.max_timestep = max_timestep
        self.current_frame = 0

        # Set up the main figure and make room at the bottom for the button
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("CommonOcean Interactive Viewer")
        plt.subplots_adjust(bottom=0.15)

        # Create a button axis [left, bottom, width, height]
        ax_button = plt.axes([0.4, 0.03, 0.2, 0.075])
        self.btn_next = Button(ax_button, 'Next Frame ➔', color='lightgray', hovercolor='skyblue')
        self.btn_next.on_clicked(self.on_click_next)

        # Bind keyboard events (Right Arrow to go forward, Left Arrow to go back)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Draw the initial frame
        self.update_plot()

    def update_plot(self):
        # FIX: Explicitly set self.ax as the active current axes before drawing
        plt.sca(self.ax)
        
        self.ax.clear()
        
        draw_params = {
            'time_begin': self.current_frame, 
            'dynamic_obstacle': {
                'draw_shape': True,
                'draw_icon': True,
                'draw_trajectory': True
            }
        }
        
        draw_object(self.scenario, draw_params=draw_params)
        if self.planning_problem_set:
            draw_object(self.planning_problem_set, draw_params={'time_begin': self.current_frame})
            
        self.ax.set_aspect('equal')
        self.ax.autoscale()
        self.ax.set_title(
            f"Timestep: {self.current_frame} / {self.max_timestep}\n"
            f"(Controls: Click 'Next Frame' button, or use Left/Right Arrow keys)", 
            fontsize=11
        )
        self.fig.canvas.draw_idle()

    def on_click_next(self, event):
        """Triggered when the on-screen button is clicked."""
        if self.current_frame < self.max_timestep:
            self.current_frame += 1
            self.update_plot()
        else:
            print("Reached the final timestep of the scenario.")

    def on_key_press(self, event):
        """Triggered when keyboard arrow keys are pressed."""
        if event.key == 'right':
            if self.current_frame < self.max_timestep:
                self.current_frame += 1
                self.update_plot()
        elif event.key == 'left':
            if self.current_frame > 0:
                self.current_frame -= 1
                self.update_plot()

# Launch the interactive viewer
viewer = InteractiveScenarioViewer(scenario, planning_problem_set, max_timestep)
plt.show()