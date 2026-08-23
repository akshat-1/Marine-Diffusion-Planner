import numpy as np

# CommonOcean I/O modules
from commonocean.common.file_reader import CommonOceanFileReader
from commonocean.common.file_writer import CommonOceanFileWriter, OverwriteExistingFile

# CommonOcean core scenario modules
from commonocean.scenario.scenario import Scenario, Tag
from commonocean.scenario.obstacle import StaticObstacle, ObstacleType

# CommonRoad geometry, state, and planning modules (Used by CommonOcean)
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.state import InitialState
from commonroad.planning.planning_problem import PlanningProblemSet  # FIXED: Added import

def create_and_save_scenario():
    # 1. Initialize a new empty Scenario
    scenario = Scenario(dt=0.1, scenario_id="ZAM_Sample-1_1_T-1")

    # 2. Define the physical shape of a vessel (Length: 50m, Width: 15m)
    vessel_shape = Rectangle(length=50.0, width=15.0)

    # 3. Define the initial state USING InitialState
    initial_state = InitialState(
        position=np.array([100.0, -20.0]), 
        orientation=1.57,  # Facing roughly North (Pi/2)
        time_step=0
    )

    # 4. Create a Static Obstacle (e.g., an anchored ship or a buoy)
    anchored_ship = StaticObstacle(
        obstacle_id=scenario.generate_object_id(), 
        obstacle_type=ObstacleType.UNKNOWN, 
        obstacle_shape=vessel_shape, 
        initial_state=initial_state
    )

    # 5. Add the ship to the scenario environment
    scenario.add_objects(anchored_ship)

    # 6. Define metadata required for the benchmark XML format
    author = "Research Team"
    affiliation = "Marine Robotics Lab"
    source = "Procedurally Generated"
    tags = {Tag.OPENSEA} 

    # 7. Initialize an empty Planning Problem Set
    planning_problem_set = PlanningProblemSet()  # FIXED: Created an empty set

    # 8. Initialize the File Writer
    writer = CommonOceanFileWriter(
        scenario=scenario, 
        planning_problem_set=planning_problem_set,  # FIXED: Replaced None with the empty set
        author=author, 
        affiliation=affiliation, 
        source=source, 
        tags=tags
    )

    # 9. Write the scenario out to an XML file
    output_filename = "ZAM_Sample-1_1_T-1.xml"
    writer.write_to_file(output_filename, OverwriteExistingFile.ALWAYS)
    
    print(f"Success! CommonOcean XML saved to: {output_filename}")

if __name__ == "__main__":
    create_and_save_scenario()