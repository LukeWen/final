import yal
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict

# Luke: This function `draw_placement` is responsible for visualizing the layout of components within a specified area.
# Luke: `components` is a list of dictionaries where each dictionary represents a component with attributes such as position and size.
# Luke: `area` is an integer representing the total area allocated for the layout, though it's not directly used in this function.
def draw_placement(components:List[Dict] = [], area:int = 0):
    # Luke: Configure Matplotlib to use a non-interactive backend ('Agg'), 
    # which is suitable for saving plots as files rather than displaying them interactively.
    matplotlib.use('Agg')  # Non-interactive backend
    
    # Luke: Create a figure and axis object for plotting. The figure size is set to 15x10 inches.
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Luke: Set the limits for the x-axis and y-axis to 3300 and 2400, respectively.
    # Luke: The aspect ratio is set to 'equal' to ensure that units are represented equally on both axes.
    # ax.set_xlim(0, 220) # ratio for example
    # ax.set_ylim(0, 160) # ratio for example
    ax.set_xlim(0, 3300)
    ax.set_ylim(0, 2400)

    ax.set_aspect('equal')
    
    # Luke: Invert the y-axis to match the typical layout coordinate system where the origin is at the top-left.
    plt.gca().invert_yaxis()
    
    # Luke: Hide the axis to focus on the layout itself without any grid or border.
    plt.axis('off')

    # Luke: Extract the last component from the list, which represents the boundary of the layout area.
    boundary = components[-1]
    
    # Luke: Draw the boundary as a rectangle with dashed lines, without filling the rectangle.
    # Luke: The boundary is drawn using the 'black' color, and its z-order is set to 2, which defines the drawing order.
    ax.add_patch(
        patches.Rectangle(
            (boundary['xmin'], boundary['ymin']),
            boundary['width'], boundary['height'],
            fill=False, edgecolor='black', linestyle='--', zorder=2
        )
    )

    # Luke: Iterate through the list of components, excluding the last one (which is the boundary).
    for component in components[:-1]:  # Exclude boundary
        # Luke: Reset the axis position to ensure no offset or distortion in placement.
        ax.reset_position()
        
        # Luke: Draw each component as a rectangle filled with the color specified in its dictionary.
        # Luke: The z-order is set to 2 to ensure these components are drawn above any other elements with lower z-order.
        ax.add_patch(
            patches.Rectangle(
                (component['xmin'], component['ymin']),
                component['width'], component['height'],
                facecolor=component['color'], zorder=2
            )
        )
        
        # Luke: Add a text label at the center of each component rectangle to display the component's index (`idx`).
        # Luke: The text color is calculated as the reverse of the component's color to ensure visibility.
        ax.text(
            component['xmin'] + component['width'] / 2,
            component['ymin'] + component['height'] / 2,
            component['idx'],
            color = yal.util.reverse_color_hex(component['color']),
            weight ='bold',
            ha = 'center', va='center', zorder=3
        )

    # Luke: Save the resulting plot as a PNG file in the specified directory with a filename that includes a timestamp.
    plt.savefig('./onpolicy/parser/output/'+ file + str(time.time())+".png")

# Luke: select a file under input directory.
file = 'example'

# Luke: Use the yal library to read the input file, which is expected to be in a specific format ('.yal').
# Luke: The modules extracted from the file are stored in the `modules` variable.
modules = yal.read('./onpolicy/parser/input/' + file + '.yal')

# Luke: If no modules are returned (i.e., `modules` is None), raise an error indicating that something went wrong with the file parsing.
if modules is None:
    raise ValueError("No modules were returned by yal.read. Please check the input file and parsing logic.")

# Luke: Convert the modules into a list of participants (i.e., components) using a utility function from the yal library.
participants = yal.util.as_participants(modules)

# Luke: Print the participants to the console for debugging or verification purposes.
print(participants)

# Luke: Call the `draw_placement` function to visualize the layout of the components based on the `participants` list.
draw_placement(components=participants)
