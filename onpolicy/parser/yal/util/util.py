"""
yal.util

Utility functions for YAL Modules in Python
"""
from .. import core
from typing import List, Dict
# Luke: colors for block plotting
colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',  # Red, Green, Blue, Yellow, Magenta, Cyan
        '#FFA500', '#800080', '#008000', '#000080', '#FF4500', '#FFD700',  # Orange, Purple, Dark Green, Navy, OrangeRed, Gold
        '#ADFF2F', '#7FFF00', '#32CD32', '#8A2BE2', '#D2691E', '#FF1493',  # GreenYellow, Chartreuse, LimeGreen, BlueViolet, Chocolate, DeepPink
        '#FF6347', '#FFDAB9', '#DC143C', '#00CED1', '#9400D3', '#FF69B4',  # Tomato, PeachPuff, Crimson, DarkTurquoise, DarkViolet, HotPink
        '#B22222', '#8B0000', '#556B2F', '#808000', '#6A5ACD', '#A52A2A',  # FireBrick, DarkRed, DarkOliveGreen, Olive, SlateBlue, Brown
        '#D2B48C', '#9ACD32', '#5F9EA0', '#FF4500', '#DA70D6', '#EE82EE',  # Tan, YellowGreen, CadetBlue, OrangeRed, Orchid, Violet
        '#BDB76B', '#FFD700', '#40E0D0', '#6495ED', '#FF8C00', '#8B008B',  # DarkKhaki, Gold, Turquoise, CornflowerBlue, DarkOrange, DarkMagenta
        '#008B8B', '#B8860B', '#A9A9A9', '#2F4F4F', '#8FBC8F', '#483D8B',  # DarkCyan, DarkGoldenRod, DarkGray, DarkSlateGray, DarkSeaGreen, DarkSlateBlue
        '#00FF7F', '#4682B4', '#D2691E', '#FF7F50', '#ADFF2F', '#DDA0DD',  # SpringGreen, SteelBlue, Chocolate, Coral, GreenYellow, Plum
        '#FF4500', '#DAA520', '#2E8B57', '#87CEFA', '#778899', '#FF6347',  # OrangeRed, GoldenRod, SeaGreen, LightSkyBlue, LightSlateGray, Tomato
        '#7B68EE', '#3CB371', '#BDB76B', '#00FA9A', '#48D1CC', '#C71585',  # MediumSlateBlue, MediumSeaGreen, DarkKhaki, MediumSpringGreen, MediumTurquoise, MediumVioletRed
        '#191970', '#7FFF00', '#F4A460', '#2E8B57', '#D8BFD8', '#FFA07A',  # MidnightBlue, Chartreuse, SandyBrown, SeaGreen, Thistle, LightSalmon
        '#CD5C5C', '#4B0082', '#8A2BE2', '#FFB6C1', '#20B2AA', '#DB7093'   # IndianRed, Indigo, BlueViolet, LightPink, LightSeaGreen, PaleVioletRed
    ]

# Luke: Initialize a dictionary representing a participant with default values.
def _init_participant():
    return { 'idx':         None
           , 'xmin':        0
           , 'ymin':        0
           , 'width':       0
           , 'height':      0
           , 'connections': {} }

# Luke: Generator function to cycle through a predefined list of colors indefinitely.
def color_generator():
    while True:
        for color in colors:
            yield color

# Luke: Initialize the color generator to be used for assigning colors to participants.
color_gen = color_generator()

# Luke: Function to retrieve the next color in sequence from the color generator.
def get_color() -> str:
    """
    Return the next color in sequence.
    """
    return next(color_gen)

# Luke: Function to convert a hex color code to an RGB tuple.
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Luke: Function to convert an RGB tuple back to a hex color code.
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

# Luke: Function to reverse the RGB color, essentially inverting it.
def reverse_color(rgb):
    r, g, b = rgb
    return (255 - r, 255 - g, 255 - b)

# Luke: Function to reverse a hex color by first converting it to RGB, inverting the RGB, and converting it back to hex.
def reverse_color_hex(hex_color):
    rgb = hex_to_rgb(hex_color)
    reversed_rgb = reverse_color(rgb)
    return rgb_to_hex(reversed_rgb)

# Luke: Function to convert a `yal.core.Module` object to a dictionary representing a participant.
def as_participant( module: core.Module, opt_fields: List[str] = None
                  , colorize: bool = True ) -> dict:
    """
    Convert a `yal.core.Module` dataclass object to a `dict` participant.

    Arguments:

    - `module`: The `yal.core.Module` object to convert
    - `opt_fields`: Optional field names (as string) from `yal.core.Module`
        to retain in the participant dict. Options are `'module_type'`,
        `'dimensions'`, `'terminals'`, `'network'`, `'placement'` and
        `'critical_nets'`. (optional, default is `None`)
    - `colorize`: Whether to add a random color to the participant. Otherwise
        the `'color'` field is `None`. (optional, default = True)

    Return:

    A dictionary of the form:

    ```
    { 'idx':         Module.module_name
    , 'xmin':        <lower left x coordinate>
    , 'ymin':        <lower left y coordinate>
    , 'width':       <width of module>
    , 'height':      <height of module>
    , 'connections': {<other idx>: <weight>} }
    ```
    """

    # Luke: Calculate the bounding box dimensions of the module.
    dims        = sorted(module.dimensions)
    x_min,y_min = dims[0]
    x_max,y_max = dims[-1]
    width       = x_max - x_min
    height      = y_max - y_min
    idx         = module.module_name
    
    # Luke: Create a dictionary representation of the module, filtering optional fields if provided.
    mod_dict    = {k: v for k,v in core.as_dict(module).items()
                        if (opt_fields and k in opt_fields)}
    
    # Luke: Start with a copy of the module dictionary and update it with default participant attributes.
    participant = mod_dict.copy()  # Start with a copy of mod_dict
    participant.update(_init_participant())  # Merge with the result of _init_participant()
    
    # Luke: Update the participant dictionary with specific attributes like index, dimensions, and color.
    participant.update({
        'idx':    idx,
        'width':  width,
        'height': height,
        'xmin':   x_min,
        'ymin':   y_min,
        'color':  get_color() if colorize and (idx != 'bound') else None
    })

    return participant

# Luke: Function to determine weighted connections between participants based on shared signals.
def connects(participant: Dict, participants: List[Dict]) -> Dict[str,int]:
    """
    Weighted connections between participants

    Arguments:

    - `participant`: The participant in question

    - `participants`: Other participants (can include the former)

    Return:

    A dictionary of the form:

    ```
    {'other participant': <weight of connection>}
    ```
    """
    name = participant['module_name']
    
    # Luke: Determine the set of signals for the participant, excluding certain common signals.
    sigs = set(participant.get('signal_names', []))
    
    # Luke: Create a dictionary of connections to other participants based on shared signals, with weights representing the number of shared signals.
    cons =  { p['module_name']: len((set(p.get('signal_names', [])) & sigs) - {'G', 'P'})
              for p in participants if p['module_name'] not in [name, 'bound'] }
    return cons

# Luke: Function to convert a list of `yal.core.Module` objects into a list of participant dictionaries.
def as_participants( modules: List[core.Module]
                   , colorize: bool      = True
                   , module_type: bool   = False
                   , dimensions: bool    = False
                   , terminals: bool     = False
                   , network: bool       = True
                   , placement: bool     = False
                   , critical_nets: bool = False
                   ) -> List[Dict]:
    """
    Convert a list of `yal.core.Module`s to a list of dictionaries.

    Arguments:

    - `modules`: A list of `yal.core.Module` objects.
    - `colorize`: Whether to add a random color to the participant (optional, default = `True`)
    - `module_type`: Whether to retain the `yal.core.Module.module_type` field
        (optional, default = `False`)
    - `dimensions`: Whether to retain the `yal.core.Module.dimensions` field
        (optional, default = `False`)
    - `terminals`: Whether to retain the `yal.core.Module.terminals` field
        (optional, default = `False`)
    - `network`: Whether to retain the `yal.core.Module.network` field
        (optional, default = `True`)
    - `placement`: Whether to retain the `yal.core.Module.placement`
        field (optional, default = `False`)
    - `critical_nets`: Whether to retain the `yal.core.Module.critical_nets` field
        (optional, default = `False`)

    Return:
    
    A list of participant dictionaries.
    """

    # Luke: Filter the optional fields based on the provided arguments.
    all_fields = zip( [ module_type, dimensions, terminals
                      , network, placement, critical_nets ]
                    , [ 'module_type', 'dimensions', 'terminals'
                      , 'network', 'placement', 'critical_nets' ] )

    fields = [f for c,f in list(all_fields) if c]

    # Luke: Convert each module into a participant dictionary, applying colorization if specified.
    parts = [as_participant(m, opt_fields=fields, colorize=colorize) for m in modules]

    participants = parts

    # Luke: Establish connections between participants if the network field is retained.
    bound   = [p for p in participants if p['idx'] == 'bound']
    network = bound[0].get('network', []) if bound else []
    cons    = {n['module_name'] : connects(n, network) for n in network}
    
    # Luke: Update each participant with the connections and return the final list of participants.
    return [{**p, **{'connections': cons.get(p['idx'])}} for p in participants]