"""Constants of robot task scenes for `trajectory_player` and `trajectory_generators`"""
# Import default constants.
from .default import * 

import carb
carb_settings_iface = carb.settings.get_settings()
robot_type = carb_settings_iface.get("/data_collect/robot_type")

# The default constants will be overwritten.
if robot_type == "g1_trihand":
    from .g1_trihand import *
elif robot_type == "g1_inspire":
    from .g1_inspire import *
elif robot_type == "openarm_leaphand":
    from .openarm_leaphand import *
else:
    raise NotImplementedError("Currently only [g1_trihand, g1_inspire, openarm_leaphand].")