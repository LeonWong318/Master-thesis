import random
from shapely.geometry import Polygon

from pkg_map.map_geometric import GeometricMap
from pkg_obstacle import geometry_tools

from pkg_dqn.environment import MapDescription
from pkg_dqn.utils.map import generate_map_dynamic, generate_map_corridor, generate_map_mpc
from pkg_dqn.utils.map import generate_map_scene_1, generate_map_scene_2

from typing import List, Tuple

class Inflator:
    def __init__(self, inflate_margin):
        self.inflate_margin = inflate_margin

    def __call__(self, polygon: List[tuple]):
        shapely_inflated = geometry_tools.polygon_inflate(Polygon(polygon), self.inflate_margin)
        return geometry_tools.polygon_to_vertices(shapely_inflated)


def get_geometric_map(rl_map: MapDescription, inflate_margin: float) -> GeometricMap:
    _, rl_boundary, rl_obstacles, _ = rl_map
    inflator = Inflator(inflate_margin)
    geometric_map = GeometricMap(
        boundary_coords=rl_boundary.vertices.tolist(),
        obstacle_list=[obs.nodes.tolist() for obs in rl_obstacles if obs.is_static],
        inflator=inflator
    )
    return geometric_map

def generate_map(scene:int=1, sub_scene:int=1, sub_scene_option:int=1, generator:bool=True) -> MapDescription:
    """
    MapDescription = Tuple[MobileRobot, Boundary, List[Obstacle], Goal]
    """
    
    if scene == 1:
        map_des = generate_map_scene_1(sub_scene, sub_scene_option)
    elif scene == 2:
        map_des = generate_map_scene_2(sub_scene, sub_scene_option)
    elif scene == 3:
        map_des = generate_map_mpc(11)()
    else:
        raise ValueError(f"Scene {scene} not recognized (should be 1, 2, or 3).")
    
    def _generate_map():
        return map_des

    if generator:
        return _generate_map
    else:
        return map_des
    