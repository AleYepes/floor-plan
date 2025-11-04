from dataclasses import dataclass, field, replace
from re import L
from typing import List, Dict, Optional, Tuple, Literal
from Pynite import FEModel3D
from Pynite.Rendering import Renderer
import numpy as np
import pandas as pd
from itertools import product
from test.test_enum import member
from tqdm.auto import tqdm


class Naming:
    @staticmethod
    def node(name: str, end: str, floor: bool = False) -> str:
        prefix = 'floor ' if floor else ''
        return f'{prefix}{name}{end}'

    @staticmethod
    def member(name: str) -> str:
        return name

    @staticmethod
    def wall(p1_name: str, p2_name: str, side: str) -> str:
        return f'{side} wall {p1_name}-{p2_name}'

    @staticmethod
    def side_wall(side: str) -> str:
        return f'{side} wall'

    @staticmethod
    def header_node(end: str) -> str:
        return f'header_{end}'

    @staticmethod
    def header_member() -> str:
        return 'header'

    @staticmethod
    def tail_header_node(tail_name: str) -> str:
        return f'{tail_name}_header'


@dataclass
class DesignParameters:
    room_length: float
    room_width: float
    room_height: float
    plank_thickness: float
    opening_width: float
    opening_length: float
    opening_x_start: float
    wall_beam_contact_depth: float
    live_load_mpa: float
    wall_thickness: float

    @property
    def floor_y(self):
        return self.room_height / 2

    @property
    def floor_to_ceiling(self):
        return self.floor_y - self.plank_thickness / 2

    @property
    def beam_length(self):
        return self.room_width + self.wall_beam_contact_depth

def load_design_parameters(path: str) -> DesignParameters:
    try:
        params_df = pd.read_csv(path)
        params = params_df.iloc[0].to_dict()
        return DesignParameters(**params)
    except FileNotFoundError:
        raise SystemExit(f"Error: Design parameters file not found at '{path}'")
    except Exception as e:
        raise SystemExit(f"Error loading design parameters from '{path}': {e}")

def load_material_strengths(path: str) -> Dict:
    try:
        materials_df = pd.read_csv(path)
        materials_df.set_index('material', inplace=True)
        return materials_df.to_dict(orient='index')
    except FileNotFoundError:
        raise SystemExit(f"Error: Material strengths file not found at '{path}'")
    except Exception as e:
        raise SystemExit(f"Error loading material strengths from '{path}': {e}")

def load_material_catalog(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise SystemExit(f"Error: Beam catalog file not found at '{path}'")
    except Exception as e:
        raise SystemExit(f"Error loading material catalog from '{path}': {e}")

# Units are mm, N, and MPa (N/mm²)
DEFAULT_PARAMS = load_design_parameters('data/design_parameters.csv')
MATERIAL_STRENGTHS = load_material_strengths('data/material_strengths.csv')
BEAM_CATALOG = load_material_catalog('data/material_catalog.csv')


@dataclass
class CrossSectionGeometry:
    A: float  # Area
    Iy: float  # Second moment about y-axis
    Iz: float  # Second moment about z-axis
    J: float   # Torsional constant
    
    @classmethod
    def from_rectangular(cls, base: float, height: float) -> 'CrossSectionGeometry':
        A = base * height
        b, h = min(base, height), max(base, height)
        J = (b**3 * h) * (1/3 - 0.21 * (b/h) * (1 - (b**4)/(12*h**4)))
        Iy = (height * base**3) / 12
        Iz = (base * height**3) / 12
        return cls(A=A, Iy=Iy, Iz=Iz, J=J)
    
    @classmethod
    def from_i_beam(cls, height: float, flange_width: float, flange_thickness: float, web_thickness: float) -> 'CrossSectionGeometry':
        A_flanges = 2 * flange_width * flange_thickness
        A_web = (height - 2 * flange_thickness) * web_thickness
        A = A_flanges + A_web
        
        Iz_flanges = 2 * (flange_width * flange_thickness**3 / 12 + 
                         flange_width * flange_thickness * ((height - flange_thickness)/2)**2)
        web_height = height - 2 * flange_thickness
        Iz_web = web_thickness * web_height**3 / 12
        Iz = Iz_flanges + Iz_web
        
        Iy_flanges = 2 * (flange_thickness * flange_width**3 / 12)
        Iy_web = web_height * web_thickness**3 / 12
        Iy = Iy_flanges + Iy_web
        
        J = (2 * flange_width * flange_thickness**3 + web_height * web_thickness**3) / 3
        return cls(A=A, Iy=Iy, Iz=Iz, J=J)


@dataclass
class MemberSpec:
    catalog_id: str
    beam_type: Literal['joist', 'tail', 'trimmer', 'header', 'plank']
    quantity: Optional[int] = None
    padding: Optional[float] = None
    
    def __post_init__(self):
        self._catalog_data = BEAM_CATALOG[BEAM_CATALOG['id'] == self.catalog_id].iloc[0]
        self.material = self._catalog_data['material']
        self.base = self._catalog_data['base']
        self.height = self._catalog_data['height']
        self.shape = self._catalog_data['shape']
        self.cost_per_m3 = self._catalog_data['cost_per_m3']
    
    @property
    def section_name(self) -> str:
        return f"sec_{self.catalog_id}"
    
    def get_geometry(self) -> CrossSectionGeometry:
        if self.shape == 'rectangular':
            return CrossSectionGeometry.from_rectangular(self.base, self.height)
        elif self.shape == 'I-beam':
            return CrossSectionGeometry.from_i_beam(self.height, self._catalog_data['flange_width'], self._catalog_data['flange_thickness'], self._catalog_data['web_thickness'])
        else:
            raise ValueError(f"Unknown shape: {self.shape}")
    
    def create_section(self, frame: FEModel3D):
        if self.section_name not in frame.sections:
            geom = self.get_geometry()
            frame.add_section(self.section_name, geom.A, geom.Iy, geom.Iz, geom.J)
    
    def get_volume(self, length: float) -> float:
        return self.get_geometry().A * length # mm^3
    
    def get_cost(self, length: float) -> float:
        volume_m3 = self.get_volume(length) / 1e9  # Convert mm^3 to m^3
        return volume_m3 * self.cost_per_m3
    
    def copy(self, **kwargs):
        return replace(self, **kwargs)


@dataclass
class NodeLocation:
    name: str
    x: float
    y: float
    z: float


@dataclass
class MemberLocation:
    name: str
    node_i: str
    node_j: str
    spec: MemberSpec


def _calculate_evenly_spaced_positions(
    n: int, 
    clear_start: float, 
    clear_end: float, 
    member_base: float,
    distribution: Literal['start_aligned', 'end_aligned', 'centered']) -> List[float]:

    if n == 0:
        return []
    
    if distribution == 'start_aligned':
        centerline_start = clear_start + member_base / 2
        centerline_end = clear_end + member_base / 2
        positions = np.linspace(centerline_start, centerline_end, n + 1).tolist()
        return positions[:-1]
    
    elif distribution == 'end_aligned':
        centerline_start = clear_start - member_base / 2
        centerline_end = clear_end - member_base / 2
        positions = np.linspace(centerline_end, centerline_start, n + 1).tolist()
        return positions[:-1]
    
    elif distribution == 'centered':
        centerline_start = clear_start + member_base / 2
        centerline_end = clear_end - member_base / 2
        positions = np.linspace(centerline_start, centerline_end, n + 2).tolist()
        return positions[1:-1]
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    

def calculate_nodes_and_members(
    east_joists: MemberSpec,
    west_joists: MemberSpec,
    tail_joists: MemberSpec,
    trimmers: MemberSpec,
    header: MemberSpec,
    planks: MemberSpec,
    params: DesignParameters) -> Tuple[List[NodeLocation], List[MemberLocation]]:

    assert trimmers.quantity == 2, "Must have exactly 2 trimmers"
    assert header.quantity == 1, "Must have exactly 1 header"
    
    # Key geometry
    opening_x_end = params.opening_x_start + params.opening_length
    trimmer_E_x = params.opening_x_start - (trimmers.base / 2)
    trimmer_W_x = opening_x_end + (trimmers.base / 2)
    header_z = (params.beam_length - params.opening_width - params.wall_beam_contact_depth/2 - header.base/2)
    
    # Beam centerline positions
    beam_positions = {}
    if east_joists.quantity > 0:
        clear_start = east_joists.padding
        clear_end = trimmer_E_x - trimmers.base / 2
        x_positions = _calculate_evenly_spaced_positions(east_joists.quantity, clear_start, clear_end, east_joists.base, 'start_aligned')
        beam_positions['east'] = [(f'east{i}', x) for i, x in enumerate(x_positions)]
    
    if tail_joists.quantity > 0:
        clear_start = params.opening_x_start + tail_joists.padding
        clear_end = opening_x_end - tail_joists.padding
        x_positions = _calculate_evenly_spaced_positions(tail_joists.quantity, clear_start, clear_end, tail_joists.base, 'centered')
        beam_positions['tail'] = [(f'tail{i}', x) for i, x in enumerate(x_positions)]
    
    if west_joists.quantity > 0:
        clear_start = trimmer_W_x + trimmers.base / 2
        clear_end = params.room_length - west_joists.padding - west_joists.base
        x_positions = _calculate_evenly_spaced_positions(west_joists.quantity, clear_start, clear_end, west_joists.base, 'end_aligned')
        beam_positions['west'] = [(f'west{i}', x) for i, x in enumerate(x_positions)]
    
    beam_positions['trimmer'] = [('trimmer_E', trimmer_E_x), ('trimmer_W', trimmer_W_x)]

    # Plank centerline positions
    clear_start = 0 + params.wall_beam_contact_depth
    clear_end = params.room_length - params.wall_beam_contact_depth
    z_positions = _calculate_evenly_spaced_positions(planks.quantity, clear_start, clear_end, planks.base, 'centered')
    plank_positions = [(f'plank{i}', z) for i, z in enumerate(z_positions)]
    
    # Beam nodes and member locations
    nodes = []
    members = []
    for group_name, group_positions in beam_positions.items():
        for beam_name, x in group_positions:
            nodes.append(NodeLocation(f'{beam_name}_S', x, params.floor_y, 0))
            if group_name == 'tail': # Tails connect to header (header_z), not to wall (beam_length)
                nodes.append(NodeLocation(f'{beam_name}_header', x, params.floor_y, header_z))
                members.append(MemberLocation(name=beam_name, node_i=f'{beam_name}_header', node_j=f'{beam_name}_S', spec=east_joists))
            else:
                nodes.append(NodeLocation(f'{beam_name}_N', x, params.floor_y, params.beam_length))
                members.append(MemberLocation(name=beam_name, node_i=f'{beam_name}_N', node_j=f'{beam_name}_S', spec=east_joists))
    
    nodes.append(NodeLocation('E_header', trimmer_E_x, params.floor_y, header_z))
    nodes.append(NodeLocation('W_header', trimmer_W_x, params.floor_y, header_z))
    members.append(MemberLocation(name='header', node_i='W_header', node_j='E_header', spec=header))

    # Corner nodes
    walls = [('E', 0), ('W', params.room_length)]
    for corner_name, x in walls:
        nodes.append(NodeLocation(f'{corner_name}_S', x, params.floor_y, 0))
        nodes.append(NodeLocation(f'{corner_name}_N', x, params.floor_y, params.beam_length))

    # Plank nodes and member locations
    plank_series_dict = {}
    beam_positions['walls'] = walls # So planks extend to walls
    for plank_name, z in plank_positions:
        plank_nodes = []
        for group_positions in beam_positions.values():
            for beam_name, x in group_positions:
                intersection_node = NodeLocation(f'{beam_name}-{plank_name}', x, params.floor_y, z)
                plank_nodes.append(intersection_node)
                nodes.append(intersection_node)
        plank_series_dict[plank_name] = plank_nodes
    
    for plank_name, plank_nodes in plank_series_dict.items():
        plank_nodes.sort(lambda node: node.x)
        for i, _ in enumerate(plank_nodes[:-1]):
            members.append(MemberLocation(name=f'{plank_name}_{i}', node_i=plank_nodes[i], node_j=plank_nodes[i+1], spec=planks))
    
    return nodes, members

    
def assemble_frame(nodes: List[NodeLocation], members: List[MemberLocation]) -> Tuple[FEModel3D, Dict[str, MemberSpec]]:
    frame = FEModel3D()
    for mat_name, props in MATERIAL_STRENGTHS.items():
        frame.add_material(
            mat_name,
            E=props['E'],
            G=props['E'] / (2 * (1 + props['nu'])),
            nu=props['nu'],
            rho=props['rho']
        )
    
    for node in nodes:
        frame.add_node(node.name, node.x, node.y, node.z)
    
    for member in members:
        member.spec.create_section(frame)
        frame.add_member(
            member.name,
            member.node_i,
            member.node_j,
            member.spec.material,
            member.spec.section_name
        )
    
    return frame


def define_supports(frame, nodes, wall_thickness, material, walls=False):
    supports = {}
    supports['north'] = [n for n in nodes if n.name.endswith('_N')].sort(key=lambda n: n.x)
    supports['south'] = [n for n in nodes if n.name.endswith('_S')].sort(key=lambda n: n.x)
    supports['east'] = [n for n in nodes if n.name.startswith('E_')].sort(key=lambda n: n.z)
    supports['west'] = [n for n in nodes if n.name.startswith('W_')].sort(key=lambda n: n.z)

    if walls:
        for support_side, support_nodes in supports.items():
            for node in support_nodes:
                frame.add_node(f'{node.name}_floor', node.x, 0, node.z)
            
            for i, _ in enumerate(support_nodes[:-1]):
                frame.add_quad(
                    f'{support_side}_wall{i}',
                    support_nodes[i].name,
                    support_nodes[i+1].name,
                    f'{support_nodes[i].name}_floor',
                    f'{support_nodes[i+1].name}_floor',
                    wall_thickness, 
                    material
                )
        for node_name in frame.nodes:
            if node_name.endswith('_floor'):
                frame.def_support(node_name, True, True, True, True, True, True)
    else:
        for support_side, support_nodes in supports.items():
            if support_side in ['north', 'south']:
                for node in support_nodes:
                    frame.def_support(node_name, True, True, True, True, True, True)

def create_model(
    east_joists: MemberSpec,
    west_joists: MemberSpec,
    tail_joists: MemberSpec,
    trimmers: MemberSpec,
    header: MemberSpec,
    planks: MemberSpec,
    params: DesignParameters = DEFAULT_PARAMS) -> Tuple:
    
    nodes, members = calculate_nodes_and_members(east_joists, west_joists, tail_joists, trimmers, header, planks, params)

    frame = assemble_frame(nodes, members)
    
    define_supports(frame, nodes, params.wall_thickness, 'brick', wall=True)
    
    for member in members:
        geom = member.spec.get_geometry()
        material = MATERIAL_STRENGTHS[member.spec.material]
        dead_load = -geom.A * material['rho']
        frame.add_member_dist_load(member.name, 'FY', dead_load, dead_load)
    
    return frame


def render(frame, deformed_scale=100, opacity=0.25):
    def set_wall_opacity(plotter, opacity=0.25):
        for actor in plotter.renderer.actors.values():
            if (hasattr(actor, 'mapper') and
                hasattr(actor.mapper, 'dataset') and
                actor.mapper.dataset.n_faces_strict > 0):
                actor.prop.opacity = opacity

    rndr = Renderer(frame)
    rndr.annotation_size = 5
    rndr.render_loads = False
    rndr.deformed_shape = True
    rndr.deformed_scale = deformed_scale
    rndr.post_update_callbacks.append(lambda plotter: set_wall_opacity(plotter, opacity=opacity))
    rndr.render_model()


# Example usage
if __name__ == '__main__':
    frame = create_model(
        east_joists=MemberSpec('W60x120', 'joist', quantity=2, padding=0),
        west_joists=MemberSpec('W60x120', 'joist', quantity=2, padding=0),
        tail_joists=MemberSpec('W60x120', 'tail', quantity=1, padding=0),
        trimmers=MemberSpec('W80x160', 'trimmer', quantity=2),
        header=MemberSpec('W60x120', 'header', quantity=1),
        planks=MemberSpec('W60x120', 'plank', quantity=8)
    )
    frame.analyze(check_statics=True)
    render(frame)





# @dataclass
# class BeamPlacement:
#     spec: MemberSpec
#     x_center: float
#     z_start: float = 0
#     z_end: float = None

#     def add_to_frame(self, frame: FEModel3D, params: DesignParameters, default_z_end: float):
#         z_end = self.z_end if self.z_end is not None else default_z_end
#         name = self.spec.name

#         frame.add_node(Naming.node(name, 'S', floor=True), self.x_center, 0, self.z_start)
#         frame.add_node(Naming.node(name, 'S'), self.x_center, params.floor_y, self.z_start)
        
#         if self.spec.beam_type != 'tail':
#             frame.add_node(Naming.node(name, 'N', floor=True), self.x_center, 0, z_end)
#             frame.add_node(Naming.node(name, 'N'), self.x_center, params.floor_y, z_end)
#             self.spec.create_section(frame)
#             frame.add_member(Naming.member(name), Naming.node(name, 'N'), Naming.node(name, 'S'), self.spec.material, self.spec.section_name)
#         else:
#             self.spec.create_section(frame)


# class GeometryResolver:
#     def __init__(
#         self,
#         east_joists: MemberSpec,
#         west_joists: MemberSpec,
#         tail_joists: MemberSpec,
#         trimmers: MemberSpec,
#         header: MemberSpec,
#         params: DesignParameters
#     ):
#         self.east_joists = east_joists
#         self.west_joists = west_joists
#         self.tail_joists = tail_joists
#         self.trimmers = trimmers
#         self.header = header
#         self.params = params
#         self.opening_x_start = params.opening_x_start

#         assert self.trimmers.quantity == 2, "Must have exactly 2 trimmers"
#         assert self.header.quantity == 1, "Must have exactly 1 header"
        
#         # Check padding doesn't exceed available space
#         max_east_padding = self.opening_x_start - self.trimmers.base - self.east_joists.base
#         assert self.east_joists.padding <= max_east_padding, \
#             f"East padding {self.east_joists.padding} exceeds max {max_east_padding}"
        
#         opening_x_end = self.opening_x_start + self.params.opening_length
#         max_west_padding = self.params.room_length - opening_x_end - self.trimmers.base - self.west_joists.base
#         assert self.west_joists.padding <= max_west_padding, \
#             f"West padding {self.west_joists.padding} exceeds max {max_west_padding}"

#         self.opening_x_end = self.opening_x_start + self.params.opening_length
        
#         self.trimmer_E_x_center = self.opening_x_start - (self.trimmers.base / 2)
#         self.trimmer_W_x_center = self.opening_x_end + (self.trimmers.base / 2)
        
#         self.tail_z_end = self.params.beam_length - self.params.opening_width - self.params.wall_beam_contact_depth/2 - self.header.base/2
#         self.header_z_pos = self.tail_z_end
        
#         self.opening_z_start = self.params.opening_width + self.params.wall_beam_contact_depth/2

#     def _resolve_beam_positions(self, n: int, clear_start: float, clear_end: float, beam_base: float, group: str) -> List[float]:
#         def east_positions(n, clear_start, clear_end, beam_base):
#             centerline_start = clear_start + beam_base / 2
#             centerline_end = clear_end + beam_base / 2
#             positions = np.linspace(centerline_start, centerline_end, n+1).tolist()
#             return positions[:-1]

#         def west_positions(n, clear_start, clear_end, beam_base):
#             centerline_start = clear_start - beam_base / 2
#             centerline_end = clear_end + beam_base / 2
#             positions = np.linspace(centerline_end, centerline_start, n+1).tolist()
#             return positions[:-1]

#         def tail_positions(n, clear_start, clear_end, beam_base):
#             centerline_start = clear_start - beam_base / 2
#             centerline_end = clear_end + beam_base / 2
#             positions = np.linspace(centerline_start, centerline_end, n+2).tolist()
#             return positions[1:-1]

#         dispatch = {
#             'east': east_positions,
#             'west': west_positions,
#             'tail': tail_positions
#         }

#         if group not in dispatch:
#             raise ValueError("Invalid group specified")

#         return dispatch[group](n, clear_start, clear_end, beam_base)

#     def resolve_all_placements(self) -> List[BeamPlacement]:
#         all_placements = []
        
#         # East joists
#         if self.east_joists.quantity > 0:
#             joist_spec_east = MemberSpec(
#                 self.east_joists.catalog_id, 
#                 'joist',
#                 quantity=self.east_joists.quantity,
#                 padding=self.east_joists.padding
#             )
#             clear_start = self.east_joists.padding
#             clear_end = self.trimmer_E_x_center - self.trimmers.base / 2
#             east_positions = self._resolve_beam_positions(
#                 self.east_joists.quantity, 
#                 clear_start, 
#                 clear_end, 
#                 joist_spec_east.base,
#                 "east"
#             )
#             for i, x in enumerate(east_positions):
#                 all_placements.append(BeamPlacement(spec=joist_spec_east.copy(name=f'E{i}'), x_center=x))
        
#         # Tail joists
#         if self.tail_joists.quantity > 0:
#             tail_spec = MemberSpec(
#                 self.tail_joists.catalog_id, 
#                 'tail',
#                 quantity=self.tail_joists.quantity,
#                 padding=self.tail_joists.padding
#             )
#             clear_start = self.opening_x_start + self.tail_joists.padding
#             clear_end = self.opening_x_end - self.tail_joists.padding
#             tail_positions = self._resolve_beam_positions(
#                 self.tail_joists.quantity, 
#                 clear_start, 
#                 clear_end, 
#                 tail_spec.base,
#                 "tail"
#             )
#             for i, x in enumerate(tail_positions):
#                 all_placements.append(BeamPlacement(spec=tail_spec.copy(name=f'T{i}'), x_center=x, z_end=self.tail_z_end))
        
#         # West joists
#         if self.west_joists.quantity > 0:
#             joist_spec_west = MemberSpec(
#                 self.west_joists.catalog_id, 
#                 'joist',
#                 quantity=self.west_joists.quantity,
#                 padding=self.west_joists.padding
#             )
#             clear_start = self.trimmer_W_x_center + self.trimmers.base / 2
#             clear_end = self.params.room_length - self.west_joists.padding - self.west_joists.base
#             west_positions = self._resolve_beam_positions(
#                 self.west_joists.quantity, 
#                 clear_start,
#                 clear_end,
#                 joist_spec_west.base,
#                 "west"
#             )
#             for i, x in enumerate(west_positions):
#                 all_placements.append(BeamPlacement(spec=joist_spec_west.copy(name=f'W{i}'), x_center=x))

#         # Trimmers
#         trimmer_spec = MemberSpec(
#             self.trimmers.catalog_id, 
#             'trimmer',
#             quantity=self.trimmers.quantity,
#             padding=self.trimmers.padding
#         )
#         all_placements.append(BeamPlacement(spec=trimmer_spec.copy(name='trimmer_E'), x_center=self.trimmer_E_x_center))
#         all_placements.append(BeamPlacement(spec=trimmer_spec.copy(name='trimmer_W'), x_center=self.trimmer_W_x_center))

#         return all_placements

#     def add_header_to_frame(self, frame: FEModel3D):
#         header_spec = self.header
#         header_spec.create_section(frame)
#         frame.add_node(Naming.header_node('E'), self.trimmer_E_x_center, 
#                        self.params.floor_y, self.header_z_pos)
#         frame.add_node(Naming.header_node('W'), self.trimmer_W_x_center, 
#                        self.params.floor_y, self.header_z_pos)
#         frame.add_member(Naming.header_member(), Naming.header_node('W'), 
#                          Naming.header_node('E'), header_spec.material, 
#                          header_spec.section_name)

#     def connect_tails_to_header(self, frame: FEModel3D, layout: 'LayoutManager'):
#         for beam_placement in layout.beams:
#             if beam_placement.spec.beam_type == 'tail':
#                 tail_node_name = Naming.tail_header_node(beam_placement.spec.name)
#                 frame.add_node(tail_node_name, beam_placement.x_center, self.params.floor_y, self.header_z_pos)
#                 frame.add_member(
#                     Naming.member(beam_placement.spec.name),
#                     tail_node_name,
#                     Naming.node(beam_placement.spec.name, 'S'),
#                     beam_placement.spec.material,
#                     beam_placement.spec.section_name
#                 )


# class LayoutManager:
#     def __init__(self, params: DesignParameters):
#         self.params = params
#         self.beams: List[BeamPlacement] = []

#     def add_beams(self, placements: List[BeamPlacement]):
#         self.beams.extend(placements)

#     def get_boundary_beams(self) -> Tuple[BeamPlacement, BeamPlacement]:
#         beams_with_position = sorted(self.beams, key=lambda p: p.x_center)
#         return beams_with_position[0], beams_with_position[-1]


# class LoadApplicator:
#     def __init__(self, beams: List[BeamPlacement], params: DesignParameters):
#         self.beams = beams
#         self.params = params

#     def apply_dead_loads(self, frame: FEModel3D):
#         for p in self.beams:
#             section_geom = p.spec.get_geometry()
#             material = MATERIAL_STRENGTHS[p.spec.material]
#             dead_load = -section_geom.A * material['rho']
#             frame.add_member_dist_load(p.spec.name, 'FY', dead_load, dead_load)
    
#     def apply_live_loads(self, frame: FEModel3D):
#         sorted_beams = sorted(self.beams, key=lambda p: p.x_center)
#         if len(sorted_beams) < 2:
#             return

#         for i, p in enumerate(sorted_beams):
#             # Calculate tributary boundaries
#             left_boundary = sorted_beams[i-1].x_center if i > 0 else 0
#             right_boundary = sorted_beams[i+1].x_center if i < len(sorted_beams) - 1 else self.params.room_length
            
#             trib_left = (p.x_center - left_boundary) / 2
#             trib_right = (right_boundary - p.x_center) / 2
            
#             load_left = self.params.live_load_mpa * trib_left
#             load_right = self.params.live_load_mpa * trib_right

#             if p.spec.beam_type in ['joist', 'tail']:
#                 total_load = load_left + load_right
#                 frame.add_member_dist_load(p.spec.name, 'FY', total_load, total_load)
#             elif p.spec.beam_type == 'trimmer':
#                 # For trimmers, one side has a load break at the opening
#                 is_left_opening = i > 0 and sorted_beams[i-1].spec.beam_type == 'tail'
#                 is_right_opening = i < len(sorted_beams)-1 and sorted_beams[i+1].spec.beam_type == 'tail'
                
#                 if is_left_opening:
#                     frame.add_member_dist_load(p.spec.name, 'FY', load_left, load_left, 0, self.params.opening_width + self.params.wall_beam_contact_depth/2)
#                     frame.add_member_dist_load(p.spec.name, 'FY', load_right, load_right)
#                 elif is_right_opening:
#                     frame.add_member_dist_load(p.spec.name, 'FY', load_left, load_left)
#                     frame.add_member_dist_load(p.spec.name, 'FY', load_right, load_right, 0, self.params.opening_width + self.params.wall_beam_contact_depth/2)
#                 else:
#                     frame.add_member_dist_load(p.spec.name, 'FY', load_left + load_right, load_left + load_right)


def auto_add_walls(frame, layout, params, wall_thickness, material):
    frame.add_node(Naming.corner_node('SW', floor=True), 0, 0, 0)
    frame.add_node(Naming.corner_node('SW', floor=False), 0, params.floor_y, 0)
    frame.add_node(Naming.corner_node('NW', floor=True), 0, 0, params.beam_length)
    frame.add_node(Naming.corner_node('NW', floor=False), 0, params.floor_y, params.beam_length)
    frame.add_node(Naming.corner_node('SE', floor=True), params.room_length, 0, 0)
    frame.add_node(Naming.corner_node('SE', floor=False), params.room_length, params.floor_y, 0)
    frame.add_node(Naming.corner_node('NE', floor=True), params.room_length, 0, params.beam_length)
    frame.add_node(Naming.corner_node('NE', floor=False), params.room_length, params.floor_y, params.beam_length)

    frame.add_quad(
        Naming.side_wall('west'),
        Naming.corner_node('SW', floor=True),
        Naming.corner_node('NW', floor=True),
        Naming.corner_node('NW', floor=False),
        Naming.corner_node('SW', floor=False),
        wall_thickness, material
    )
    
    frame.add_quad(
        Naming.side_wall('east'),
        Naming.corner_node('SE', floor=True),
        Naming.corner_node('NE', floor=True),
        Naming.corner_node('NE', floor=False),
        Naming.corner_node('SE', floor=False),
        wall_thickness, material
    )
    
    sorted_beams = sorted(layout.beams, key=lambda p: p.x_center)
    if sorted_beams:
        first_beam = sorted_beams[0]
        frame.add_quad(
            Naming.wall('corner_W', first_beam.spec.name, 'south'),
            Naming.corner_node('SW', floor=True),
            Naming.node(first_beam.spec.name, 'S', floor=True),
            Naming.node(first_beam.spec.name, 'S'),
            Naming.corner_node('SW', floor=False),
            wall_thickness, material
        )
        
        for i in range(len(sorted_beams) - 1):
            prev_beam = sorted_beams[i]
            beam = sorted_beams[i+1]
            frame.add_quad(
                Naming.wall(prev_beam.spec.name, beam.spec.name, 'south'),
                Naming.node(prev_beam.spec.name, 'S', floor=True),
                Naming.node(beam.spec.name, 'S', floor=True),
                Naming.node(beam.spec.name, 'S'),
                Naming.node(prev_beam.spec.name, 'S'),
                wall_thickness, material
            )

        last_beam = sorted_beams[-1]
        frame.add_quad(
            Naming.wall(last_beam.spec.name, 'corner_E', 'south'),
            Naming.node(last_beam.spec.name, 'S', floor=True),
            Naming.corner_node('SE', floor=True),
            Naming.corner_node('SE', floor=False),
            Naming.node(last_beam.spec.name, 'S'),
            wall_thickness, material
        )
    
    north_reaching_beams = [b for b in layout.beams if b.spec.beam_type not in ['tail', 'header']]
    north_reaching_beams.sort(key=lambda p: p.x_center)
    if north_reaching_beams:
        first_beam = north_reaching_beams[0]
        frame.add_quad(
            Naming.wall('corner_W', first_beam.spec.name, 'north'),
            Naming.corner_node('NW', floor=True),
            Naming.node(first_beam.spec.name, 'N', floor=True),
            Naming.node(first_beam.spec.name, 'N'),
            Naming.corner_node('NW', floor=False),
            wall_thickness, material
        )

        for i in range(len(north_reaching_beams) - 1):
            prev_beam = north_reaching_beams[i]
            beam = north_reaching_beams[i+1]
            frame.add_quad(
                Naming.wall(prev_beam.spec.name, beam.spec.name, 'north'),
                Naming.node(prev_beam.spec.name, 'N', floor=True),
                Naming.node(beam.spec.name, 'N', floor=True),
                Naming.node(beam.spec.name, 'N'),
                Naming.node(prev_beam.spec.name, 'N'),
                wall_thickness, material
            )

        last_beam = north_reaching_beams[-1]
        frame.add_quad(
            Naming.wall(last_beam.spec.name, 'corner_E', 'north'),
            Naming.node(last_beam.spec.name, 'N', floor=True),
            Naming.corner_node('NE', floor=True),
            Naming.corner_node('NE', floor=False),
            Naming.node(last_beam.spec.name, 'N'),
            wall_thickness, material
        )


# @dataclass
# class BeamTelemetry:
#     name: str
#     material: str
#     catalog_id: str
#     length: float
#     max_moment_mz: float
#     min_moment_mz: float
#     max_shear_fy: float
#     min_shear_fy: float
#     max_deflection: float
#     min_deflection: float
#     volume: float
#     cost: float
#     stress_max: float
#     shear_stress_max: float
#     passes_bending: bool
#     passes_shear: bool

# @dataclass
# class GroupTelemetry:
#     group_name: str
#     count: int
#     max_deflection: float
#     mean_deflection: float
#     max_moment: float
#     mean_moment: float
#     max_shear: float
#     mean_shear: float
#     total_volume: float
#     total_cost: float
#     all_pass_bending: bool
#     all_pass_shear: bool
#     beam_details: List[BeamTelemetry]

# @dataclass
# class SystemTelemetry:
#     max_header_deflection_mm: float
#     max_deflection_overall_mm: float
#     worst_member: str
#     total_volume_m3: float
#     total_cost: float
#     system_passes: bool

# @dataclass
# class Telemetry:
#     beam_telemetries: Dict[str, BeamTelemetry] = field(default_factory=dict)
#     group_telemetries: Dict[str, GroupTelemetry] = field(default_factory=dict)
#     system_telemetry: SystemTelemetry = None


# class TelemetryCollector:
#     """Collects and analyzes telemetry from FE analysis"""
    
#     def __init__(self, frame, beam_specs: Dict[str, MemberSpec]):
#         self.frame = frame
#         self.beam_specs = beam_specs
        
#     def calculate_beam_stress(self, member_name: str, moment: float, spec: MemberSpec) -> float:
#         """Calculate bending stress: sigma = M * c / I"""
#         geometry = spec.get_geometry()
#         c = spec.height / 2
#         stress = abs(moment * c / geometry.Iz)
#         return stress
    
#     def calculate_shear_stress(self, member_name: str, shear: float, spec: MemberSpec) -> float:
#         """Calculate average shear stress"""
#         geometry = spec.get_geometry()
#         if spec.shape == 'rectangular':
#             tau = 1.5 * abs(shear) / geometry.A
#         else:  # I-beam
#             catalog_data = spec._catalog_data
#             web_area = (spec.height - 2*catalog_data['flange_thickness']) * catalog_data['web_thickness']
#             tau = abs(shear) / web_area
#         return tau
    
#     def collect_beam_telemetry(self, member_name: str, spec: MemberSpec) -> BeamTelemetry:
#         member = self.frame.members[member_name]
        
#         # Get forces and deflections
#         max_moment = member.max_moment('Mz', 'Combo 1')
#         min_moment = member.min_moment('Mz', 'Combo 1')
#         max_shear = member.max_shear('Fy', 'Combo 1')
#         min_shear = member.min_shear('Fy', 'Combo 1')
#         max_deflection = member.max_deflection('dy', 'Combo 1')
#         min_deflection = member.min_deflection('dy', 'Combo 1')
        
#         # Calculate stresses
#         moment_for_stress = max(abs(max_moment), abs(min_moment))
#         stress_max = self.calculate_beam_stress(member_name, moment_for_stress, spec)
        
#         shear_for_stress = max(abs(max_shear), abs(min_shear))
#         shear_stress_max = self.calculate_shear_stress(member_name, shear_for_stress, spec)
        
#         # Check against material strengths
#         material_props = MATERIAL_STRENGTHS[spec.material]
#         passes_bending = stress_max <= material_props['f_mk']
#         passes_shear = shear_stress_max <= material_props['f_vk']
        
#         # Calculate cost and volume
#         length = member.L()
#         volume = spec.get_volume(length)
#         cost = spec.get_cost(length)
        
#         return BeamTelemetry(
#             name=member_name,
#             material=spec.material,
#             catalog_id=spec.catalog_id,
#             length=length,
#             max_moment_mz=max_moment,
#             min_moment_mz=min_moment,
#             max_shear_fy=max_shear,
#             min_shear_fy=min_shear,
#             max_deflection=max_deflection,
#             min_deflection=min_deflection,
#             volume=volume,
#             cost=cost,
#             stress_max=stress_max,
#             shear_stress_max=shear_stress_max,
#             passes_bending=passes_bending,
#             passes_shear=passes_shear
#         )
    
#     def collect_group_telemetry(self, group_name: str, member_names: List[str]) -> GroupTelemetry:
#         beam_telemetries = [self.collect_beam_telemetry(name, self.beam_specs[name]) for name in member_names]
        
#         if not beam_telemetries:
#             return GroupTelemetry(
#                 group_name=group_name, count=0, max_deflection=0, mean_deflection=0,
#                 max_moment=0, mean_moment=0, max_shear=0, mean_shear=0,
#                 total_volume=0, total_cost=0, all_pass_bending=True, all_pass_shear=True,
#                 beam_details=[]
#             )
        
#         deflections = [abs(b.min_deflection) for b in beam_telemetries]
#         moments = [max(abs(b.max_moment_mz), abs(b.min_moment_mz)) for b in beam_telemetries]
#         shears = [max(abs(b.max_shear_fy), abs(b.min_shear_fy)) for b in beam_telemetries]
        
#         return GroupTelemetry(
#             group_name=group_name,
#             count=len(beam_telemetries),
#             max_deflection=max(deflections),
#             mean_deflection=np.mean(deflections),
#             max_moment=max(moments),
#             mean_moment=np.mean(moments),
#             max_shear=max(shears),
#             mean_shear=np.mean(shears),
#             total_volume=sum(b.volume for b in beam_telemetries),
#             total_cost=sum(b.cost for b in beam_telemetries),
#             all_pass_bending=all(b.passes_bending for b in beam_telemetries),
#             all_pass_shear=all(b.passes_shear for b in beam_telemetries),
#             beam_details=beam_telemetries
#         )
    
#     def collect_system_telemetry(self, beam_groups: Dict[str, List[str]]) -> Telemetry:
#         """Collect telemetry for entire system"""
#         telemetry = Telemetry()
        
#         for group_name, member_names in beam_groups.items():
#             if not member_names:
#                 continue
#             group_telem = self.collect_group_telemetry(group_name, member_names)
#             telemetry.group_telemetries[group_name] = group_telem
#             for beam_telem in group_telem.beam_details:
#                 telemetry.beam_telemetries[beam_telem.name] = beam_telem
        
#         # Overall metrics
#         all_beam_stats = list(telemetry.beam_telemetries.values())
#         max_deflection_overall = max((abs(b.min_deflection) for b in all_beam_stats), default=0)
#         worst_member = max(all_beam_stats, key=lambda b: abs(b.min_deflection), default=None)
        
#         total_volume = sum(g.total_volume for g in telemetry.group_telemetries.values())
#         total_cost = sum(g.total_cost for g in telemetry.group_telemetries.values())
        
#         system_passes = all(g.all_pass_bending and g.all_pass_shear for g in telemetry.group_telemetries.values())
        
#         header_deflection = abs(self.frame.members[Naming.header_member()].min_deflection('dy', 'Combo 1'))
        
#         telemetry.system_telemetry = SystemTelemetry(
#             max_header_deflection_mm=header_deflection,
#             max_deflection_overall_mm=max_deflection_overall,
#             worst_member=worst_member.name if worst_member else '',
#             total_volume_m3=total_volume / 1e9,
#             total_cost=total_cost,
#             system_passes=system_passes,
#         )
#         return telemetry


# def generate_and_analyze_floor(
#     east_joists: MemberSpec,
#     west_joists: MemberSpec,
#     tail_joists: MemberSpec,
#     trimmers: MemberSpec,
#     header: MemberSpec,
# ) -> tuple:
#     """Generate floor model, run analysis, and return frame with results"""
#     resolver = GeometryResolver(
#         east_joists=east_joists,
#         west_joists=west_joists,
#         tail_joists=tail_joists,
#         trimmers=trimmers,
#         header=header,
#         params=DEFAULT_PARAMS
#     )
#     all_placements = resolver.resolve_all_placements()
#     frame = FEModel3D()
    
#     for mat_name, props in MATERIAL_STRENGTHS.items():
#         frame.add_material(
#             mat_name, 
#             E=props['E'], 
#             G=props['E'] / (2 * (1 + props['nu'])), 
#             nu=props['nu'], 
#             rho=props['rho']
#         )
    
#     beam_spec_map = {}
#     for beam_placement in all_placements:
#         beam_spec_map[beam_placement.spec.name] = beam_placement.spec
    
#     layout = LayoutManager(params=DEFAULT_PARAMS)
#     layout.add_beams(all_placements)
#     for beam_placement in layout.beams:
#         beam_placement.add_to_frame(frame, DEFAULT_PARAMS, DEFAULT_PARAMS.beam_length)

#     resolver.add_header_to_frame(frame)
#     resolver.connect_tails_to_header(frame, layout)
#     beam_spec_map['header'] = resolver.header


#     # TEST: Add a single plank member along the centerline
#     plank_spec = MemberSpec(catalog_id='W60x120', beam_type='plank', quantity=1)
#     plank_spec.create_section(frame)
#     plank_node_start = 'plank_test_start'
#     plank_node_end = 'plank_test_end'
#     frame.add_node(plank_node_start, plank_spec.base/2, DEFAULT_PARAMS.floor_y, DEFAULT_PARAMS.beam_length/2)
#     frame.add_node(plank_node_end, DEFAULT_PARAMS.room_length - plank_spec.base/2, DEFAULT_PARAMS.floor_y, DEFAULT_PARAMS.beam_length/2)
#     frame.add_member('plank_test', plank_node_end, plank_node_start, plank_spec.material, plank_spec.section_name)
#     beam_spec_map['plank_test'] = plank_spec

#     plank_geom = plank_spec.get_geometry()
#     plank_material = MATERIAL_STRENGTHS[plank_spec.material]
#     plank_dead_load = -plank_geom.A * plank_material['rho']
#     frame.add_member_dist_load('plank_test', 'FY', plank_dead_load, plank_dead_load)


#     auto_add_walls(frame, layout, DEFAULT_PARAMS, wall_thickness=DEFAULT_PARAMS.wall_thickness, material='brick')
#     for node_name in frame.nodes:
#         if node_name.startswith('floor') or 'floor' in node_name:
#             frame.def_support(node_name, True, True, True, True, True, True)
    
#     header_geom = resolver.header.get_geometry()
#     header_material = MATERIAL_STRENGTHS[resolver.header.material]
#     header_dead_load = -header_geom.A * header_material['rho']
#     frame.add_member_dist_load(Naming.header_member(), 'FY', header_dead_load, header_dead_load)

#     load_applicator = LoadApplicator(layout.beams, layout.params)
#     load_applicator.apply_dead_loads(frame)
#     # load_applicator.apply_live_loads(frame) # Old live loads on beams
    
#     try:
#         frame.analyze(check_statics=True)
#     except Exception as e:
#         raise RuntimeError(f"Analysis failed: {e}")
    
#     return frame, resolver, beam_spec_map


# class FloorOptimizer:
#     """Handles optimization and telemetry collection"""
    
#     def __init__(self):
#         self.results = []
        
#     def run_single_configuration(
#         self,
#         east_joists: MemberSpec,
#         west_joists: MemberSpec,
#         tail_joists: MemberSpec,
#         trimmers: MemberSpec,
#         header: MemberSpec,
#         config_id: int = None
#     ) -> Tuple[Optional[FEModel3D], Dict]:
#         """Run analysis for a single configuration and collect telemetry"""
#         frame = None
#         try:
#             frame, resolver, beam_spec_map = generate_and_analyze_floor(
#                 east_joists=east_joists,
#                 west_joists=west_joists,
#                 tail_joists=tail_joists,
#                 trimmers=trimmers,
#                 header=header,
#             )
#             collector = TelemetryCollector(frame, beam_spec_map)
            
#             beam_groups = {
#                 'east_joists': [name for name, spec in beam_spec_map.items() if spec.beam_type == 'joist' and name.startswith('E')],
#                 'west_joists': [name for name, spec in beam_spec_map.items() if spec.beam_type == 'joist' and name.startswith('W')],
#                 'tail_joists': [name for name, spec in beam_spec_map.items() if spec.beam_type == 'tail'],
#                 'trimmers': [name for name, spec in beam_spec_map.items() if spec.beam_type == 'trimmer'],
#                 'header': [name for name, spec in beam_spec_map.items() if spec.beam_type == 'header']
#             }
#             telemetry = collector.collect_system_telemetry(beam_groups)
            
#             return frame, {
#                 'config_id': config_id,
#                 'east_joists': east_joists,
#                 'west_joists': west_joists,
#                 'tail_joists': tail_joists,
#                 'trimmers': trimmers,
#                 'header': header,
#                 'telemetry': telemetry,
#                 'success': True,
#                 'error': None
#             }
            
#         except Exception as e:
#             return None, {
#                 'config_id': config_id,
#                 'east_joists': east_joists,
#                 'west_joists': west_joists,
#                 'tail_joists': tail_joists,
#                 'trimmers': trimmers,
#                 'header': header,
#                 'telemetry': None,
#                 'success': False,
#                 'error': str(e)
#             }
    
#     def run_grid_search(self, param_space: Dict[str, List]) -> pd.DataFrame:
#         """
#         Run grid search over parameter space
        
#         Args:
#             param_space: Dictionary defining parameter ranges, e.g.:
#                 {
#                     'east_joist_catalog_id': ['W60x120', 'W80x160'],
#                     'east_quantity': [2, 3, 4],
#                     'trimmer_catalog_id': ['W80x160', 'W100x200'],
#                     'opening_x_start': [800, 820, 840],
#                     ...
#                 }
#         """
#         keys = list(param_space.keys())
#         combinations = list(product(*[param_space[k] for k in keys]))
        
#         print(f"Running grid search over {len(combinations)} configurations...")
        
#         for i, combo in tqdm(enumerate(combinations), total=len(combinations)):
#             config_params = dict(zip(keys, combo))
            
#             # Build configuration
#             east_joists_spec = MemberSpec(
#                 catalog_id=config_params['east_joist_catalog_id'],
#                 beam_type='joist',
#                 quantity=config_params['east_quantity'],
#                 padding=config_params.get('east_padding', 0)
#             )
#             west_joists_spec = MemberSpec(
#                 catalog_id=config_params['west_joist_catalog_id'],
#                 beam_type='joist',
#                 quantity=config_params['west_quantity'],
#                 padding=config_params.get('west_padding', 0)
#             )
#             tail_joists_spec = MemberSpec(
#                 catalog_id=config_params['tail_joist_catalog_id'],
#                 beam_type='tail',
#                 quantity=config_params['tail_quantity'],
#                 padding=config_params.get('tail_padding', 0)
#             )
#             trimmers_spec = MemberSpec(
#                 catalog_id=config_params['trimmer_catalog_id'],
#                 beam_type='trimmer',
#                 quantity=2
#             )
#             header_spec = MemberSpec(
#                 catalog_id=config_params['header_catalog_id'],
#                 beam_type='header',
#                 quantity=1
#             )
            
#             _, result = self.run_single_configuration(
#                 east_joists=east_joists_spec,
#                 west_joists=west_joists_spec,
#                 tail_joists=tail_joists_spec,
#                 trimmers=trimmers_spec,
#                 header=header_spec,
#                 config_id=i
#             )
#             self.results.append(result)
            
#             if (i + 1) % 10 == 0:
#                 print(f"Completed {i + 1}/{len(combinations)} configurations")
        
#         return self.results_to_dataframe()
    
#     def results_to_dataframe(self) -> pd.DataFrame:
#         """Convert results to pandas DataFrame for analysis"""
#         rows = []
        
#         for result in self.results:
#             if not result['success']:
#                 row = {
#                     'config_id': result['config_id'],
#                     'success': False,
#                     'error': result['error']
#                 }
#                 rows.append(row)
#                 continue
            
#             telem = result['telemetry']
            
#             row = {
#                 # Configuration
#                 'config_id': result['config_id'],
#                 'east_catalog_id': result['east_joists'].catalog_id,
#                 'east_quantity': result['east_joists'].quantity,
#                 'west_catalog_id': result['west_joists'].catalog_id,
#                 'west_quantity': result['west_joists'].quantity,
#                 'tail_catalog_id': result['tail_joists'].catalog_id,
#                 'tail_quantity': result['tail_joists'].quantity,
#                 'trimmer_catalog_id': result['trimmers'].catalog_id,
#                 'header_catalog_id': result['header'].catalog_id,
#                 'opening_x_start': DEFAULT_PARAMS.opening_x_start,
                
#                 # Overall metrics
#                 'max_deflection_mm': telem.system_telemetry.max_deflection_overall_mm,
#                 'header_deflection_mm': telem.system_telemetry.max_header_deflection_mm,
#                 'worst_member': telem.system_telemetry.worst_member,
#                 'total_volume_m3': telem.system_telemetry.total_volume_m3,
#                 'total_cost': telem.system_telemetry.total_cost,
#                 'system_passes': telem.system_telemetry.system_passes,
                
#                 # Group-level metrics
#                 **{f'{group}_max_deflection': stats.max_deflection 
#                    for group, stats in telem.group_telemetries.items()},
#                 **{f'{group}_mean_deflection': stats.mean_deflection 
#                    for group, stats in telem.group_telemetries.items()},
#                 **{f'{group}_max_moment': stats.max_moment 
#                    for group, stats in telem.group_telemetries.items()},
#                 **{f'{group}_passes_bending': stats.all_pass_bending 
#                    for group, stats in telem.group_telemetries.items()},
#                 **{f'{group}_passes_shear': stats.all_pass_shear 
#                    for group, stats in telem.group_telemetries.items()},
#                 **{f'{group}_total_cost': stats.total_cost 
#                    for group, stats in telem.group_telemetries.items()},
                
#                 'success': True,
#                 'error': None
#             }
            
#             rows.append(row)
        
#         return pd.DataFrame(rows)
    
#     def get_pareto_front(self, df: pd.DataFrame, objectives: List[str] = ['total_cost', 'max_deflection_mm']) -> pd.DataFrame:
#         """Find Pareto-optimal solutions"""
#         valid = df[(df['success'] == True) & (df['system_passes'] == True)].copy()
        
#         if len(valid) == 0:
#             print("No valid configurations found!")
#             return pd.DataFrame()
        
#         # Find Pareto front
#         is_pareto = np.ones(len(valid), dtype=bool)
        
#         for i, row_i in valid.iterrows():
#             for j, row_j in valid.iterrows():
#                 if i == j:
#                     continue
                
#                 # Check if j dominates i
#                 dominates = True
#                 for obj in objectives:
#                     if row_j[obj] > row_i[obj]:
#                         dominates = False
#                         break
                
#                 if dominates:
#                     strictly_better = any(row_j[obj] < row_i[obj] for obj in objectives)
#                     if strictly_better:
#                         is_pareto[valid.index.get_loc(i)] = False
#                         break
        
#         pareto_indices = valid.index[is_pareto]
#         return valid.loc[pareto_indices]


# if __name__ == '__main__':
    # Single solution with rendering
    # optimizer = FloorOptimizer()
    # frame, result = optimizer.run_single_configuration(
    #     east_joists=MemberSpec(catalog_id='W60x120', beam_type='joist', quantity=1, padding=0),
    #     west_joists=MemberSpec(catalog_id='W60x120', beam_type='joist', quantity=1, padding=0),
    #     tail_joists=MemberSpec(catalog_id='W60x120', beam_type='tail', quantity=1, padding=0),
    #     trimmers=MemberSpec(catalog_id='W80x160', beam_type='trimmer', quantity=2),
    #     header=MemberSpec(catalog_id='W60x120', beam_type='header', quantity=1),
    #     config_id=0
    # )

    # if result['success']:
    #     telem = result['telemetry']
    #     print(f"\nSystem passes all checks: {telem.system_telemetry.system_passes}")
    #     print(f"Total cost: ${telem.system_telemetry.total_cost:.2f}")
    #     print(f"Total volume: {telem.system_telemetry.total_volume_m3:.4f} m³")
    #     print(f"Max deflection: {telem.system_telemetry.max_deflection_overall_mm:.3f} mm")
    #     print(f"Header deflection: {telem.system_telemetry.max_header_deflection_mm:.3f} mm")
    #     print(f"Worst member: {telem.system_telemetry.worst_member}")
        
    #     print("\nGroup Statistics:")
    #     for group_name, stats in telem.group_telemetries.items():
    #         print(f"\n  {group_name}:")
    #         print(f"    Count: {stats.count}")
    #         print(f"    Max deflection: {stats.max_deflection:.3f} mm")
    #         print(f"    Mean deflection: {stats.mean_deflection:.3f} mm")
    #         print(f"    Cost: ${stats.total_cost:.2f}")
    #         print(f"    Passes bending: {stats.all_pass_bending}")
    #         print(f"    Passes shear: {stats.all_pass_shear}")
    # else:
    #     print(f"Analysis failed: {result['error']}")


    # def set_wall_opacity(plotter, opacity=0.5):
    #     for actor in plotter.renderer.actors.values():
    #         if (hasattr(actor, 'mapper') and
    #             hasattr(actor.mapper, 'dataset') and
    #             actor.mapper.dataset.n_faces_strict > 0):
    #             actor.prop.opacity = opacity

    # rndr = Renderer(frame)
    # rndr.annotation_size = 5
    # rndr.render_loads = False
    # rndr.deformed_shape = True
    # rndr.deformed_scale = 1000
    # opacity = 0.25
    # rndr.post_update_callbacks.append(lambda plotter: set_wall_opacity(plotter, opacity=opacity))
    # rndr.render_model()