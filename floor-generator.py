from dataclasses import dataclass, replace
from re import L
from typing import List, Dict, Optional, Tuple, Literal
from Pynite import FEModel3D
from Pynite.Rendering import Renderer
import numpy as np
import pandas as pd


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
class CrossSectionProperties:
    A: float  # Area
    Iy: float  # Second moment about y-axis
    Iz: float  # Second moment about z-axis
    J: float   # Torsional constant
    
    @classmethod
    def from_rectangular(cls, base: float, height: float) -> 'CrossSectionProperties':
        A = base * height
        b, h = min(base, height), max(base, height)
        J = (b**3 * h) * (1/3 - 0.21 * (b/h) * (1 - (b**4)/(12*h**4)))
        Iy = (height * base**3) / 12
        Iz = (base * height**3) / 12
        return cls(A=A, Iy=Iy, Iz=Iz, J=J)
    
    @classmethod
    def from_i_beam(cls, height: float, flange_width: float, flange_thickness: float, web_thickness: float) -> 'CrossSectionProperties':
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
    
    def get_geometry(self) -> CrossSectionProperties:
        if self.shape == 'rectangular':
            return CrossSectionProperties.from_rectangular(self.base, self.height)
        elif self.shape == 'I-beam':
            return CrossSectionProperties.from_i_beam(self.height, self._catalog_data['flange_width'], self._catalog_data['flange_thickness'], self._catalog_data['web_thickness'])
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
class Member:
    name: str
    node_i: str
    node_j: str
    spec: MemberSpec
    

def calculate_nodes_and_members(
    east_joists: MemberSpec,
    west_joists: MemberSpec,
    tail_joists: MemberSpec,
    trimmers: MemberSpec,
    header: MemberSpec,
    planks: MemberSpec,
    params: DesignParameters) -> Tuple[List[NodeLocation], List[Member]]:

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
            positions = np.linspace(centerline_start, centerline_end, n + 1).tolist()
            return positions[1:]
        
        elif distribution == 'centered':
            centerline_start = clear_start + member_base / 2
            centerline_end = clear_end - member_base / 2
            positions = np.linspace(centerline_start, centerline_end, n + 2).tolist()
            return positions[1:-1]
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    assert trimmers.quantity == 2, "Must have exactly 2 trimmers"
    assert header.quantity == 1, "Must have exactly 1 header"
    
    # Stairs opening params
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
    clear_end = params.room_width - params.wall_beam_contact_depth
    planks.quantity = int((clear_end - clear_start) // planks.base)
    z_positions = _calculate_evenly_spaced_positions(planks.quantity, clear_start-planks.base, clear_end+planks.base, planks.base, 'centered')
    plank_positions = [(f'p{i}', z) for i, z in enumerate(z_positions)]
    
    # Beam nodes and member locations
    nodes = []
    members = []
    for group_name, group_positions in beam_positions.items():
        for beam_name, x in group_positions:
            nodes.append(NodeLocation(f'{beam_name}_S', x, params.floor_y, 0))
            if group_name == 'tail': # Tails connect to header (header_z), not to wall (beam_length)
                nodes.append(NodeLocation(f'{beam_name}_header', x, params.floor_y, header_z))
                members.append(Member(name=beam_name, node_i=f'{beam_name}_header', node_j=f'{beam_name}_S', spec=east_joists))
            else:
                nodes.append(NodeLocation(f'{beam_name}_N', x, params.floor_y, params.beam_length))
                members.append(Member(name=beam_name, node_i=f'{beam_name}_N', node_j=f'{beam_name}_S', spec=east_joists))
    
    nodes.append(NodeLocation('header_E', trimmer_E_x, params.floor_y, header_z))
    nodes.append(NodeLocation('header_W', trimmer_W_x, params.floor_y, header_z))
    members.append(Member(name='header', node_i='header_W', node_j='header_E', spec=header))

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
        plank_nodes.sort(key=lambda node: node.x)
        for i, _ in enumerate(plank_nodes[:-1]):
            members.append(Member(name=f'{plank_name}_{i}', node_i=plank_nodes[i].name, node_j=plank_nodes[i+1].name, spec=planks))
    
    return nodes, members

    
def assemble_frame(nodes: List[NodeLocation], members: List[Member]) -> Tuple[FEModel3D, Dict[str, MemberSpec]]:
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
    supports['north'] = sorted([n for n in nodes if n.name.endswith('_N')], key=lambda n: n.x)
    supports['south'] = sorted([n for n in nodes if n.name.endswith('_S')], key=lambda n: n.x)
    supports['east'] = sorted([n for n in nodes if n.name.startswith('E_')], key=lambda n: n.z)
    supports['west'] = sorted([n for n in nodes if n.name.startswith('W_')], key=lambda n: n.z)

    if walls:
        floored = []
        for support_side, support_nodes in supports.items():
            for node in support_nodes:
                if node.name not in floored:
                    floored.append(node.name)
                    frame.add_node(f'{node.name}_floor', node.x, 0, node.z)

        for support_side, support_nodes in supports.items():
            for i, _ in enumerate(support_nodes[:-1]):
                frame.add_quad(
                    f'{support_side}_wall{i}',
                    support_nodes[i].name,
                    f'{support_nodes[i].name}_floor',
                    f'{support_nodes[i+1].name}_floor',
                    support_nodes[i+1].name,
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
                    frame.def_support(node.name, True, True, True, True, True, True)


def create_model(
    east_joists: MemberSpec,
    west_joists: MemberSpec,
    tail_joists: MemberSpec,
    trimmers: MemberSpec,
    header: MemberSpec,
    planks: MemberSpec,
    params: DesignParameters = DEFAULT_PARAMS,
    walls: bool = False) -> Tuple:
    
    nodes, members = calculate_nodes_and_members(east_joists, west_joists, tail_joists, trimmers, header, planks, params)

    frame = assemble_frame(nodes, members)
    
    define_supports(frame, nodes, params.wall_thickness, 'brick', walls=walls)
    
    for member in members:
        geom = member.spec.get_geometry()
        material = MATERIAL_STRENGTHS[member.spec.material]
        dead_load = -geom.A * material['rho']
        frame.add_member_dist_load(member.name, 'FY', dead_load, dead_load)
    
    return frame


def render(frame, deformed_scale=100, opacity=0.25):
    def _set_wall_opacity(plotter, opacity=0.25):
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
    rndr.post_update_callbacks.append(lambda plotter: _set_wall_opacity(plotter, opacity=opacity))
    rndr.render_model()


if __name__ == '__main__':
    frame = create_model(
        east_joists=MemberSpec('W60x120', quantity=2, padding=50),
        west_joists=MemberSpec('W60x120', quantity=2, padding=50),
        tail_joists=MemberSpec('W60x120', quantity=0, padding=0),
        trimmers=MemberSpec('W80x160', quantity=2),
        header=MemberSpec('W60x120', quantity=1),
        planks=MemberSpec('W600x18'),
        walls=True
    )
    frame.analyze(check_statics=True)
    render(frame)