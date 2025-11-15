print('Starting...')

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Literal
from Pynite import FEModel3D
from Pynite.Rendering import Renderer
import numpy as np
import pandas as pd
from collections import defaultdict
import bisect
import math

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor


@dataclass
class DesignParameters:
    room_length: float
    room_width: float
    room_height: float
    opening_width: float
    opening_length: float
    opening_x_start: float
    live_load_mpa: float
    wall_thickness: float

    @property
    def wall_beam_contact_depth(self):
        return self.wall_thickness / 2 - (WALL_BRICK_PARAMS.interior_wall_thickness / 2)

    @property
    def floor_z(self):
        return self.room_height / 2

    @property
    def beam_length(self):
        return self.room_width + self.wall_beam_contact_depth
    
    @property
    def opening_y(self):
        return self.beam_length - self.opening_width - self.wall_beam_contact_depth/2

@dataclass
class WallBrickParameters:
    thickness: float
    length: float
    height: float
    exterior_wall_thickness: float
    interior_wall_thickness: float
    cavity_width: float
    cavity_height: float

    @property
    def cavity_depth(self):
        return (self.thickness - (2 * self.exterior_wall_thickness) - self.interior_wall_thickness) / 2


def prep_data():
    # Units are mm, N, and MPa (N/mm²)
    # INPUT PARAMS
    input_params = pd.read_csv('data/design_parameters.csv').iloc[0].to_dict()
    input_params = DesignParameters(**input_params)

    # MATERIAL STRENGTH
    material_str = pd.read_csv('data/material_strengths.csv')
    material_str['G_05'] = material_str['G'] * 5/6
    material_str['E_05'] = material_str['E_0'] * 2/3
    material_str = material_str.set_index('material').to_dict(orient='index')

    # CONNECTORS
    connectors = pd.read_csv('data/connectors.csv')
    train_data = connectors.dropna(subset=['grams'])
    predict_data = connectors[connectors['grams'].isnull()].copy()

    ## Fill missing weights
    X_train = train_data[['base', 'height']]
    y_train = train_data['grams']
    X_predict = predict_data[['base', 'height']]
    model = LinearRegression()
    model.fit(X_train, y_train)
    predict_data['grams'] = model.predict(X_predict)
    connectors = pd.concat([train_data, predict_data])

    ## Smoothen price and groupby base and height
    X = connectors[['base', 'height']]
    y = connectors['price_unit']
    huber = HuberRegressor(epsilon=1.24)
    huber.fit(X, y)
    connectors['price_unit'] = huber.predict(X)
    connectors['weight_N'] = connectors['grams'] / 100
    connectors = connectors[['base', 'height', 'weight_N', 'price_unit']].groupby(['base', 'height']).mean()
    connectors = connectors.reset_index()

    # MATERIAL_CATALOG
    material_catalog = pd.read_csv('data/material_catalog.csv')
    material_catalog[['base', 'height']] = material_catalog[['base', 'height']].fillna(0).astype(int)
    material_catalog['id'] = (material_catalog['material'] + '_' + material_catalog['base'].astype(str) + 'x' + material_catalog['height'].astype(str))

    ## Create fitting double beams
    doubled_beams = material_catalog[(material_catalog['type'] == 'beam') & (material_catalog['material'] != 'steel')].copy()
    doubled_beams['base'] = doubled_beams['base'] * 2
    doubled_beams['type'] = 'double'
    doubled_beams['source'] = doubled_beams['id']
    material_catalog = pd.concat([material_catalog, doubled_beams], ignore_index=True)
    material_catalog['viable_connector'] = material_catalog['base'].isin(connectors['base'])

    ## Standardize floor materials to 200mm width
    is_floor = material_catalog['type'] == 'floor'
    material_catalog.loc[is_floor, 'cost_unit'] = material_catalog.loc[is_floor, 'cost_unit'] * (200 / material_catalog.loc[is_floor, 'base'])
    material_catalog.loc[is_floor, 'base'] = 200

    material_catalog['id'] = (material_catalog['material'] + '_' + material_catalog['base'].astype(str) + 'x' + material_catalog['height'].astype(str))

    # EUROCODE FACTORS
    eurocode_factors = pd.read_csv('data/eurocode_material_factors.csv').set_index('material_prefix').to_dict(orient='index')

    # WALL_BRICK_PARAMS
    wall_brick_params = pd.read_csv('data/bricks.csv')
    wall_brick_params['index'] = wall_brick_params['thickness']
    wall_brick_params = wall_brick_params.set_index('index').to_dict(orient='index')
    wall_brick_params = WallBrickParameters(**wall_brick_params[input_params.wall_thickness])

    return input_params, material_str, material_catalog, connectors, eurocode_factors, wall_brick_params


INPUT_PARAMS, MATERIAL_STRENGTHS, MATERIAL_CATALOG, CONNECTORS, EUROCODE_FACTORS, WALL_BRICK_PARAMS = prep_data()

DL_COMBO = 'DL'
LL_COMBO = 'LL'
ULS_COMBO = 'ULS_Strength'


def _get_eurocode_factors(material_name):
    if material_name.startswith('c'):
        return EUROCODE_FACTORS['c']
    elif material_name.startswith('gl'):
        return EUROCODE_FACTORS['gl']
    elif material_name.startswith('osb'):
        return EUROCODE_FACTORS['osb']
    elif material_name.startswith('p'):
        return EUROCODE_FACTORS['p']
    elif material_name.startswith('s'):
        return EUROCODE_FACTORS['s']
    elif material_name.startswith('brick'):
        return EUROCODE_FACTORS['brick']
    else:
        raise ValueError("Material not supported")

@dataclass
class CrossSectionProperties:
    A: float
    Iy: float
    Iz: float
    J: float
    
    @classmethod
    def from_rectangular(cls, base: float, height: float) -> 'CrossSectionProperties':
        A = base * height
        b, h = min(base, height), max(base, height)
        J = (b**3 * h) * (1/3 - 0.21 * (b/h) * (1 - (b**4)/(12*h**4)))
        Iz = (height * base**3) / 12
        Iy = (base * height**3) / 12
        return cls(A=A, Iy=Iy, Iz=Iz, J=J)
    
    # @classmethod
    # def from_i_beam(cls, height: float, base: float, flange_thickness: float, web_thickness: float) -> 'CrossSectionProperties':
    #     A_flanges = 2 * base * flange_thickness
    #     A_web = (height - 2 * flange_thickness) * web_thickness
    #     A = A_flanges + A_web
        
    #     Iz_flanges = 2 * (base * flange_thickness**3 / 12 + 
    #                      base * flange_thickness * ((height - flange_thickness)/2)**2)
    #     web_height = height - 2 * flange_thickness
    #     Iz_web = web_thickness * web_height**3 / 12
    #     Iz = Iz_flanges + Iz_web
        
    #     Iy_flanges = 2 * (flange_thickness * base**3 / 12)
    #     Iy_web = web_height * web_thickness**3 / 12
    #     Iy = Iy_flanges + Iy_web
        
    #     J = (2 * base * flange_thickness**3 + web_height * web_thickness**3) / 3
    #     return cls(A=A, Iy=Iy, Iz=Iz, J=J)

@dataclass
class MemberSpec:
    material_id: str
    quantity: Optional[int] = None
    padding: Optional[float] = None
    
    def __post_init__(self):
        self._catalog_data = MATERIAL_CATALOG[MATERIAL_CATALOG['id'] == self.material_id].iloc[0]
        self.material = self._catalog_data['material']
        self.type = self._catalog_data['type']
        self.base = self._catalog_data['base']
        self.height = self._catalog_data['height']
        self.length = self._catalog_data['length']
        self.shape = self._catalog_data['shape']
        self.cost_per_m3 = self._catalog_data['cost_unit']
    
    @property
    def section_name(self) -> str:
        return f"sec_{self.material_id}"
    
    def get_geometry(self) -> CrossSectionProperties:
        if self.shape == 'rectangular':
            return CrossSectionProperties.from_rectangular(self.base, self.height)
        # elif self.shape.startswith('IP'):
        #     return CrossSectionProperties.from_i_beam(self.height, self.base, self._catalog_data['flange_thickness'], self._catalog_data['web_thickness'])
        else:
            raise ValueError(f"Unknown shape: {self.shape}")
    
    def create_section(self, frame: FEModel3D):
        if self.section_name not in frame.sections:
            geom = self.get_geometry()
            frame.add_section(self.section_name, geom.A, Iy=geom.Iy, Iz=geom.Iz, J=geom.J)

@dataclass
class NodeLocation:
    name: str
    X: float
    Y: float
    Z: float

@dataclass
class Member:
    name: str
    node_i: str
    node_j: str
    spec: MemberSpec
    

def calculate_nodes_and_members(hyperparams: dict, free_tail=False) -> Tuple[List[NodeLocation], List[Member]]:

    def _calculate_evenly_spaced_positions(
        n: int, 
        clear_start: float, 
        clear_end: float, 
        member_base: float,
        distribution: Literal['start_aligned', 'end_aligned', 'centered'],
        tail_padding=None) -> List[float]:

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
            if tail_padding:
                positions = np.linspace(centerline_start, centerline_end, n).tolist()
                return positions
            positions = np.linspace(centerline_start, centerline_end, n + 2).tolist()
            return positions[1:-1]
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    assert hyperparams['trimmerW'].quantity == 1, "Must have exactly 1 west trimmer"
    assert hyperparams['trimmerE'].quantity == 1, "Must have exactly 1 east trimmer"
    assert hyperparams['header'].quantity == 1, "Must have exactly 1 header"
    
    # Stairs opening params
    opening_x_end = INPUT_PARAMS.opening_x_start + INPUT_PARAMS.opening_length
    trimmer_W_x = INPUT_PARAMS.opening_x_start - (hyperparams['trimmerW'].base / 2)
    trimmer_E_x = opening_x_end + (hyperparams['trimmerE'].base / 2)

    header_y = INPUT_PARAMS.opening_y - hyperparams['header'].base / 2
    
    # Beam centerline positions
    beam_positions = {}
    if hyperparams['west_joists'].quantity > 0:
        clear_start = hyperparams['west_joists'].padding
        clear_end = trimmer_W_x - hyperparams['trimmerW'].base / 2
        x_positions = _calculate_evenly_spaced_positions(hyperparams['west_joists'].quantity,
                                                         clear_start,
                                                         clear_end,
                                                         hyperparams['west_joists'].base,
                                                         'start_aligned')
        beam_positions['west_joists'] = [(f'west{i}', x) for i, x in enumerate(x_positions)]
    
    if hyperparams['east_joists'].quantity > 0:
        clear_start = trimmer_E_x + hyperparams['trimmerE'].base / 2
        clear_end = INPUT_PARAMS.room_length - hyperparams['east_joists'].padding
        x_positions = _calculate_evenly_spaced_positions(hyperparams['east_joists'].quantity,
                                                         clear_start,
                                                         clear_end,
                                                         hyperparams['east_joists'].base,
                                                         'end_aligned')
        beam_positions['east_joists'] = [(f'east{i}', x) for i, x in enumerate(x_positions)]
    
    if hyperparams['tail_joists'].quantity > 0:
        clear_end = opening_x_end - hyperparams['tail_joists'].padding
        if free_tail:
            clear_start = INPUT_PARAMS.opening_x_start
        else:
            clear_start = INPUT_PARAMS.opening_x_start + hyperparams['tail_joists'].padding
        x_positions = _calculate_evenly_spaced_positions(hyperparams['tail_joists'].quantity,
                                                         clear_end,
                                                         clear_start,
                                                         hyperparams['tail_joists'].base,
                                                         'centered',
                                                         tail_padding=hyperparams['tail_joists'].padding)
        beam_positions['tail_joists'] = [(f'tail{i}', x) for i, x in enumerate(x_positions)]
    
    # Plank centerline positions
    clear_start = INPUT_PARAMS.wall_beam_contact_depth / 2
    clear_end = INPUT_PARAMS.beam_length - INPUT_PARAMS.wall_beam_contact_depth / 2
    hyperparams['planks'].quantity = int((clear_end - clear_start) // hyperparams['planks'].base)
    y_positions = _calculate_evenly_spaced_positions(hyperparams['planks'].quantity,
                                                     clear_start-hyperparams['planks'].base,
                                                     clear_end+hyperparams['planks'].base,
                                                     hyperparams['planks'].base,
                                                     'centered')
    plank_positions = [(f'p{i}', y) for i, y in enumerate(y_positions)]
    
    # trimmer centerlines
    beam_positions['trimmerW'] = [('trimmerW', trimmer_W_x)]
    beam_positions['trimmerE'] = [('trimmerE', trimmer_E_x)]

    # joist nodes and Member creation
    nodes = []
    members = []
    for group_name, group_positions in beam_positions.items():
        spec = hyperparams[group_name]
        for beam_name, x in group_positions:
            nodes.append(NodeLocation(f'{beam_name}_S', x, 0, INPUT_PARAMS.floor_z))
            if group_name == 'tail_joists': # Tails connect to header (header_y), not to wall (beam_length)
                nodes.append(NodeLocation(f'{beam_name}_header', x, header_y, INPUT_PARAMS.floor_z))
                members.append(Member(name=beam_name, node_i=f'{beam_name}_S', node_j=f'{beam_name}_header', spec=spec))
            else:
                nodes.append(NodeLocation(f'{beam_name}_N', x, INPUT_PARAMS.beam_length, INPUT_PARAMS.floor_z))
                members.append(Member(name=beam_name, node_i=f'{beam_name}_S', node_j=f'{beam_name}_N', spec=spec))
    
    nodes.append(NodeLocation('headerE', trimmer_E_x, header_y, INPUT_PARAMS.floor_z))
    nodes.append(NodeLocation('headerW', trimmer_W_x, header_y, INPUT_PARAMS.floor_z))
    members.append(Member(name='header', node_i='headerW', node_j='headerE', spec=hyperparams['header']))

    # Corner nodes
    walls = [('W', 0), ('E', INPUT_PARAMS.room_length)]
    for corner_name, x in walls:
        nodes.append(NodeLocation(f'{corner_name}_S', x, 0, INPUT_PARAMS.floor_z))
        nodes.append(NodeLocation(f'{corner_name}_N', x, INPUT_PARAMS.beam_length, INPUT_PARAMS.floor_z))

    # Plank nodes and member locations
    plank_series_dict = {}
    beam_positions['walls'] = walls # So planks extend to walls
    for plank_name, y in plank_positions:
        plank_nodes = []
        for group_positions in beam_positions.values():
            for beam_name, x in group_positions:
                if x > trimmer_W_x and x < trimmer_E_x and y > header_y:
                    continue
                intersection_node = NodeLocation(f'{beam_name}-{plank_name}', x, y, INPUT_PARAMS.floor_z)
                plank_nodes.append(intersection_node)
                nodes.append(intersection_node)
        plank_series_dict[plank_name] = plank_nodes
    
    for plank_name, plank_nodes in plank_series_dict.items():
        plank_nodes.sort(key=lambda node: node.X)
        for i, _ in enumerate(plank_nodes[:-1]):
            if plank_nodes[i].X == trimmer_W_x and plank_nodes[i].Y > header_y:
                continue
            members.append(Member(name=f'{plank_name}_{i}', node_i=plank_nodes[i].name, node_j=plank_nodes[i+1].name, spec=hyperparams['planks']))
    
    return nodes, members

    
def assemble_frame(nodes: List[NodeLocation], members: List[Member]) -> Tuple[FEModel3D, Dict[str, MemberSpec]]:
    frame = FEModel3D()
    for mat_name, props in MATERIAL_STRENGTHS.items():
        if props['G'] is None:
            G = props['E_0'] / (2 * (1 + props['nu']))
        else:
            G = props['G']
        frame.add_material(
            mat_name,
            E=props['E_0'],
            G=G,
            nu=props['nu'],
            rho=props['rho']
        )
    
    for node in nodes:
        frame.add_node(node.name, node.X, node.Y, node.Z)
    
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


def define_supports(frame, nodes, wall_thickness, material, walls=False) -> None:
    supports = {}
    supports['north'] = sorted([n for n in nodes if n.name.endswith('_N')], key=lambda n: n.X)
    supports['south'] = sorted([n for n in nodes if n.name.endswith('_S')], key=lambda n: n.X)
    supports['east'] = sorted([n for n in nodes if n.name.startswith('E_')], key=lambda n: n.Y)
    supports['west'] = sorted([n for n in nodes if n.name.startswith('W_')], key=lambda n: n.Y)

    if walls:
        foundation = []
        for support_side, support_nodes in supports.items():
            for node in support_nodes:
                if node.name not in foundation:
                    foundation.append(node.name)
                    frame.add_node(f'{node.name}_foundation', node.X, node.Y, 0)

        for support_side, support_nodes in supports.items():
            for i, _ in enumerate(support_nodes[:-1]):
                frame.add_quad(
                    f'{support_side}_wall{i}',
                    support_nodes[i].name,
                    f'{support_nodes[i].name}_foundation',
                    f'{support_nodes[i+1].name}_foundation',
                    support_nodes[i+1].name,
                    wall_thickness, 
                    material
                )
        for node_name in frame.nodes:
            if node_name.endswith('_foundation'):
                frame.def_support(node_name, True, True, True, True, True, True)
    else:
        for support_side, support_nodes in supports.items():
            if support_side in ['north', 'south']:
                for node in support_nodes:
                    frame.def_support(node.name, True, True, True, True, True, True)


def _find_compatible_connector(base: float, height: float):
    member_connectors = CONNECTORS[CONNECTORS['base'] == base]
    connector = member_connectors[member_connectors['height'] <= height]
    if connector.empty:
        return member_connectors.mean()
    return connector.mean()
    

def apply_loads(frame: FEModel3D, members: List[Member], exclude_loads_under_big_beam=False) -> Tuple[float, float]:
    total_dl_force, total_ll_force = 0.0, 0.0

    # Dead loads
    for member in members:
        geom = member.spec.get_geometry()
        material = MATERIAL_STRENGTHS[member.spec.material]
        dead_load = -geom.A * material['rho']
        frame.add_member_dist_load(member.name, 'FZ', dead_load, dead_load, case=DL_COMBO)
        total_dl_force += dead_load * frame.members[member.name].L()

        if member.name.startswith('tail'):
            header = next((m for m in members if m.name.startswith('header')))
            connector = _find_compatible_connector(base=member.spec.base, height=header.spec.height)
            frame.add_member_pt_load(member.name, 'FZ', -connector['weight_N'], frame.members[member.name].L(), case=DL_COMBO)
            total_dl_force += -connector['weight_N']
        elif member.name.startswith('header'):
            trimmer = next((m for m in members if m.name.startswith('trimmer')))
            connector = _find_compatible_connector(base=member.spec.base, height=trimmer.spec.height)
            frame.add_member_pt_load(member.name, 'FZ', -connector['weight_N'], 0, case=DL_COMBO)
            frame.add_member_pt_load(member.name, 'FZ', -connector['weight_N'], frame.members[member.name].L(), case=DL_COMBO)
            total_dl_force += -connector['weight_N'] * 2
        elif member.name.startswith('trimmerE'):
            y1 = INPUT_PARAMS.opening_y
            y2 = frame.members[member.name].L() - INPUT_PARAMS.wall_beam_contact_depth/2

            stair_angle = 57 # TO-DO: fetch from configs
            stair_radians = stair_angle * math.pi / 180
            total_dl_N = 258 # 258 for the stair's packaged weight. TO-DO: fetch from configs
            total_ll_N = 1200 # 1000 for a big person. TO-DO: fetch from configs

            # Calc forces for upper floor
            dl_N = math.cos(stair_radians)/2 * total_dl_N
            ll_N = math.cos(stair_radians)/2 * total_ll_N

            vertical_line_dl = math.sin(stair_radians) * -dl_N / (y2 - y1) 
            lateral_line_dl = math.cos(stair_radians) * dl_N / (y2 - y1) 
            frame.add_member_dist_load(member.name, 'FZ', vertical_line_dl, vertical_line_dl, case=DL_COMBO, x1=y1, x2=y2)
            frame.add_member_dist_load(member.name, 'FY', lateral_line_dl, lateral_line_dl, case=DL_COMBO, x1=y1, x2=y2)

            vertical_line_ll = math.sin(stair_radians) * -ll_N / (y2 - y1) 
            lateral_line_ll = math.cos(stair_radians) * ll_N / (y2 - y1) 
            frame.add_member_dist_load(member.name, 'FZ', vertical_line_ll, vertical_line_ll, case=LL_COMBO, x1=y1, x2=y2)
            frame.add_member_dist_load(member.name, 'FY', lateral_line_ll, lateral_line_ll, case=LL_COMBO, x1=y1, x2=y2)

    # Live loads
    plank_members = [m for m in members if m.name.startswith('p')]
    plank_y_values = sorted(set(frame.nodes[m.node_i].Y for m in plank_members))
    standard_spacing = plank_y_values[1] - plank_y_values[0]
    
    tail_planks = []
    for m in plank_members:
        if 'tail' in m.node_i or 'tail' in m.node_j or ('trimmer' in m.node_i and 'trimmer' in m.node_j):
            tail_planks.append(m)
    
    min_y = min(plank_y_values)
    max_y = max(plank_y_values)
    max_tail_plank_y = max(frame.nodes[m.node_i].Y for m in tail_planks)
    for member in plank_members:
        member_y = frame.nodes[member.node_i].Y
        
        if member_y == min_y:
            tributary_width = member.spec.base / 2 + standard_spacing / 2
        elif member_y == max_y:
            tributary_width = member.spec.base / 2 + standard_spacing / 2
        elif member_y == max_tail_plank_y and member in tail_planks:
            tributary_width = INPUT_PARAMS.opening_y - member_y + standard_spacing / 2
        else:
            tributary_width = standard_spacing
        
        if exclude_loads_under_big_beam and member_y < 600: # 600 is the clear distance of the structural beam hovering above the floor. This beam will take the weight of this region.
            live_load = -INPUT_PARAMS.live_load_mpa * tributary_width * 0.2 # For the little that fits under the structural beam
        else:
            live_load = -INPUT_PARAMS.live_load_mpa * tributary_width

        frame.add_member_dist_load(member.name, 'FZ', live_load, live_load, case=LL_COMBO)
        total_ll_force += live_load * frame.members[member.name].L()

    # Declare loads
    frame.add_load_combo(DL_COMBO, {'DL': 1})
    frame.add_load_combo(LL_COMBO, {'LL': 1})
    frame.add_load_combo(ULS_COMBO, {'DL': 1.35, 'LL': 1.5})

    return total_dl_force, total_ll_force


def create_model(hyperparams: dict, walls: bool = True, exclude_loads_under_big_beam=False, free_tail=False) -> Tuple:
    
    nodes, members = calculate_nodes_and_members(hyperparams, free_tail=free_tail)
    frame = assemble_frame(nodes, members)
    define_supports(frame, nodes, INPUT_PARAMS.wall_thickness, 'brick', walls=walls)
    total_dl_force, total_ll_force = apply_loads(frame, members, exclude_loads_under_big_beam=exclude_loads_under_big_beam)
    
    return frame, nodes, members


def calculate_purchase_quantity(frame: FEModel3D, members: List[Member], skip_planks=False):

    def _find_optimal_cuts(member_lengths: List[float], stock_length: float):
        member_lengths.sort(reverse=True)
        stock_parts = []
        remaining_lengths_in_stock_parts = []
        for member_len in member_lengths:
            index_for_best_stock_part = bisect.bisect_left(remaining_lengths_in_stock_parts, member_len)
            
            if index_for_best_stock_part < len(remaining_lengths_in_stock_parts):
                old_remaining_lengths = remaining_lengths_in_stock_parts.pop(index_for_best_stock_part)
                best_stock_part = stock_parts.pop(index_for_best_stock_part)
                best_stock_part.append(member_len)

                new_remaining_length = old_remaining_lengths - member_len
                
                new_index_for_stock_part = bisect.bisect_left(remaining_lengths_in_stock_parts, new_remaining_length)
                remaining_lengths_in_stock_parts.insert(new_index_for_stock_part, new_remaining_length)
                stock_parts.insert(new_index_for_stock_part, best_stock_part)
                
            else:
                new_remaining_length = stock_length - member_len
                
                new_insertion_index = bisect.bisect_left(remaining_lengths_in_stock_parts, new_remaining_length)
                remaining_lengths_in_stock_parts.insert(new_insertion_index, new_remaining_length)
                stock_parts.insert(new_insertion_index, [member_len])

        return len(stock_parts), stock_parts
    
    def _calc_mortar_kg_required(beam_base, beam_height):

        def _calc_total_cavity_size(num_cavities, cavity_size):
            return num_cavities * cavity_size + (num_cavities-1) * WALL_BRICK_PARAMS.interior_wall_thickness

        num_cavities_x = beam_base // WALL_BRICK_PARAMS.cavity_width
        while _calc_total_cavity_size(num_cavities_x, WALL_BRICK_PARAMS.cavity_width) < beam_base:
            num_cavities_x += 1
        required_opening_width = _calc_total_cavity_size(num_cavities_x, WALL_BRICK_PARAMS.cavity_width)

        num_cavities_y = beam_height // WALL_BRICK_PARAMS.cavity_height
        while _calc_total_cavity_size(num_cavities_y, WALL_BRICK_PARAMS.cavity_height) < beam_height:
            num_cavities_y += 1
        required_opening_height = _calc_total_cavity_size(num_cavities_y, WALL_BRICK_PARAMS.cavity_height)

        wall_niche_volume = required_opening_width * required_opening_height * WALL_BRICK_PARAMS.cavity_depth
        beam_volume = beam_base * beam_height * WALL_BRICK_PARAMS.cavity_depth
        req_vol = wall_niche_volume - beam_volume
        mm_3_per_kg = 1000000 / 1.8 # Calculated from 1.8kg/m ²/mm

        return req_vol/mm_3_per_kg

    total_cost = 0.0
    mortar_required = 0
    materials = defaultdict(list)
    for member in members:
        pynite_member = frame.members[member.name]
        member_length = float(pynite_member.L())

        # Members
        if member.spec.type == 'double':
            source_id = MATERIAL_CATALOG[MATERIAL_CATALOG['id'] == member.spec.material_id].iloc[0]['source']
            materials[source_id].extend([member_length] * 2)
        else:
            if skip_planks and member.name.startswith('p'):
                continue
            material_id = member.spec.material_id
            materials[material_id].extend([member_length])

        # Connectors and mortar
        if member.name.startswith('p'):
            continue
        elif member.name.startswith('header'):
            trimmer = next((m for m in members if m.name.startswith('trimmer')))
            connector = _find_compatible_connector(base=member.spec.base, height=trimmer.spec.height)
            total_cost += connector['price_unit'] * 2
        elif member.name.startswith('tail'):
            header = next((m for m in members if m.name.startswith('header')))
            connector = _find_compatible_connector(base=member.spec.base, height=header.spec.height)
            total_cost += connector['price_unit']
            mortar_required += _calc_mortar_kg_required(member.spec.base, member.spec.height)
        else:
            mortar_required += _calc_mortar_kg_required(member.spec.base, member.spec.height) * 2

    all_material_cuts = {}
    for material_id, lengths in materials.items():
        material_specs = MATERIAL_CATALOG[MATERIAL_CATALOG['id'] == material_id].iloc[0]
        num_beams_to_buy, current_material_cuts = _find_optimal_cuts(lengths, material_specs['length'])
        all_material_cuts[material_id] = current_material_cuts
        total_cost += num_beams_to_buy * material_specs['cost_unit']
    
    mortar = MATERIAL_CATALOG[MATERIAL_CATALOG['type'] == 'mortar'].iloc[0]
    mortar_units = math.ceil(mortar_required / mortar.weight)
    all_material_cuts['mortar'] = mortar_required
    total_cost += mortar_units * mortar.cost_unit

    return total_cost, all_material_cuts


def evaluate_stresses(frame: FEModel3D, members: List[Member]):
    # Following chapters in https://www.swedishwood.com/siteassets/5-publikationer/pdfer/sw-design-of-timber-structures-vol2-2022.pdf

    def _calc_size_factor_kh(spec):
        """
        Chapter 3.3 Size effects (EN 1995-1-1, 3.2-3.4)
        Used for bending and tension

        k_h : size factor 
        """
        if spec.material.startswith('c') and spec.height <= 150:
            k_h = min((150/spec.height)**0.2, 1.3)
        elif spec.material.startswith('gl') and spec.height <= 600:
            k_h = min((600/spec.height)**0.1, 1.1)
        else:
            k_h = 1
        return k_h


    def _calc_bending_moment_capacity(factors, material_props, spec, geometry, member_length):
        """
        Chapter 4 Bending (EN 1995-1-1, 6.3.3)
        
        W_y : section modulus about the major axis (y)
        W_z : section modulus about the minor axis (z)
        M_y_crit : critical bending moment about the major axis (y)
        l_ef : effective length of the beam for a uniformly distributed load
        omega_m_crit : critical bending stress calculated according to the classical theory of lateral stability, using 5-percentile stiffness values (EN 1995-1-1, 6.3.3)
        lambda_rel_m : relative slenderness ratio in bending
        k_crit : factor accounting for the effect of lateral buckling
        f_mk : characteristic bending strength
        f_md : design value of bending strength

        M_yRd : bending moment capacity about the major axis (y)
        M_zRd : bending moment capacity about the minor axis (z)
        """
        W_y = geometry.Iy / (spec.height / 2)
        W_z = geometry.Iz / (spec.base / 2)
        M_y_crit = math.pi * math.sqrt(material_props['E_05'] * geometry.Iz * material_props['G_05'] * geometry.J)

        l_ef = member_length * 0.9
        omega_m_crit = M_y_crit / (W_y * l_ef)

        lambda_rel_m = math.sqrt(material_props['f_mk'] / omega_m_crit)
        if lambda_rel_m <= 0.75:
            k_crit = 1.0
        elif lambda_rel_m <= 1.4:
            k_crit = 1.56 - 0.75 * lambda_rel_m
        else:
            k_crit = 1 / lambda_rel_m ** 2

        k_h = _calc_size_factor_kh(spec)
        k_mod = factors['k_mod']
        gamma_M = factors['gamma_M']
        f_md = (k_h * material_props['f_mk'] * k_mod) / gamma_M
        M_yRd = f_md * W_y * k_crit
        M_zRd = f_md * W_z

        return M_yRd, M_zRd


    def _calc_axial_tension_capacity(factors, material_props, spec, geometry):
        """
        Chapter 5.1 Axial Tension

        f_t0k : characteristic tension strength parallel to the grain
        f_t0d : design tension strength parallel to the grain
        f_t90k : characteristic tension strength perpendicular to the grain
        f_t90d : design tension strength perpendicular to the grain

        N_t0Rd : design load capacity in axial tension about the strong axis (y)
        N_t90Rd : design load capacity in axial tension about the weak axis (z)
        """
        k_mod = factors['k_mod']
        gamma_M = factors['gamma_M']
        k_h = _calc_size_factor_kh(spec)

        f_t0d = (k_h * material_props['f_t0k'] * k_mod) / gamma_M
        f_t90d = (k_h * material_props['f_t90k'] * k_mod) / gamma_M
        N_t0Rd = f_t0d * geometry.A
        N_t90Rd = f_t90d * geometry.A

        return N_t0Rd, N_t90Rd


    def _calc_instability_factor(factors, material_props, geometry, member_length):
        """
        Chapter 5.2 Compression (EN 1995-1-1, 6.3.2)

        i_y : radius of gyration about the strong axis (y)
        i_z : radius of gyration about the weak axis (z)
        l_ef : effective length
        lambda_y, lambda_rel_y : slenderness ratios corresponding to bending about the y-axis (deflection in the z-direction)
        lambda_z, lambda_rel_z : slenderness ratios corresponding to bending about the z-axis (deflection in the y-direction)
        f_c0k : characteristic compression strength parallel to grain

        k, k_c : instability factors
        """
        beta_c = factors['straightness_factor_beta_c']
        
        i_y = math.sqrt(geometry.Iy / geometry.A)
        i_z = math.sqrt(geometry.Iz / geometry.A)
        l_ef = member_length * 0.9
        lambda_y = l_ef / i_y
        lambda_z = l_ef / i_z
        lambda_y_rel = (lambda_y / math.pi) * math.sqrt(material_props['f_c0k'] / material_props['E_05'])
        lambda_z_rel = (lambda_z / math.pi) * math.sqrt(material_props['f_c0k'] / material_props['E_05'])
        risk_buckling = lambda_y_rel > 0.3 or lambda_z_rel > 0.3

        k_y = (0.5 * (1 + beta_c * (lambda_y_rel - 0.3) + lambda_y_rel**2))
        k_z = (0.5 * (1 + beta_c * (lambda_z_rel - 0.3) + lambda_z_rel**2))
        k_cy = 1 / (k_y + math.sqrt(k_y**2 - lambda_y_rel**2))
        k_cz = 1 / (k_z + math.sqrt(k_z**2 - lambda_z_rel**2))

        return k_cy, k_cz, risk_buckling


    def _calc_axial_compression_capacity(factors, material_props, spec, geometry, member_length):
        """
        Chapter 5.2 Axial Compression (EN 1995-1-1, 6.1.5)
        
        f_c0k : characteristic compression strength parallel to grain
        f_c0d : design compression strength parallel to grain
        f_c90k : characteristic compression strength perpendicular to grain
        f_c90d : design compression strength perpendicular to grain
        k_cz : instability factor c about the weak axis (z)
        k_cy : instability factor c about the strong axis (y)
        A_ef : effective contact area in compression

        N_c0Rd : design load capacity in axial compression about the strong axis (y)
        N_c90Rd : design load capacity in axial compression about the weak axis (z)
        """
        k_mod = factors['k_mod']
        gamma_M = factors['gamma_M']
        f_c0d = (material_props['f_c0k'] * k_mod) / gamma_M
        f_c90d = (material_props['f_c90k'] * k_mod) / gamma_M
        k_c90 = factors['config_and_deformation_factor_k_c90']
        k_cy, k_cz, risk_buckling = _calc_instability_factor(factors, material_props, geometry, member_length)

        N_c0Rd = f_c0d * geometry.A * k_cy
        A_ef = spec.base * INPUT_PARAMS.wall_beam_contact_depth
        N_c90Rd = k_c90 * f_c90d * A_ef

        return N_c0Rd, N_c90Rd, k_cy, k_cz, risk_buckling


    def _calc_shear_ratio(factors, material_props, spec, V_Ed):
        """
        Chapter 6 Cross section subjected to shear (EN 1995-1-1, 6.1.7)
        
        b_ef : effective width for area calculations
        f_vk : characteristic shear strength
        f_vd : design shear strength
        V_Rd : design load capacity in shear strength
        """
        k_mod = factors['k_mod']
        gamma_M = factors['gamma_M']
        k_cr = factors['k_cr']

        if spec.material.startswith('c') or spec.material.startswith('gl'):
            b_ef = spec.base * k_cr
        else:
            b_ef = spec.base

        f_vd = (material_props['f_vk'] * k_mod) / gamma_M
        V_Rd = f_vd * (b_ef * spec.height) / 1.5

        return V_Ed / V_Rd
    

    def _calc_torsion_ratio(factors, material_props, spec, T_Ed):
        """
        (EN 1995-1-1, 6.2.4, 6.1.8)

        k_shape : factor depending on the shape of the cross-section
        W_t : torsional section modulus of a solid rectangular section
        f_vk : characteristic shear strength
        tau_tor_d : design torsional stress
        T_Rd : design load capacity in torsional strength
        """
        k_shape = min(1 + (0.15 * spec.height/spec.base), 2)
        k_t = 1 / (3 + 1.8 * (spec.base / spec.height))
        W_t = spec.height * spec.base**2 * k_t
        f_vd = (material_props['f_vk'] * factors['k_mod']) / factors['gamma_M']
        tau_tor_d = (T_Ed / W_t)
        T_Rd = (k_shape * f_vd)

        return tau_tor_d / T_Rd


    def _calc_combined_bending_axial_tension_ratio(M_yRd, M_zRd, N_t0Rd, M_yEd, M_zEd, N_t0Ed, k_m):
        """
        Chapter 7.2 Combined bending and axial tension (EN 1995-1-1, 6.2.3)

        k_m: reduction factor depending on cross-sectional shape

        M_yEd : design load effect from bending moments about the strong axis (y)
        M_zEd : design load effect from bending moments about the weak axis (z)
        N_t0Ed : design load effect from axial tension about the strong axis (y)

        M_yRd : design load capacity in bending moments about the strong axis (y)
        M_zRd : design load capacity in bending moments about the weak axis (z)
        N_t0Rd : design load capacity in axial tension about the strong axis (y)

        ratio_y, ratio_z : ratios for member stress over member capacity
        Member failure occurs when ratio_y or ratio_z exceed a value of 1
        """
        ratio_y = (M_yEd * k_m / M_yRd) + (M_zEd / M_zRd) + (N_t0Ed / N_t0Rd)
        ratio_z = (M_yEd / M_yRd) + (M_zEd * k_m / M_zRd) + (N_t0Ed / N_t0Rd)

        return ratio_y, ratio_z


    def _calc_combined_bending_axial_compression_ratio(risk_buckling, M_yRd, M_zRd, N_c0Rd, M_yEd, M_zEd, N_c0Ed, k_cy, k_cz, k_m):
        """
        Chapter 7.3 Combined bending and axial compression (EN 1995-1-1, 6.2.4, 6.3.2)

        k_m : reduction factor depending on cross-sectional shape
        k_cz : instability factor c about the weak axis (z)
        k_cy : instability factor c about the strong axis (y)

        M_yEd : design load effect from bending moments about the strong axis (y)
        M_zEd : design load effect from bending moments about the weak axis (z)
        N_c0Ed : design load effect from axial compression about the strong axis (y)

        M_yRd : design load capacity in bending moments about the strong axis (y)
        M_zRd : design load capacity in bending moments about the weak axis (z)
        N_c0Rd : design load capacity in axial compression about the strong axis (y)

        ratio_y, ratio_z : ratios for member stress over member capacity
        Member failure occurs when ratio_y or ratio_z exceed a value of 1
        """
        if risk_buckling:
            ratio_y = (M_yEd * k_m / M_yRd) + (M_zEd / M_zRd) + (N_c0Ed / (N_c0Rd * k_cz))
            ratio_z = (M_yEd / M_yRd) + (M_zEd * k_m / M_zRd) + (N_c0Ed / (N_c0Rd * k_cy))
        else:
            ratio_y = (M_yEd * k_m / M_yRd) + (M_zEd / M_zRd) + (N_c0Ed / N_c0Rd)**2
            ratio_z = (M_yEd / M_yRd) + (M_zEd * k_m / M_zRd) + (N_c0Ed / N_c0Rd)**2

        return ratio_y, ratio_z


    def _calc_compression_bearing_ratio(factors, material_props, spec, pynite_member, member_end):
        brick_props = MATERIAL_STRENGTHS['brick']
        mortar_props = MATERIAL_STRENGTHS['mortar']
        brick_gamma_M = 2.2
        alpha = 0.7
        beta = 0.3
        K = 0.45

        bearing_area = spec.base * INPUT_PARAMS.wall_beam_contact_depth    
        reaction_force_j = abs(pynite_member.shear('Fz', member_end, ULS_COMBO))
        bearing_pressure_j = reaction_force_j / bearing_area

        f_c90d_beam = (material_props['f_c90k'] * factors['k_mod']) / factors['gamma_M']
        f_kd = K * brick_props['f_c0k']**alpha * mortar_props['f_c0k']**beta / brick_gamma_M

        beam_bearing_ratio = bearing_pressure_j / f_c90d_beam
        brick_bearing_ratio = bearing_pressure_j / f_kd

        return beam_bearing_ratio, brick_bearing_ratio


    frame.analyze(check_stability=False)
    support_node_names = {node.name for node in frame.nodes.values() if node.name.endswith(('_N', '_S'))}
    
    member_evaluations = []
    for member in members:
        ratios = {}
        pynite_member = frame.members[member.name]
        spec = member.spec
        geometry = spec.get_geometry()
        material_props = MATERIAL_STRENGTHS[spec.material]
        factors = _get_eurocode_factors(spec.material)
        member_length = pynite_member.L()

        # ULS Internal Forces
        max_moment_y = pynite_member.max_moment('My', ULS_COMBO)
        min_moment_y = pynite_member.min_moment('My', ULS_COMBO)
        M_yEd = max(abs(max_moment_y), abs(min_moment_y))

        max_moment_z = pynite_member.max_moment('Mz', ULS_COMBO)
        min_moment_z = pynite_member.min_moment('Mz', ULS_COMBO)
        M_zEd = max(abs(max_moment_z), abs(min_moment_z))

        N_t0Ed = abs(min(0, pynite_member.min_axial(ULS_COMBO)))
        N_c0Ed = max(0, pynite_member.max_axial(ULS_COMBO))

        max_shear_z = pynite_member.max_shear('Fz', ULS_COMBO)
        min_shear_z = pynite_member.min_shear('Fz', ULS_COMBO)
        V_Ed = max(abs(max_shear_z), abs(min_shear_z))

        max_torsion = pynite_member.max_torque(ULS_COMBO)
        min_torsion = pynite_member.min_torque(ULS_COMBO)
        T_Ed = max(abs(max_torsion), abs(min_torsion))
        
        if spec.material.startswith('s'):
            pass
            # gamma_M = factors['gamma_M']
            # yield_strength = material_props['f_mk']
            
            # plastic_section_modulus_z = geometry.Iz / (spec.height / 2) * 1.12 if spec.height > 0 else 1e-9 # Approximation for I-sections
            
            # bending_resistance_d = plastic_section_modulus_z * yield_strength / gamma_M
            # axial_resistance_d = geometry.A * yield_strength / gamma_M
            # shear_resistance_d = (geometry.A * 0.58) * (yield_strength / np.sqrt(3)) / gamma_M
            
            # ratios['shear'] = V_Ed / shear_resistance_d
            # ratios['bending'] = M_yEd / bending_resistance_d
            # ratios['axial'] = abs(N_t0Ed) / axial_resistance_d
            # ratios['interaction'] = (abs(N_t0Ed) / axial_resistance_d) + (M_yEd / bending_resistance_d)
        else:
            if spec.shape == 'rectangular':
                k_m = 0.7
            else:
                k_m = 1

            # (EN 1995-1-1, 6.1.6)
            M_yRd, M_zRd = _calc_bending_moment_capacity(factors, material_props, spec, geometry, member_length)
            ratios['bending_y'] = (M_yEd / M_yRd * k_m) + (M_zEd / M_zRd)
            ratios['bending_z'] = (M_yEd / M_yRd) + (M_zEd / M_zRd * k_m)

            # (EN 1995-1-1, 6.1.2 - 6.1.3)
            N_t0Rd, N_t90Rd = _calc_axial_tension_capacity(factors, material_props, spec, geometry)
            ratios['axial_tension_0'] = N_t0Ed / N_t0Rd
            # ratios['axial_tension_90'] = N_t90Ed / N_t0Rd

            # (EN 1995-1-1, 6.2.3)
            bending_axial_tension_ratio_y, bending_axial_tension_ratio_z = _calc_combined_bending_axial_tension_ratio(M_yRd, M_zRd, N_t0Rd, M_yEd, M_zEd, N_t0Ed, k_m)
            ratios['bending_axial_tension_y'] = bending_axial_tension_ratio_y
            ratios['bending_axial_tension_z'] = bending_axial_tension_ratio_z

            # (EN 1995-1-1, 6.1.4 - 6.1.5)
            N_c0Rd, N_c90Rd, k_cy, k_cz, risk_buckling = _calc_axial_compression_capacity(factors, material_props, spec, geometry, member_length)
            ratios['axial_compression_0'] = N_c0Ed / N_c0Rd
            # ratios['axial_compression_90'] = N_c90Ed / (N_c90Rd * factors['config_and_deformation_factor_k_c90'])

            # (EN 1995-1-1, 6.2.4, 6.3.2)
            bending_axial_compression_ratio_y, bending_axial_compression_ratio_z = _calc_combined_bending_axial_compression_ratio(risk_buckling, M_yRd, M_zRd, N_c0Rd, M_yEd, M_zEd, N_c0Ed, k_cy, k_cz, k_m)
            ratios['bending_axial_compression_y'] = bending_axial_compression_ratio_y
            ratios['bending_axial_compression_z'] = bending_axial_compression_ratio_z

            # (EN 1995-1-1, 6.1.7)
            ratios['shear'] = _calc_shear_ratio(factors, material_props, spec, V_Ed)

            # (EN 1995-1-1, 6.1.8)
            ratios['torsion'] = _calc_torsion_ratio(factors, material_props, spec, T_Ed)

            # (EN 1995-1-1, 7.2)
            w_inst_dl = abs(pynite_member.min_deflection('dz', 'DL'))
            w_inst_ll = abs(pynite_member.min_deflection('dz', 'LL'))
            w_fin = w_inst_dl * (1 + factors['k_def']) + w_inst_ll * (1 + factors['psi_2'] * factors['k_def'])
            ratios['net_deflection'] = w_fin / (member_length / 300) # 300 is a Spanish suggestion - https://cdn.transportes.gob.es/portal-web-transportes/carreteras/normativa_tecnica/21_eurocodigos/AN_UNE-EN-1995-1-1.pdf
        
        # Bearing Check
        if member.node_i in support_node_names:
            beam_bearing_i, brick_bearing_i = _calc_compression_bearing_ratio(factors, material_props, spec, pynite_member, member_end=0)
            ratios['beam_bearing_N'] = beam_bearing_i
            ratios['brick_bearing_N'] = brick_bearing_i

        if member.node_j in support_node_names:
            beam_bearing_j, brick_bearing_j = _calc_compression_bearing_ratio(factors, material_props, spec, pynite_member, member_end=member_length)
            ratios['beam_bearing_S'] = beam_bearing_j
            ratios['brick_bearing_S'] = brick_bearing_j

        member_evaluations.append({
            'member_name': member.name,
            'material_id': member.spec.material_id,
            **ratios
            })
        
    part_evaluations = pd.DataFrame(member_evaluations)
    
    return part_evaluations.drop(columns=['material_id']).set_index('member_name')


def group_stresses_by_member(raw_part_evaluations, join_planks=False):
    part_evaluations = raw_part_evaluations.copy()
    split_names = part_evaluations.index.str.split('_')
    part_evaluations['base_name'] = split_names.str[0]
    part_evaluations['part_num'] = pd.to_numeric(split_names.str[1], errors='coerce').astype('Int8')

    if join_planks:
        part_evaluations['member_group'] = part_evaluations['base_name'].str.replace(r'\d+', '', regex=True)
    else:
        for base_name in part_evaluations[part_evaluations['part_num'].notna()]['base_name'].unique():
            mask = part_evaluations['base_name'] == base_name
            parts = part_evaluations.loc[mask].sort_values('part_num')
            part_nums = parts['part_num'].values
            gaps = np.where(np.diff(part_nums) > 1)[0]

            group_id = 0
            current_group = f"{base_name}_{chr(ord('A') + group_id)}"
            for idx, row_idx in enumerate(parts.index):
                part_evaluations.at[row_idx, 'member_group'] = current_group
                if idx in gaps:
                    group_id += 1
                    current_group = f"{base_name}_{chr(ord('A') + group_id)}"

    part_evaluations['member_group'] = part_evaluations['member_group'].fillna(part_evaluations['base_name'])
    num_cols = part_evaluations.select_dtypes(include=['number']).columns.tolist()
    member_evaluations = part_evaluations[num_cols + ['member_group']].groupby('member_group').mean()

    return member_evaluations.drop(columns=['part_num'])


def render(frame, deformed_scale=100, opacity=0.25, combo_name='ULS_Strength') -> None:
    def _set_wall_opacity(plotter, opacity=0.25):
        for actor in plotter.renderer.actors.values():
            if (hasattr(actor, 'mapper') and
                hasattr(actor.mapper, 'dataset') and
                actor.mapper.dataset.n_faces_strict > 0):
                actor.prop.opacity = opacity

    rndr = Renderer(frame)
    rndr.combo_name = combo_name
    rndr.annotation_size = 5
    rndr.render_loads = False
    rndr.render_nodes = False
    rndr.deformed_shape = True
    rndr.deformed_scale = deformed_scale
    rndr.post_update_callbacks.append(lambda plotter: _set_wall_opacity(plotter, opacity=opacity))
    rndr.render_model()


if __name__ == '__main__':
    # Units are mm, N, and MPa (N/mm²)
    hyperparams = {
        'east_joists' : MemberSpec('c24_80x160', quantity=1, padding=181),
        'west_joists' : MemberSpec('c24_80x160', quantity=1, padding=161),
        'tail_joists' : MemberSpec('c24_45x90', quantity=1, padding=643),
        'trimmerW' : MemberSpec('c24_80x160', quantity=1),
        'trimmerE' : MemberSpec('c24_80x160', quantity=1),
        'header' : MemberSpec('c24_45x90', quantity=1),
        'planks' : MemberSpec('c18_200x25'),
        }

    frame, nodes, members = create_model(hyperparams, walls=True, exclude_loads_under_big_beam=True, free_tail=False)
    member_evaluations = evaluate_stresses(frame, members)
    total_cost, cuts = calculate_purchase_quantity(frame, members)
    render(frame, deformed_scale=100, opacity=0.2, combo_name=ULS_COMBO)
