from dataclasses import dataclass, replace
from enum import member
from re import L
from typing import List, Dict, Optional, Tuple, Literal
from Pynite import FEModel3D
from Pynite.Rendering import Renderer
import numpy as np
import pandas as pd
from collections import defaultdict
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args
import warnings
import bisect

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor


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
    def beam_length(self):
        return self.room_width + self.wall_beam_contact_depth
    
    @property
    def opening_z(self):
        return self.beam_length - self.opening_width - self.wall_beam_contact_depth/2


def prep_data():
    # Units are mm, N, and MPa (N/mm²)
    # INPUT PARAMS
    input_params = pd.read_csv('data/design_parameters.csv').iloc[0].to_dict()
    input_params = DesignParameters(**input_params)

    # MATERIAL STRENGTH
    material_str = pd.read_csv('data/material_strengths.csv').set_index('material').to_dict(orient='index')

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
    material_catalog['id'] = (material_catalog['material'] + '_' + material_catalog['base'].astype(str) + 'x' + material_catalog['height'].astype(str))

    ## Add double beams
    doubled_beams = material_catalog[(material_catalog['type'] == 'beam') & (material_catalog['material'] != 'steel')].copy()
    doubled_beams['base'] = doubled_beams['base'] * 2
    doubled_beams['type'] = 'double'
    doubled_beams['source'] = doubled_beams['id']
    material_catalog = pd.concat([material_catalog, doubled_beams], ignore_index=True)

    rows_to_drop = ((material_catalog['type'] == 'beam') | (material_catalog['type'] == 'double')) & (~material_catalog['base'].isin(connectors['base']))
    material_catalog = material_catalog[~rows_to_drop]

    ## Standardize floor materials to 200mm width
    is_floor = material_catalog['type'] == 'floor'
    material_catalog.loc[is_floor, 'cost_unit'] = material_catalog.loc[is_floor, 'cost_unit'] * (200 / material_catalog.loc[is_floor, 'base'])
    material_catalog.loc[is_floor, 'base'] = 200

    material_catalog['id'] = (material_catalog['material'] + '_' + material_catalog['base'].astype(str) + 'x' + material_catalog['height'].astype(str))

    return input_params, material_str, material_catalog, connectors

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
    def from_i_beam(cls, height: float, base: float, flange_thickness: float, web_thickness: float) -> 'CrossSectionProperties':
        A_flanges = 2 * base * flange_thickness
        A_web = (height - 2 * flange_thickness) * web_thickness
        A = A_flanges + A_web
        
        Iz_flanges = 2 * (base * flange_thickness**3 / 12 + 
                         base * flange_thickness * ((height - flange_thickness)/2)**2)
        web_height = height - 2 * flange_thickness
        Iz_web = web_thickness * web_height**3 / 12
        Iz = Iz_flanges + Iz_web
        
        Iy_flanges = 2 * (flange_thickness * base**3 / 12)
        Iy_web = web_height * web_thickness**3 / 12
        Iy = Iy_flanges + Iy_web
        
        J = (2 * base * flange_thickness**3 + web_height * web_thickness**3) / 3
        return cls(A=A, Iy=Iy, Iz=Iz, J=J)

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
        elif self.shape.startswith('IP'):
            return CrossSectionProperties.from_i_beam(self.height, self.base, self._catalog_data['flange_thickness'], self._catalog_data['web_thickness'])
        else:
            raise ValueError(f"Unknown shape: {self.shape}")
    
    def create_section(self, frame: FEModel3D):
        if self.section_name not in frame.sections:
            geom = self.get_geometry()
            frame.add_section(self.section_name, geom.A, geom.Iy, geom.Iz, geom.J)

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
    

def calculate_nodes_and_members(
    east_joists: MemberSpec,
    west_joists: MemberSpec,
    tail_joists: MemberSpec,
    trimmers: MemberSpec,
    header: MemberSpec,
    planks: MemberSpec) -> Tuple[List[NodeLocation], List[Member]]:

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
    opening_x_end = INPUT_PARAMS.opening_x_start + INPUT_PARAMS.opening_length
    trimmer_E_x = INPUT_PARAMS.opening_x_start - (trimmers.base / 2)
    trimmer_W_x = opening_x_end + (trimmers.base / 2)
    header_z = INPUT_PARAMS.opening_z - header.base/2
    
    # Beam centerline positions
    beam_positions = {}
    if east_joists.quantity > 0:
        clear_start = east_joists.padding
        clear_end = trimmer_E_x - trimmers.base / 2
        x_positions = _calculate_evenly_spaced_positions(east_joists.quantity, clear_start, clear_end, east_joists.base, 'start_aligned')
        beam_positions['east'] = [(f'east{i}', x) for i, x in enumerate(x_positions)]
    
    if tail_joists.quantity > 0:
        clear_start = INPUT_PARAMS.opening_x_start + tail_joists.padding
        clear_end = opening_x_end - tail_joists.padding
        x_positions = _calculate_evenly_spaced_positions(tail_joists.quantity, clear_start, clear_end, tail_joists.base, 'centered')
        beam_positions['tail'] = [(f'tail{i}', x) for i, x in enumerate(x_positions)]
    
    if west_joists.quantity > 0:
        clear_start = trimmer_W_x + trimmers.base / 2
        clear_end = INPUT_PARAMS.room_length - west_joists.padding
        x_positions = _calculate_evenly_spaced_positions(west_joists.quantity, clear_start, clear_end, west_joists.base, 'end_aligned')
        beam_positions['west'] = [(f'west{i}', x) for i, x in enumerate(x_positions)]
    
    beam_positions['trimmer'] = [('trimmerE', trimmer_E_x), ('trimmerW', trimmer_W_x)]

    # Plank centerline positions
    clear_start = INPUT_PARAMS.wall_beam_contact_depth / 2
    clear_end = INPUT_PARAMS.beam_length - INPUT_PARAMS.wall_beam_contact_depth / 2
    planks.quantity = int((clear_end - clear_start) // planks.base)
    z_positions = _calculate_evenly_spaced_positions(planks.quantity, clear_start-planks.base, clear_end+planks.base, planks.base, 'centered')
    plank_positions = [(f'p{i}', z) for i, z in enumerate(z_positions)]
    
    # Beam nodes and member locations
    nodes = []
    members = []
    for group_name, group_positions in beam_positions.items():
        for beam_name, x in group_positions:
            nodes.append(NodeLocation(f'{beam_name}_S', x, INPUT_PARAMS.floor_y, 0))
            if group_name == 'tail': # Tails connect to header (header_z), not to wall (beam_length)
                nodes.append(NodeLocation(f'{beam_name}_header', x, INPUT_PARAMS.floor_y, header_z))
                members.append(Member(name=beam_name, node_i=f'{beam_name}_header', node_j=f'{beam_name}_S', spec=east_joists))
            else:
                nodes.append(NodeLocation(f'{beam_name}_N', x, INPUT_PARAMS.floor_y, INPUT_PARAMS.beam_length))
                members.append(Member(name=beam_name, node_i=f'{beam_name}_N', node_j=f'{beam_name}_S', spec=east_joists))
    
    nodes.append(NodeLocation('headerE', trimmer_E_x, INPUT_PARAMS.floor_y, header_z))
    nodes.append(NodeLocation('headerW', trimmer_W_x, INPUT_PARAMS.floor_y, header_z))
    members.append(Member(name='header', node_i='headerW', node_j='headerE', spec=header))

    # Corner nodes
    walls = [('E', 0), ('W', INPUT_PARAMS.room_length)]
    for corner_name, x in walls:
        nodes.append(NodeLocation(f'{corner_name}_S', x, INPUT_PARAMS.floor_y, 0))
        nodes.append(NodeLocation(f'{corner_name}_N', x, INPUT_PARAMS.floor_y, INPUT_PARAMS.beam_length))

    # Plank nodes and member locations
    plank_series_dict = {}
    beam_positions['walls'] = walls # So planks extend to walls
    for plank_name, z in plank_positions:
        plank_nodes = []
        for group_positions in beam_positions.values():
            for beam_name, x in group_positions:
                if x > trimmer_E_x and x < trimmer_W_x and z > header_z:
                    continue
                intersection_node = NodeLocation(f'{beam_name}-{plank_name}', x, INPUT_PARAMS.floor_y, z)
                plank_nodes.append(intersection_node)
                nodes.append(intersection_node)
        plank_series_dict[plank_name] = plank_nodes
    
    for plank_name, plank_nodes in plank_series_dict.items():
        plank_nodes.sort(key=lambda node: node.X)
        for i, _ in enumerate(plank_nodes[:-1]):
            if plank_nodes[i].X == trimmer_E_x and plank_nodes[i].Z > header_z:
                continue
            members.append(Member(name=f'{plank_name}_{i}', node_i=plank_nodes[i].name, node_j=plank_nodes[i+1].name, spec=planks))
    
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
    supports['east'] = sorted([n for n in nodes if n.name.startswith('E_')], key=lambda n: n.Z)
    supports['west'] = sorted([n for n in nodes if n.name.startswith('W_')], key=lambda n: n.Z)

    if walls:
        foundation = []
        for support_side, support_nodes in supports.items():
            for node in support_nodes:
                if node.name not in foundation:
                    foundation.append(node.name)
                    frame.add_node(f'{node.name}_foundation', node.X, 0, node.Z)

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
    return connector.mean()
    

def apply_loads(frame: FEModel3D, members: List[Member]) -> None:
    # Dead loads
    for member in members:
        geom = member.spec.get_geometry()
        material = MATERIAL_STRENGTHS[member.spec.material]
        dead_load = -geom.A * material['rho']
        frame.add_member_dist_load(member.name, 'FY', dead_load, dead_load, case=DL_COMBO)

        if member.name.startswith('tail'):
            header = next((m for m in members if m.name.startswith('header')))
            connector = _find_compatible_connector(base=member.spec.base, height=header.spec.height)
            frame.add_member_pt_load(member.name, 'FY', -connector['weight_N'], 0)
        elif member.name.startswith('header'):
            trimmer = next((m for m in members if m.name.startswith('trimmer')))
            connector = _find_compatible_connector(base=member.spec.base, height=trimmer.spec.height)
            frame.add_member_pt_load(member.name, 'FY', -connector['weight_N'], 0, case=DL_COMBO)
            frame.add_member_pt_load(member.name, 'FY', -connector['weight_N'], frame.members[member.name].L(), case=DL_COMBO)

    # Live loads
    plank_members = [m for m in members if m.name.startswith('p')]
    plank_z_values = sorted(set(frame.nodes[m.node_i].Z for m in plank_members))
    standard_spacing = plank_z_values[1] - plank_z_values[0]
    
    tail_planks = []
    for m in plank_members:
        if 'tail' in m.node_i or 'tail' in m.node_j or ('trimmer' in m.node_i and 'trimmer' in m.node_j):
            tail_planks.append(m)
    
    min_z = min(plank_z_values)
    max_z = max(plank_z_values)
    max_tail_plank_z = max(frame.nodes[m.node_i].Z for m in tail_planks)
    for member in plank_members:
        member_z = frame.nodes[member.node_i].Z
        
        if member_z == min_z:
            tributary_width = member.spec.base / 2 + standard_spacing / 2
        elif member_z == max_z:
            tributary_width = member.spec.base / 2 + standard_spacing / 2
        elif member_z == max_tail_plank_z and member in tail_planks:
            tributary_width = INPUT_PARAMS.opening_z - member_z + standard_spacing / 2
        else:
            tributary_width = standard_spacing
        
        live_load = -INPUT_PARAMS.live_load_mpa * tributary_width
        frame.add_member_dist_load(member.name, 'FY', live_load, live_load, case=LL_COMBO)


def create_model(
    east_joists: MemberSpec,
    west_joists: MemberSpec,
    tail_joists: MemberSpec,
    trimmers: MemberSpec,
    header: MemberSpec,
    planks: MemberSpec,
    walls: bool = True) -> Tuple:
    
    nodes, members = calculate_nodes_and_members(east_joists, west_joists, tail_joists, trimmers, header, planks)
    frame = assemble_frame(nodes, members)
    define_supports(frame, nodes, INPUT_PARAMS.wall_thickness, 'brick', walls=walls)
    apply_loads(frame, members)
    
    return frame, nodes, members


def calculate_purchase_quantity(frame: FEModel3D, members: List[Member]):

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
    
    total_cost = 0.0
    materials = defaultdict(list)
    for member in members:
        pynite_member = frame.members[member.name]
        member_length = float(pynite_member.L())

        if member.spec.type == 'double':
            source_id = MATERIAL_CATALOG[MATERIAL_CATALOG['id'] == member.spec.material_id].iloc[0]['source']
            materials[source_id].extend([member_length] * 2)
        else:
            material_id = member.spec.material_id
            materials[material_id].extend([member_length])

        if member.name.startswith('tail'):
            header = next((m for m in members if m.name.startswith('header')))
            connector = _find_compatible_connector(base=member.spec.base, height=header.spec.height)
            total_cost += connector['price_unit']
        elif member.name.startswith('header'):
            trimmer = next((m for m in members if m.name.startswith('trimmer')))
            connector = _find_compatible_connector(base=member.spec.base, height=trimmer.spec.height)
            total_cost += connector['price_unit'] * 2

    all_material_cuts = {}
    for material_id, lengths in materials.items():
        material_specs = MATERIAL_CATALOG[MATERIAL_CATALOG['id'] == material_id].iloc[0]
        num_beams_to_buy, current_material_cuts = _find_optimal_cuts(lengths, material_specs['length'])
        all_material_cuts[material_id] = current_material_cuts
        total_cost += num_beams_to_buy * material_specs['cost_unit']
    
    return total_cost, all_material_cuts


# def evaluate_stresses(frame: FEModel3D, members: List[Member]) -> List[Dict]:
#     frame.analyze()
#     support_node_names = {n.name for n in nodes if n.name.endswith('_N') or n.name.endswith('_S')}
#     evaluations = []
#     for member in members:
#         pynite_member = frame.members[member.name]
#         geom = member.spec.get_geometry()
#         mat_props = MATERIAL_STRENGTHS[member.spec.material]
#         ratios = {}

#         # Deflection
#         deflection = abs(pynite_member.min_deflection('dy', 'Combo 1'))
#         ratios['deflection'] = deflection / (pynite_member.L() / 360)

#         # Bending Stress
#         max_moment = max(abs(pynite_member.max_moment('Mz', 'Combo 1')), abs(pynite_member.min_moment('Mz', 'Combo 1')))
#         bending_stress = (max_moment * (member.spec.height / 2) / geom.Iz)
#         ratios['bending'] = bending_stress / mat_props['f_mk']

#         # Shear Stress
#         max_shear = max(abs(pynite_member.max_shear('Fy', 'Combo 1')), abs(pynite_member.min_shear('Fy', 'Combo 1')))
#         if member.spec.shape == 'rectangular':
#             shear_stress = 1.5 * abs(max_shear) / geom.A
#         else:
#             catalog_data = member.spec._catalog_data
#             web_height = member.spec.height - 2 * catalog_data['flange_thickness']
#             web_thickness = catalog_data['web_thickness']
#             web_area = web_height * web_thickness
#             shear_stress = abs(max_shear) / web_area
#         ratios['shear'] = shear_stress / mat_props['f_vk']

#         # Axial Stress
#         max_axial = max(abs(pynite_member.max_axial('Combo 1')), abs(pynite_member.min_axial('Combo 1')))
#         axial_stress = max_axial / geom.A
#         ratios['axial'] = axial_stress / mat_props['f_c90k']

#         # Torsion Stress
#         max_torsion = max(abs(pynite_member.max_torque('Combo 1')), abs(pynite_member.min_torque('Combo 1')))
#         torsion_stress = max_torsion * (member.spec.height / 2) / geom.J
#         ratios['torsion'] = torsion_stress / mat_props['f_mk']

#         # Bearing/Crushing on walls
#         bearing_area = member.spec.base * INPUT_PARAMS.wall_beam_contact_depth
#         if member.node_i in support_node_names:
#             reaction_force = abs(pynite_member.F('Fy', 0, 'Combo 1'))
#             bearing_stress = reaction_force / bearing_area
                        
#             ratios['wood_contact_i'] = bearing_stress / MATERIAL_STRENGTHS[member.spec.material]['f_c90k']
#             ratios['brick_contact_i'] = bearing_stress / MATERIAL_STRENGTHS['brick']['f_c0k']

#         if member.node_j in support_node_names:
#             reaction_force = abs(pynite_member.F('Fy', pynite_member.L(), 'Combo 1'))
#             bearing_stress = reaction_force / bearing_area
            
#             ratios['wood_contact_j'] = bearing_stress / MATERIAL_STRENGTHS[member.spec.material]['f_c90k']
#             ratios['brick_contact_j'] = bearing_stress / MATERIAL_STRENGTHS['brick']['f_c0k']

#         # Combined Stresses
#         passes_strength = all(r <= 1 for r in ratios.values())
#         grade = ratios.values().sum()

#         evaluations.append({
#             'member_name': member.name,
#             'material_id': member.spec.material_id,
#             'member_passes': passes_strength,
#             'member_grade': grade,
#         })

#     return evaluations


def evaluate_stresses(frame: FEModel3D, members: List[Member], buckling_K: float = 1.0):

    def _r_radius_of_gyration(I, A):
        return (I / A) ** 0.5 if A > 0 else 1e-9

    frame.add_load_combo('ULS_Strength', {'DL': 1.35, 'LL': 1.5})
    frame.analyze()
    support_node_names = {n.name for n in frame.nodes.values() if n.name.endswith('_N') or n.name.endswith('_S')}
    results = []
    for member in members:
        pynite_member = frame.members[member.name]
        L = float(pynite_member.L())
        geom = member.spec.get_geometry()
        mat_props = MATERIAL_STRENGTHS[member.spec.material]
        ratios = {}

        # Deflection (serviceability)
        deflection_dl = abs(pynite_member.min_deflection('dy', 'DL'))
        deflection_ll = abs(pynite_member.min_deflection('dy', 'LL'))
        if member.spec.material_id.startswith('c'):
            deflection_factor = 1.8
        elif member.spec.material_id.startswith('osb'):
            deflection_factor = 2.5
        else:
            deflection_factor = 1.0

        quasi_permanent_factor = 0.8
        deflection = (deflection_dl * deflection_factor) + (deflection_ll * deflection_factor * quasi_permanent_factor)
        ratios['deflection_ratio'] = deflection / (L / 360)

        # Bending stress (normal due to Mz)
        max_Mz = max(abs(pynite_member.max_moment('Mz', 'ULS_Strength')), abs(pynite_member.min_moment('Mz', 'ULS_Strength')))
        # bending normal stress at extreme fiber (use half height)
        sigma_bending = (max_Mz * (member.spec.height / 2.0)) / geom.Iz
        ratios['bending_ratio'] = abs(sigma_bending) / mat_props['f_mk'] if mat_props['f_mk'] > 0 else 0.0

        # Shear stress
        max_shear = max(abs(pynite_member.max_shear('Fy', 'ULS_Strength')), abs(pynite_member.min_shear('Fy', 'ULS_Strength')))
        if member.spec.shape == 'rectangular':
            shear_stress = 1.5 * (max_shear) / geom.A
        else:
            catalog_data = member.spec._catalog_data
            web_height = member.spec.height - 2 * catalog_data.get('flange_thickness', 0)
            web_thickness = catalog_data.get('web_thickness', 1e-6)
            web_area = web_height * web_thickness if web_height > 0 else geom.A
            shear_stress = max_shear / web_area
        ratios['shear_ratio'] = abs(shear_stress) / mat_props['f_vk'] if mat_props['f_vk'] > 0 else 0.0

        # Axial stress
        max_axial = max(abs(pynite_member.max_axial('ULS_Strength')), abs(pynite_member.min_axial('ULS_Strength')))
        # note: min/max_axial returns sign; we'll use sign for buckling direction if compressive
        axial_pos = max([pynite_member.max_axial('ULS_Strength'), pynite_member.min_axial('ULS_Strength')], key=abs)
        axial_stress = axial_pos / geom.A if geom.A > 0 else 0.0
        # compare axial stress with appropriate axial capacity. for wood use f_c90k, steel use f_mk (or f_yield)
        # but f_c90k is compression across grain; for along-grain compression different property may be needed.
        # we'll conservatively use f_c90k for compressive axial in contact with walls and for axial; adapt as needed.
        if axial_pos < 0:
            # compression
            axial_capacity_key = 'f_c90k'
        else:
            axial_capacity_key = 'f_mk' if member.spec.material.startswith('s') else 'f_t0k'
        axial_capacity = mat_props.get(axial_capacity_key, mat_props.get('f_mk', 1e-9))
        ratios['axial_ratio'] = abs(axial_stress) / axial_capacity if axial_capacity > 0 else 0.0

        # Torsion stress
        max_torque = max(abs(pynite_member.max_torque('ULS_Strength')), abs(pynite_member.min_torque('ULS_Strength')))
        torsion_stress = max_torque * (member.spec.height / 2.0) / geom.J if geom.J > 0 else 0.0
        # compare torsion to bending strength for timber or to f_mk for steel (conservative)
        ratios['torsion_ratio'] = abs(torsion_stress) / mat_props['f_mk'] if mat_props['f_mk'] > 0 else 0.0

        # Bearing where supported
        bearing_checks = {}
        bearing_area = member.spec.base * INPUT_PARAMS.wall_beam_contact_depth
        if member.node_i in support_node_names:
            reaction_force_i = abs(pynite_member.shear('Fy', 0, 'ULS_Strength'))
            bearing_stress_i = reaction_force_i / bearing_area if bearing_area > 0 else 0.0
            bearing_checks['wood_contact_i'] = bearing_stress_i / MATERIAL_STRENGTHS[member.spec.material]['f_c90k'] if MATERIAL_STRENGTHS[member.spec.material]['f_c90k'] > 0 else 0.0
            bearing_checks['brick_contact_i'] = bearing_stress_i / MATERIAL_STRENGTHS['brick']['f_c0k'] if MATERIAL_STRENGTHS['brick']['f_c0k'] > 0 else 0.0
        if member.node_j in support_node_names:
            reaction_force_j = abs(pynite_member.shear('Fy', pynite_member.L(), 'ULS_Strength'))
            bearing_stress_j = reaction_force_j / bearing_area if bearing_area > 0 else 0.0
            bearing_checks['wood_contact_j'] = bearing_stress_j / MATERIAL_STRENGTHS[member.spec.material]['f_c90k'] if MATERIAL_STRENGTHS[member.spec.material]['f_c90k'] > 0 else 0.0
            bearing_checks['brick_contact_j'] = bearing_stress_j / MATERIAL_STRENGTHS['brick']['f_c0k'] if MATERIAL_STRENGTHS['brick']['f_c0k'] > 0 else 0.0

        ratios.update(bearing_checks)

        # Buckling (Euler) for compression
        buckling_ratio = 0.0
        if axial_pos < 0:  # compression (sign convention: negative compressive)
            # Use the smaller radius of gyration (more conservative) based on Iy and Iz
            r_y = _r_radius_of_gyration(geom.Iy, geom.A)
            r_z = _r_radius_of_gyration(geom.Iz, geom.A)
            r_min = min(r_y, r_z)
            slenderness = (buckling_K * L) / (r_min + 1e-12)
            # Euler critical load (Pcr) using the relevant I (use the axis with min r -> corresponding I)
            # For simplicity choose I corresponding to r_min
            I_for_Pcr = geom.Iy if r_min == r_y else geom.Iz
            Pcr = (np.pi ** 2) * mat_props['E_0'] * I_for_Pcr / (buckling_K * L) ** 2
            # member axial force (most extreme)
            P_act = abs(axial_pos)
            if Pcr > 0:
                buckling_ratio = P_act / Pcr
            else:
                buckling_ratio = 1e6
        ratios['buckling_ratio'] = buckling_ratio

        # Lateral-torsional buckling heuristic (very simplified)
        # If a beam is unbraced (we don't model bracing), conservative check:
        # if Lb / r_y > 150 (very slender) flag as risk. This is purely heuristic.
        r_y = _r_radius_of_gyration(geom.Iy, geom.A)
        ltb_ratio = (L / (r_y + 1e-12)) / 150.0  # >1 means suspect
        ratios['ltb_ratio'] = ltb_ratio

        # Combined failure check
        member_utilization = 0.0
        member_passes = True

        mat_name = member.spec.material.lower()
        if mat_name.startswith('s'):  # simple rule: steel identifiers start with 's275' in your file
            # combine axial + bending into a normal stress, then von Mises including shear/torsion
            sigma_normal = axial_stress + sigma_bending  # sign included
            tau_equiv = max(abs(shear_stress), abs(torsion_stress))
            sigma_vm = (sigma_normal ** 2 + 3.0 * (tau_equiv ** 2)) ** 0.5
            # use f_mk as yield for steel
            fy = mat_props['f_mk'] if mat_props['f_mk'] > 0 else 1e-9
            member_utilization = sigma_vm / fy
            # keep also shear and deflection and buckling checks:
            member_utilization = max(member_utilization, ratios['shear_ratio'], ratios['deflection_ratio'], ratios['buckling_ratio'], ratios['ltb_ratio'])
            member_passes = member_utilization <= 1.0

        else:
            # Timber / OSB / brittle-like: use quadratic interaction of normalized components
            r_b = ratios['bending_ratio']
            r_s = ratios['shear_ratio']
            r_a = ratios['axial_ratio']
            # torsion treated like bending in capacity terms for simplicity
            r_t = ratios['torsion_ratio']
            # quadratic interaction (conservative but not too extreme)
            interaction = r_b**2 + r_s**2 + r_a**2 + r_t**2
            # also consider deflection and buckling
            member_utilization = max(interaction ** 0.5, ratios['deflection_ratio'], ratios['buckling_ratio'], ratios['ltb_ratio'])
            member_passes = member_utilization <= 1.0

        # Grade: keep the old "sum of ratios" if you like, but provide more meaningful metric
        grade = sum([v for k,v in ratios.items() if isinstance(v, (int, float))])

        results.append({
            'member_name': member.name,
            'material_id': member.spec.material_id,
            'checks': ratios,
            'member_utilization': float(member_utilization),
            'member_passes': bool(member_passes),
            'member_grade_sum': float(grade)
        })

    return results


def render(frame, deformed_scale=100, opacity=0.25) -> None:
    def _set_wall_opacity(plotter, opacity=0.25):
        for actor in plotter.renderer.actors.values():
            if (hasattr(actor, 'mapper') and
                hasattr(actor.mapper, 'dataset') and
                actor.mapper.dataset.n_faces_strict > 0):
                actor.prop.opacity = opacity

    rndr = Renderer(frame)
    rndr.combo_name = 'ULS_Strength'
    rndr.annotation_size = 5
    rndr.render_loads = False
    rndr.deformed_shape = True
    rndr.deformed_scale = deformed_scale
    rndr.post_update_callbacks.append(lambda plotter: _set_wall_opacity(plotter, opacity=opacity))
    rndr.render_model()


if __name__ == '__main__':

    INPUT_PARAMS, MATERIAL_STRENGTHS, MATERIAL_CATALOG, CONNECTORS = prep_data()

    east_joists = MemberSpec('c24_60x120', quantity=1, padding=0)
    tail_joists = MemberSpec('c24_60x120', quantity=1, padding=0)
    west_joists = MemberSpec('c24_60x120', quantity=1, padding=0)
    trimmers = MemberSpec('c24_80x160', quantity=2)
    header = MemberSpec('c24_60x120', quantity=1)
    planks = MemberSpec('c18_200x25')

    DL_COMBO = 'DL'
    LL_COMBO = 'LL'

    frame, nodes, members = create_model(
        east_joists=east_joists,
        west_joists=west_joists,
        tail_joists=tail_joists,
        trimmers=trimmers,
        header=header,
        planks=planks,
    )
    member_evaluations = evaluate_stresses(frame, members)
    print(pd.DataFrame(member_evaluations))
    total_cost, cuts = calculate_purchase_quantity(frame, members)
    render(frame, deformed_scale=100, opacity=0.25)