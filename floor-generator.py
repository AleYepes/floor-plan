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


def evaluate_stresses(frame: FEModel3D, members: List[Member]):
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

        # Deflection
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

        # Bending stress
        max_Mz = max(abs(pynite_member.max_moment('Mz', 'ULS_Strength')), abs(pynite_member.min_moment('Mz', 'ULS_Strength')))
        sigma_bending = (max_Mz * (member.spec.height / 2.0)) / geom.Iz
        ratios['bending_ratio'] = abs(sigma_bending) / mat_props['f_mk']

        # Shear stress
        max_shear = max(abs(pynite_member.max_shear('Fy', 'ULS_Strength')), abs(pynite_member.min_shear('Fy', 'ULS_Strength')))
        if member.spec.shape == 'rectangular':
            shear_stress = 1.5 * (max_shear) / geom.A
        else:
            catalog_data = member.spec._catalog_data
            web_height = member.spec.height - 2 * catalog_data.get('flange_thickness', 0)
            web_thickness = catalog_data.get('web_thickness', 1e-6)
            web_area = web_height * web_thickness
            shear_stress = max_shear / web_area
        ratios['shear_ratio'] = abs(shear_stress) / mat_props['f_vk']

        # Axial stress
        axial_pos = max([pynite_member.max_axial('ULS_Strength'), pynite_member.min_axial('ULS_Strength')], key=abs)
        axial_stress = axial_pos / geom.A
        if axial_pos < 0:
            axial_capacity_key = 'f_c0k'
        else:
            axial_capacity_key = 'f_mk' if member.spec.material.startswith('s') else 'f_t0k'
        axial_capacity = mat_props.get(axial_capacity_key, mat_props.get('f_mk', 1e-9))
        ratios['axial_ratio'] = abs(axial_stress) / axial_capacity

        # Torsion stress
        max_torque = max(abs(pynite_member.max_torque('ULS_Strength')), abs(pynite_member.min_torque('ULS_Strength')))
        torsion_stress = max_torque * (member.spec.height / 2.0) / geom.J
        ratios['torsion_ratio'] = abs(torsion_stress) / mat_props['f_mk']

        # Bearing where supported
        bearing_checks = {}
        bearing_area = member.spec.base * INPUT_PARAMS.wall_beam_contact_depth
        if member.node_i in support_node_names:
            reaction_force_i = abs(pynite_member.shear('Fy', 0, 'ULS_Strength'))
            bearing_stress_i = reaction_force_i / bearing_area
            bearing_checks['wood_contact_i'] = bearing_stress_i / MATERIAL_STRENGTHS[member.spec.material]['f_c90k']
            bearing_checks['brick_contact_i'] = bearing_stress_i / MATERIAL_STRENGTHS['brick']['f_c0k']
        if member.node_j in support_node_names:
            reaction_force_j = abs(pynite_member.shear('Fy', pynite_member.L(), 'ULS_Strength'))
            bearing_stress_j = reaction_force_j / bearing_area
            bearing_checks['wood_contact_j'] = bearing_stress_j / MATERIAL_STRENGTHS[member.spec.material]['f_c90k']
            bearing_checks['brick_contact_j'] = bearing_stress_j / MATERIAL_STRENGTHS['brick']['f_c0k']

        ratios.update(bearing_checks)

        # Buckling (Euler) for compression
        buckling_K = 1.0
        buckling_ratio = 0.0
        if axial_pos < 0:  # compression (sign convention: negative compressive)
            r_y = (geom.Iy / geom.A) ** 0.5
            r_z = (geom.Iz / geom.A) ** 0.5
            r_min = min(r_y, r_z)
            I_for_Pcr = geom.Iy if r_min == r_y else geom.Iz
            Pcr = (np.pi ** 2) * mat_props['E_0'] * I_for_Pcr / (buckling_K * L) ** 2
            P_act = abs(axial_pos)
            if Pcr > 0:
                buckling_ratio = P_act / Pcr
            else:
                buckling_ratio = 1e6
        ratios['buckling_ratio'] = buckling_ratio

        # Combined failure check
        mat_name = member.spec.material.lower()
        if mat_name.startswith('s'):
            # combine axial + bending into a normal stress, then von Mises including shear/torsion
            sigma_normal = axial_stress + sigma_bending 
            tau_equiv = max(abs(shear_stress), abs(torsion_stress))
            sigma_vm = (sigma_normal ** 2 + 3.0 * (tau_equiv ** 2)) ** 0.5
            fy = mat_props['f_mk'] if mat_props['f_mk'] > 0 else 1e-9
            member_utilization = sigma_vm / fy
            member_utilization = max(member_utilization, ratios['shear_ratio'], ratios['deflection_ratio'], ratios['buckling_ratio'])

        else:
            # Timber / OSB / brittle-like: use quadratic interaction of normalized components
            interaction = ratios['bending_ratio']**2 + ratios['shear_ratio']**2 + ratios['axial_ratio']**2 + ratios['torsion_ratio']**2
            member_utilization = max(interaction ** 0.5, ratios['deflection_ratio'], ratios['buckling_ratio'])

        results.append({
            'member_name': member.name,
            'material_id': member.spec.material_id,
            'ratios': ratios,
            'member_utilization': float(member_utilization),
        })

    return results


def evaluate_stresses(frame: FEModel3D, members: List[Member], buckling_K: float = 1.0):

    def _get_eurocode_factors(material_name: str):
        if material_name.startswith('s'):
            return {'gamma_M': 1.0, 'k_mod': 1.0, 'k_def': 0.0, 'k_cr': 1.0, 'k_c90': 1.0, 'material_type': 'steel'}
        elif material_name.startswith('osb'):
             return {'gamma_M': 1.2, 'k_mod': 0.7, 'k_def': 2.25, 'k_cr': 1.0, 'k_c90': 1.0, 'material_type': 'wood'}
        elif material_name.startswith('c'):
            return {'gamma_M': 1.3, 'k_mod': 0.8, 'k_def': 0.6, 'k_cr': 0.67, 'k_c90': 1.5, 'material_type': 'wood'}
        elif material_name == 'brick':
             return {'gamma_M': 2.5, 'k_mod': 1.0, 'k_def': 0.0, 'k_cr': 1.0, 'k_c90': 1.0, 'material_type': 'masonry'}
        else:
            return {'gamma_M': 1.0, 'k_mod': 1.0, 'k_def': 0.0, 'k_cr': 1.0, 'k_c90': 1.0, 'material_type': 'generic'}

    frame.add_load_combo('ULS_Strength', {'DL': 1.35, 'LL': 1.5})
    frame.analyze()
    support_node_names = {n.name for n in frame.nodes.values() if n.name.endswith('_N') or n.name.endswith('_S')}
    results = []
    
    for member in members:
        pynite_member = frame.members[member.name]
        L = float(pynite_member.L())
        geom = member.spec.get_geometry()
        mat_props = MATERIAL_STRENGTHS[member.spec.material]
        factors = _get_eurocode_factors(member.spec.material)
        
        # Design strengths
        gamma_M = factors['gamma_M']
        k_mod = factors['k_mod']
        
        f_md = (mat_props['f_mk'] * k_mod) / gamma_M
        f_vd = (mat_props['f_vk'] * k_mod) / gamma_M
        f_c0d = (mat_props.get('f_c0k', 0) * k_mod) / gamma_M
        f_t0d = (mat_props.get('f_t0k', 0) * k_mod) / gamma_M
        f_c90d = (mat_props.get('f_c90k', 0) * k_mod) / gamma_M
        
        ratios = {}

        # Deflection (EC5 7.2)
        # Using quasi-permanent combination for final deflection
        psi_2 = 0.3  # standard for live loads in storage
        k_def = factors['k_def']
        
        u_inst_dl = abs(pynite_member.min_deflection('dy', 'DL'))
        u_inst_ll = abs(pynite_member.min_deflection('dy', 'LL'))
        
        u_fin = u_inst_dl * (1 + k_def) + u_inst_ll * (1 + psi_2 * k_def)
        ratios['deflection_ratio'] = u_fin / (L / 360)

        # Retrieve Internal forces (ULS)
        M_Ed = max(abs(pynite_member.max_moment('Mz', 'ULS_Strength')), abs(pynite_member.min_moment('Mz', 'ULS_Strength')))
        V_Ed = max(abs(pynite_member.max_shear('Fy', 'ULS_Strength')), abs(pynite_member.min_shear('Fy', 'ULS_Strength')))
        N_Ed_raw = [pynite_member.max_axial('ULS_Strength'), pynite_member.min_axial('ULS_Strength')]
        N_Ed = max(N_Ed_raw, key=abs) # Max axial magnitude
        T_Ed = max(abs(pynite_member.max_torque('ULS_Strength')), abs(pynite_member.min_torque('ULS_Strength')))

        if factors['material_type'] == 'wood':
            # Timber Member Design (EC5)
            
            # Shear (with k_cr correction for cracks)
            b_eff = member.spec.base * factors['k_cr']
            shear_area = b_eff * member.spec.height
            tau_d = 1.5 * V_Ed / shear_area if member.spec.shape == 'rectangular' else V_Ed / shear_area
            ratios['shear_ratio'] = abs(tau_d) / f_vd

            # Bending stress
            sigma_m_d = (M_Ed * (member.spec.height / 2.0)) / geom.Iz
            ratios['bending_ratio'] = sigma_m_d / f_md

            # Axial stress
            sigma_c0_d = abs(N_Ed) / geom.A
            if N_Ed < 0: # Compression
                ratios['axial_ratio'] = sigma_c0_d / f_c0d
                P_crit = (np.pi**2 * mat_props['E_0'] * min(geom.Iy, geom.Iz)) / (L * buckling_K)**2
                ratios['buckling_ratio'] = abs(N_Ed) / (P_crit / gamma_M) if P_crit > 0 else 10.0
            else: # Tension
                ratios['axial_ratio'] = sigma_c0_d / f_t0d
                ratios['buckling_ratio'] = 0.0

            # Lateral Torsional Buckling (LTB) simplified
            sigma_mcrit = (np.pi * (mat_props['E_0'] * geom.Iz * mat_props['G'] * geom.J)**0.5) / (L * geom.Iz/(member.spec.height/2))
            rel_slenderness_m = (mat_props['f_mk'] / sigma_mcrit)**0.5
            
            if rel_slenderness_m <= 0.75:
                k_crit = 1.0
            else:
                # Simplified curve for solid timber
                k_crit = 1.56 - 0.75 * rel_slenderness_m
                k_crit = max(min(k_crit, 1.0), 0.1)
            
            ratios['ltb_ratio'] = sigma_m_d / (k_crit * f_md)
            
            # Combined interaction (EC5 6.2.3/6.2.4 linear)
            if N_Ed < 0: # Compression + Bending
                interaction = (sigma_c0_d / f_c0d) + (sigma_m_d / f_md) + ratios.get('ltb_ratio', 0)*0 # Include LTB implicitly in bending term if needed
                interaction_val = max(interaction, ratios['ltb_ratio']) # Take worse of interaction or LTB
            else: # Tension + Bending
                interaction_val = (sigma_c0_d / f_t0d) + (sigma_m_d / f_md)

        elif factors['material_type'] == 'steel':
            # Steel Member Design (EC3 - simplified)
            M_pl_Rd = f_md * (geom.Iy / (member.spec.height / 2)) # f_md already includes gamma_M=1.0
            N_pl_Rd = geom.A * f_md # actually fy/gammaM0
            V_pl_Rd = (geom.A * 0.5) * (f_md / (3**0.5)) # approximate shear area
            
            # Check high shear (EC3 6.2.10)
            if V_Ed > 0.5 * V_pl_Rd:
                reduction_factor = (1 - (2 * V_Ed / V_pl_Rd - 1)**2)
                M_pl_Rd *= reduction_factor
            
            ratios['shear_ratio'] = V_Ed / V_pl_Rd
            ratios['bending_ratio'] = M_Ed / M_pl_Rd
            ratios['axial_ratio'] = abs(N_Ed) / N_pl_Rd
            
            # Combined interaction (EC3 simplified linear conservatively)
            interaction_val = (abs(N_Ed) / N_pl_Rd) + (M_Ed / M_pl_Rd)
            ratios['buckling_ratio'] = 0.0 # Placeholder for Euler if needed
        else:
             interaction_val = 1e9 # Unknown material

        # Bearing (EC5 6.1.5 with k_c90 enhancement)
        bearing_area = member.spec.base * INPUT_PARAMS.wall_beam_contact_depth
        if member.node_i in support_node_names:
            reaction_force = abs(pynite_member.shear('Fy', 0, 'ULS_Strength'))
            sigma_c90_d = reaction_force / bearing_area
            ratios['wood_contact_i'] = sigma_c90_d / ((factors['k_c90'] * f_c90d))
        if member.node_j in support_node_names:
            reaction_force = abs(pynite_member.shear('Fy', L, 'ULS_Strength'))
            sigma_c90_d = reaction_force / bearing_area
            ratios['wood_contact_j'] = sigma_c90_d / ((factors['k_c90'] * f_c90d))

        # Torsion
        tau_tor_d = (T_Ed * (member.spec.height / 2.0)) / geom.J
        ratios['torsion_ratio'] = abs(tau_tor_d) / f_vd

        # Overall utilization considering all individual ratios and interaction
        member_utilization = max(
            interaction_val,
            ratios['shear_ratio'],
            ratios['deflection_ratio'],
            ratios['torsion_ratio'],
            ratios.get('buckling_ratio', 0),
            ratios.get('wood_contact_i', 0),
            ratios.get('wood_contact_j', 0),
            ratios.get('brick_contact_i', 0),
            ratios.get('brick_contact_j', 0)
        )

        results.append({
            'member_name': member.name,
            'material_id': member.spec.material_id,
            'ratios': ratios,
            'member_utilization': float(member_utilization),
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