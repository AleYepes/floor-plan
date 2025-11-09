from dataclasses import dataclass, replace
from enum import member
from re import A, L
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
import math

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

    # EUROCODE FACTORS
    eurocode_factors = pd.read_csv('data/eurocode_material_factors.csv').set_index('material_prefix').to_dict(orient='index')

    return input_params, material_str, material_catalog, connectors, eurocode_factors


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
    else:
        raise ValueError("Material not supported")


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

    # Combined load
    frame.add_load_combo(ULS_COMBO, {'DL': 1.35, 'LL': 1.5})


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


def _calc_size_factor_kh(spec):
    if spec.material.startswith('c') and spec.height <= 150:
        size_factor_kh = min((150/spec.height)**0.2, 1.3)
    elif spec.material.startswith('gl') and spec.height <= 600:
        size_factor_kh = min((150/spec.height)**0.1, 1.1)
    else:
        size_factor_kh = 1
    return size_factor_kh


def _calc_instability_factor(factors, material_props, geometry, member_length):
    beta_c = factors['straightness_factor_beta_c']
    
    radius_gyration = math.sqrt(geometry.Iy / geometry.A)
    effective_length = member_length * 0.9
    slenderness_ratio = effective_length / radius_gyration
    relative_slenderness_ratio = (slenderness_ratio / math.pi) * math.sqrt(material_props['f_c0k'] / material_props['E_05'])
    risk_buckling = relative_slenderness_ratio > 0.3

    instability_factor_k = (0.5 * (1 + beta_c * (relative_slenderness_ratio - 0.3) + relative_slenderness_ratio**2))
    instability_factor_c = 1 / (instability_factor_k + math.sqrt(instability_factor_k**2 - relative_slenderness_ratio**2))


def _calc_bending_moment_capacity(factors, material_props, spec, geometry, member_length):
    # Chapter 4 Bending
    k_mod = factors['k_mod']
    gamma_M = factors['gamma_M']
    size_factor_kh = _calc_size_factor_kh(spec)

    f_m_d = (size_factor_kh * material_props['f_mk'] * k_mod) / gamma_M
    section_modulus_W_y = geometry.Iy / (spec.height / 2)
    section_modulus_W_z = geometry.Iz / (spec.base / 2)
    effective_length = member_length * 0.9

    critical_bending_moment_strong_axis = math.pi * math.sqrt(material_props['E_05'] * geometry.Iz * material_props['G_05'] * geometry.J)
    critical_bending_stress = critical_bending_moment_strong_axis / (section_modulus_W_y * effective_length)

    relative_slenderness_ratio = math.sqrt(material_props['f_mk'] / critical_bending_stress)
    if relative_slenderness_ratio <= 0.75:
        lateral_buckling_factor_k_crit = 1.0
    elif relative_slenderness_ratio <= 1.4:
        lateral_buckling_factor_k_crit = 1.56 - 0.75 * relative_slenderness_ratio
    else:
        lateral_buckling_factor_k_crit = 1 / relative_slenderness_ratio ** 2

    bending_moment_capacity_y = f_m_d * section_modulus_W_y * lateral_buckling_factor_k_crit
    bending_moment_capacity_z = f_m_d * section_modulus_W_z

    return bending_moment_capacity_y, bending_moment_capacity_z


def _calc_axial_tension_capacity(factors, material_props, geometry):
    # Chapter 5.1 Axial Tension
    k_mod = factors['k_mod']
    gamma_M = factors['gamma_M']

    f_t0_d = (material_props['f_t0k'] * k_mod) / gamma_M
    f_t90_d = (material_props['f_t90k'] * k_mod) / gamma_M
    axial_tension_capacity_0 = f_t0_d * geometry.A
    axial_tension_capacity_90 = f_t90_d * geometry.A

    return axial_tension_capacity_0, axial_tension_capacity_90


def _get_bending_axial_tension_ratio(spec, M_y_Rd, M_z_Rd, Nt_0_Rd, M_y_Ed, M_z_Ed, Nt_0_Ed):
    # Chapter 7.2 Combined bending and axial tension
    if spec.shape == 'rectangular':
        reduction_factor_km = 0.7
    else:
        reduction_factor_km = 1

    ratio_y = (M_y_Ed / M_y_Rd * reduction_factor_km) + (M_z_Ed / M_z_Rd) + (Nt_0_Ed / Nt_0_Rd)
    ratio_z = (M_y_Ed / M_y_Rd) + (M_z_Ed / M_z_Rd * reduction_factor_km) + (Nt_0_Ed / Nt_0_Rd)

    return ratio_y, ratio_z


def _calc_axial_compression_capacity(factors, material_props, spec, geometry, member_length):
    # Chapter 5.2 Axial Compression
    k_mod = factors['k_mod']
    gamma_M = factors['gamma_M']
    size_factor_kh = _calc_size_factor_kh(spec)
    k_c90 = factors['config_and_deformation_factor_k_c90']
    beta_c = factors['straightness_factor_beta_c']
    f_c90_d = (material_props['f_c90k'] * k_mod) / gamma_M

    radius_gyration = math.sqrt(geometry.Iy / geometry.A)
    effective_length = member_length * 0.9
    slenderness_ratio = effective_length / radius_gyration
    relative_slenderness_ratio = (slenderness_ratio / math.pi) * math.sqrt(material_props['f_c0k'] / material_props['E_05'])
    risk_buckling = relative_slenderness_ratio > 0.3

    instability_factor_k = (0.5 * (1 + beta_c * (relative_slenderness_ratio - 0.3) + relative_slenderness_ratio**2))
    instability_factor_c = 1 / (instability_factor_k + math.sqrt(instability_factor_k**2 - relative_slenderness_ratio**2))

    f_c0_d = (size_factor_kh * material_props['f_c0k'] * k_mod) / gamma_M
    axial_compression_capacity_0 = f_c0_d * geometry.A * instability_factor_c

    effective_contact_area = spec.base * INPUT_PARAMS.wall_beam_contact_depth
    axial_compression_capacity_90 = k_c90 * f_c90_d * effective_contact_area

    return axial_compression_capacity_0, axial_compression_capacity_90, risk_buckling


def _get_bending_axial_compression_ratio(spec, risk_buckling, M_y_Rd, M_z_Rd, Nc_0_Rd, M_y_Ed, M_z_Ed, Nc_0_Ed):
    # Chapter 7.3 Combined bending and axial compression
    if spec.shape == 'rectangular':
        reduction_factor_km = 0.7
    else:
        reduction_factor_km = 1

    if risk_buckling:
        ratio_y = (M_y_Ed / M_y_Rd * reduction_factor_km) + (M_z_Ed / M_z_Rd) + (Nc_0_Ed / Nc_0_Rd)
        ratio_z = (M_y_Ed / M_y_Rd) + (M_z_Ed / M_z_Rd * reduction_factor_km) + (Nc_0_Ed / Nc_0_Rd)
    else:
        ratio_y = (M_y_Ed / M_y_Rd * reduction_factor_km) + (M_z_Ed / M_z_Rd) + (Nc_0_Ed / Nc_0_Rd)**2
        ratio_z = (M_y_Ed / M_y_Rd) + (M_z_Ed / M_z_Rd * reduction_factor_km) + (Nc_0_Ed / Nc_0_Rd)**2


    return ratio_y, ratio_z


def _calc_shear_capacity(factors, material_props, spec, geometry):
    # Chapter 6 Cross section subjected to shear
    k_mod = factors['k_mod']
    gamma_M = factors['gamma_M']
    k_cr = factors['k_cr']

    f_v_d = (material_props['f_vk'] * k_mod) / gamma_M
    shear_capacity = f_v_d * geometry.A / 1.5

    if spec.material.startswith('c') or spec.material.startswith('gl'):
        effective_width = spec.base * k_cr

    return shear_capacity


def evaluate_stresses(frame: FEModel3D, members: List[Member]):
    # Using equations and values detailed in https://www.swedishwood.com/siteassets/5-publikationer/pdfer/sw-design-of-timber-structures-vol2-2022.pdf

    frame.add_load_combo(ULS_COMBO, {'DL': 1.35, 'LL': 1.5})
    frame.analyze(check_stability=False)

    psi_0_storage = 1
    psi_1_storage = 0.9
    psi_2_storage = 0.8
    buckling_K = 1.0
    support_node_names = {node.name for node in frame.nodes.values() if node.name.endswith(('_N', '_S'))}
    
    evaluation_results = []
    for member in members:
        ratios = {}

        pynite_member = frame.members[member.name]
        spec = member.spec
        geometry = spec.get_geometry()
        material_props = MATERIAL_STRENGTHS[spec.material]
        factors = _get_eurocode_factors(spec.material)

        # ULS Internal Forces
        max_moment_y = pynite_member.max_moment('My', ULS_COMBO)
        min_moment_y = pynite_member.min_moment('My', ULS_COMBO)
        M_y_Ed = max(abs(max_moment_y), abs(min_moment_y))

        max_moment_z = pynite_member.max_moment('Mz', ULS_COMBO)
        min_moment_z = pynite_member.min_moment('Mz', ULS_COMBO)
        M_z_Ed = max(abs(max_moment_z), abs(min_moment_z))

        Nt_0_Ed = pynite_member.max_axial(ULS_COMBO)
        Nc_0_Ed = pynite_member.min_axial(ULS_COMBO)

        max_shear_y = pynite_member.max_shear('Fy', ULS_COMBO)
        min_shear_y = pynite_member.min_shear('Fy', ULS_COMBO)
        V_Ed = max(abs(max_shear_y), abs(min_shear_y))
        
        if factors.material_type.startswith('s'):
            # gamma_M = factors['gamma_M']
            # yield_strength = material_props['f_mk']
            
            # plastic_section_modulus_z = geometry.Iz / (spec.height / 2) * 1.12 if spec.height > 0 else 1e-9 # Approximation for I-sections
            
            # bending_resistance_d = plastic_section_modulus_z * yield_strength / gamma_M if gamma_M > 0 else 1e9
            # axial_resistance_d = geometry.A * yield_strength / gamma_M if gamma_M > 0 else 1e9
            # shear_resistance_d = (geometry.A * 0.58) * (yield_strength / np.sqrt(3)) / gamma_M if gamma_M > 0 else 1e9
            
            # ratios['shear'] = V_Ed / shear_resistance_d
            # ratios['bending'] = M_y_Ed / bending_resistance_d
            # ratios['axial'] = abs(Nt_0_Ed) / axial_resistance_d
            # ratios['interaction'] = (abs(Nt_0_Ed) / axial_resistance_d) + (M_y_Ed / bending_resistance_d)
        else:          
            # Combined bending and axial tension check
            M_y_Rd, M_z_Rd = _calc_bending_moment_capacity(factors, material_props, spec, geometry, pynite_member.L())
            Nt_0_Rd, Nt_Rd_90 = _calc_axial_tension_capacity(factors, material_props, geometry)

            bending_axial_tension_y, bending_axial_tension_z = _get_bending_axial_tension_ratio(spec, M_y_Rd, M_z_Rd, Nt_0_Rd, M_y_Ed, M_z_Ed, Nt_0_Ed)
            ratios['bending_axial_tension_y'] = bending_axial_tension_y
            ratios['bending_axial_tension_z'] = bending_axial_tension_z

            # Combined bending and axial compression check
            Nc_0_Rd, Nc_Rd_90, risk_buckling = _calc_axial_compression_capacity(factors, material_props, spec, geometry, pynite_member.L())
            bending_axial_compression_y, bending_axial_compression_z = _get_bending_axial_compression_ratio(spec, risk_buckling, M_y_Rd, M_z_Rd, Nc_0_Rd, M_y_Ed, M_z_Ed, Nc_0_Ed)


            shear_capacity = _calc_shear_capacity(factors, material_props, spec, geometry)

            # SLS deflection check
            w_inst_dl = abs(pynite_member.min_deflection('dy', 'DL'))
            w_inst_ll = abs(pynite_member.min_deflection('dy', 'LL'))
            w_fin = w_inst_dl * (1 + factors.get('k_def', 0)) + w_inst_ll * (1 + psi_2_storage * factors.get('k_def', 0))
            ratios['net_deflection'] = w_fin / (pynite_member.L()/150)
        
        # Bearing Check
        if member.node_i in support_node_names or member.node_j in support_node_names:
            bearing_area = spec.base * INPUT_PARAMS.wall_beam_contact_depth
            if bearing_area > 0:
                reaction_force_i = abs(pynite_member.shear('Fy', 0.0, ULS_COMBO))
                bearing_stress_i = reaction_force_i / bearing_area
                
                reaction_force_j = abs(pynite_member.shear('Fy', pynite_member.L(), ULS_COMBO))
                bearing_stress_j = reaction_force_j / bearing_area
                
                bearing_stress_d = max(bearing_stress_i, bearing_stress_j)
                
                if material_type == 'wood':
                    f_compression_perp_d = (material_props['f_c90k'] * k_mod) / gamma_M if gamma_M > 0 else 1e9
                    ratios['bearing'] = bearing_stress_d / f_compression_perp_d if f_compression_perp_d > 0 else 0
            
        valid_ratios = [val for val in ratios.values() if not np.isnan(val)]
        member_utilization = max(valid_ratios) if valid_ratios else 0.0

        evaluation_results.append({
            'member_name': member.name,
            'material_id': spec.material_id,
            'ratios': ratios,
            'member_utilization': member_utilization
        })

    all_ratio_keys = {key for res in evaluation_results for key in res['ratios'].keys()}
    rows = []
    for res in evaluation_results:
        row_data = {
            'member_name': res['member_name'],
            'material_id': res['material_id'],
            'member_utilization': res['member_utilization']
        }
        for key in all_ratio_keys:
            row_data[f'ratio_{key}'] = res['ratios'].get(key)
        rows.append(row_data)
    results_df = pd.DataFrame(rows)
    
    return evaluation_results, results_df


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

    INPUT_PARAMS, MATERIAL_STRENGTHS, MATERIAL_CATALOG, CONNECTORS, EUROCODE_FACTORS = prep_data()

    east_joists = MemberSpec('c24_60x120', quantity=1, padding=0)
    tail_joists = MemberSpec('c24_60x120', quantity=1, padding=0)
    west_joists = MemberSpec('c24_60x120', quantity=1, padding=0)
    trimmers = MemberSpec('c24_80x160', quantity=2)
    header = MemberSpec('c24_60x120', quantity=1)
    planks = MemberSpec('c18_200x25')

    DL_COMBO = 'DL'
    LL_COMBO = 'LL'
    ULS_COMBO = 'ULS_Strength'

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