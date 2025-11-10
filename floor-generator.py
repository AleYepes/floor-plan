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
    elif material_name.startswith('brick'):
        return EUROCODE_FACTORS['brick']
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
    

def apply_loads(frame: FEModel3D, members: List[Member]) -> Tuple[float, float]:
    total_dl_force, total_ll_force = 0.0, 0.0

    # Dead loads
    for member in members:
        geom = member.spec.get_geometry()
        material = MATERIAL_STRENGTHS[member.spec.material]
        dead_load = -geom.A * material['rho']
        frame.add_member_dist_load(member.name, 'FY', dead_load, dead_load, case=DL_COMBO)
        total_dl_force += dead_load * frame.members[member.name].L()

        if member.name.startswith('tail'):
            header = next((m for m in members if m.name.startswith('header')))
            connector = _find_compatible_connector(base=member.spec.base, height=header.spec.height)
            frame.add_member_pt_load(member.name, 'FY', -connector['weight_N'], 0)
            total_dl_force += -connector['weight_N']
        elif member.name.startswith('header'):
            trimmer = next((m for m in members if m.name.startswith('trimmer')))
            connector = _find_compatible_connector(base=member.spec.base, height=trimmer.spec.height)
            frame.add_member_pt_load(member.name, 'FY', -connector['weight_N'], 0, case=DL_COMBO)
            frame.add_member_pt_load(member.name, 'FY', -connector['weight_N'], frame.members[member.name].L(), case=DL_COMBO)
            total_dl_force += -connector['weight_N'] * 2

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
        total_ll_force += live_load * frame.members[member.name].L()

    # Combined load
    frame.add_load_combo(ULS_COMBO, {'DL': 1.35, 'LL': 1.5})

    return total_dl_force, total_ll_force


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
    total_dl_force, total_ll_force = apply_loads(frame, members)
    
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
    
    W_y : section modulus about the strong axis (y)
    W_z : section modulus about the strong axis (z)
    M_y_crit : critical bending moment about the strong axis (y)
    l_ef : effective length of the beam for a uniformly distributed load
    omega_m_crit : critical bending stress calculated according to the classical theory of lateral stability, using 5-percentile stiffness values (EN 1995-1-1, 6.3.3)
    lambda_rel_m : relative slenderness ratio in bending
    k_crit : factor accounting for the effect of lateral buckling
    f_mk : characteristic bending strength
    f_md : design value of bending strength

    M_yRd : bending moment capacity about the strong axis (y)
    M_zRd : bending moment capacity about the weak axis (z)
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
    Chapter 5.2 Axial Compression
    
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


def _calc_shear_capacity(factors, material_props, spec):
    """
    Chapter 6 Cross section subjected to shear (EN 1995-1-1, 6.1.7 - 6.1.8)
    
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

    return V_Rd


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


def evaluate_stresses(frame: FEModel3D, members: List[Member]):
    # Following chapters in https://www.swedishwood.com/siteassets/5-publikationer/pdfer/sw-design-of-timber-structures-vol2-2022.pdf

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

        # ULS Internal Forces
        max_moment_y = pynite_member.max_moment('My', ULS_COMBO)
        min_moment_y = pynite_member.min_moment('My', ULS_COMBO)
        M_yEd = max(abs(max_moment_y), abs(min_moment_y))

        max_moment_z = pynite_member.max_moment('Mz', ULS_COMBO)
        min_moment_z = pynite_member.min_moment('Mz', ULS_COMBO)
        M_zEd = max(abs(max_moment_z), abs(min_moment_z))

        N_t0Ed = pynite_member.max_axial(ULS_COMBO)
        N_c0Ed = pynite_member.min_axial(ULS_COMBO)

        max_shear_y = pynite_member.max_shear('Fy', ULS_COMBO)
        min_shear_y = pynite_member.min_shear('Fy', ULS_COMBO)
        V_Ed = max(abs(max_shear_y), abs(min_shear_y))

        max_torsion = pynite_member.max_torque(ULS_COMBO)
        min_torsion = pynite_member.min_torque(ULS_COMBO)
        tau_tor_d = max(abs(max_torsion), abs(min_torsion))
        
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
            M_yRd, M_zRd = _calc_bending_moment_capacity(factors, material_props, spec, geometry, pynite_member.L())
            ratios['bending_y'] = (M_yEd / M_yRd * k_m) + (M_zEd / M_zRd)
            ratios['bending_z'] = (M_yEd / M_yRd) + (M_zEd / M_zRd * k_m)

            # (EN 1995-1-1, 6.1.2 - 6.1.3)
            N_t0Rd, N_t90Rd = _calc_axial_tension_capacity(factors, material_props, spec, geometry)
            ratios['axial_tension_ratio_0'] = N_t0Ed / N_t0Rd
            # ratios['axial_tension_ratio_90'] = N_t90Ed / N_t0Rd

            # (EN 1995-1-1, 6.2.3)
            bending_axial_tension_ratio_y, bending_axial_tension_ratio_z = _calc_combined_bending_axial_tension_ratio(M_yRd, M_zRd, N_t0Rd, M_yEd, M_zEd, N_t0Ed, k_m)
            ratios['bending_axial_tension_ratio_y'] = bending_axial_tension_ratio_y
            ratios['bending_axial_tension_ratio_z'] = bending_axial_tension_ratio_z

            # (EN 1995-1-1, 6.1.4 - 6.1.5)
            N_c0Rd, N_c90Rd, k_cy, k_cz, risk_buckling = _calc_axial_compression_capacity(factors, material_props, spec, geometry, pynite_member.L())
            ratios['axial_compression_ratio_0'] = N_c0Ed / N_c0Rd
            # ratios['axial_compression_ratio_90'] = N_c90Ed / (N_c90Rd * factors['config_and_deformation_factor_k_c90'])

            # (EN 1995-1-1, 6.2.4, 6.3.2)
            bending_axial_compression_ratio_y, bending_axial_compression_ratio_z = _calc_combined_bending_axial_compression_ratio(risk_buckling, M_yRd, M_zRd, N_c0Rd, M_yEd, M_zEd, N_c0Ed, k_cy, k_cz, k_m)
            ratios['bending_axial_compression_ratio_y'] = bending_axial_compression_ratio_y
            ratios['bending_axial_compression_ratio_z'] = bending_axial_compression_ratio_z

            # (EN 1995-1-1, 6.1.7)
            V_Rd = _calc_shear_capacity(factors, material_props, spec)
            ratios['shear'] = V_Ed / V_Rd

            # (EN 1995-1-1, 6.1.8)
            k_shape = min(1 + (0.15 * spec.height/spec.base), 2)
            f_vd = (material_props['f_vk'] * factors['k_mod']) / factors['gamma_M']
            ratios['torsion'] = tau_tor_d / (k_shape * f_vd)

            # (EN 1995-1-1, 7.2)
            w_inst_dl = abs(pynite_member.min_deflection('dy', 'DL'))
            w_inst_ll = abs(pynite_member.min_deflection('dy', 'LL'))
            w_fin = w_inst_dl * (1 + factors['k_def']) + w_inst_ll * (1 + factors['psi_2'] * factors['k_def'])
            ratios['net_deflection'] = w_fin / (pynite_member.L() / 300) # 300 is a Spanish suggestion - https://cdn.transportes.gob.es/portal-web-transportes/carreteras/normativa_tecnica/21_eurocodigos/AN_UNE-EN-1995-1-1.pdf
        
        # Bearing Check
        brick_props = MATERIAL_STRENGTHS['brick']
        mortar_props = MATERIAL_STRENGTHS['mortar']
        brick_gamma_M = 2.2
        alpha = 0.7
        beta = 0.3
        K = 0.45
        if member.node_i in support_node_names or member.node_j in support_node_names:
            bearing_area = spec.base * INPUT_PARAMS.wall_beam_contact_depth
            reaction_force_i = abs(pynite_member.shear('Fy', 0.0, ULS_COMBO))
            bearing_pressure_i = reaction_force_i / bearing_area

            f_c90d_beam = (material_props['f_c90k'] * factors['k_mod']) / factors['gamma_M']
            f_kd = K * brick_props['f_c0k']**alpha * mortar_props['f_c0k']**beta / brick_gamma_M
            ratios['beam_bearing_i'] = bearing_pressure_i / f_c90d_beam
            ratios['brick_bearing_i'] = bearing_pressure_i / f_kd

        if member.node_j in support_node_names or member.node_j in support_node_names:
            bearing_area = spec.base * INPUT_PARAMS.wall_beam_contact_depth    
            reaction_force_j = abs(pynite_member.shear('Fy', pynite_member.L(), ULS_COMBO))
            bearing_pressure_j = reaction_force_j / bearing_area

            f_c90d_beam = (material_props['f_c90k'] * factors['k_mod']) / factors['gamma_M']
            f_kd = K * brick_props['f_c0k']**alpha * mortar_props['f_c0k']**beta / brick_gamma_M
            ratios['beam_bearing_j'] = bearing_pressure_j / f_c90d_beam
            ratios['brick_bearing_j'] = bearing_pressure_j / f_kd

        member_evaluations.append({
            'member_name': member.name,
            'material_id': member.spec.material_id,
            **ratios
            })
    
    return pd.DataFrame(member_evaluations)


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
    rndr.deformed_shape = True
    rndr.deformed_scale = deformed_scale
    rndr.post_update_callbacks.append(lambda plotter: _set_wall_opacity(plotter, opacity=opacity))
    rndr.render_model()


if __name__ == '__main__':
    # Units are mm, N, and MPa (N/mm²)
    INPUT_PARAMS, MATERIAL_STRENGTHS, MATERIAL_CATALOG, CONNECTORS, EUROCODE_FACTORS = prep_data()

    east_joists = MemberSpec('c24_60x120', quantity=1, padding=0)
    tail_joists = MemberSpec('c24_60x120', quantity=1, padding=0)
    west_joists = MemberSpec('c24_60x120', quantity=1, padding=0)
    trimmers = MemberSpec('c24_60x120', quantity=2)
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
    total_cost, cuts = calculate_purchase_quantity(frame, members)
    render(frame, deformed_scale=100, opacity=0.2, combo_name=ULS_COMBO)