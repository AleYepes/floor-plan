from dataclasses import dataclass, replace
from re import L
from typing import List, Dict, Optional, Tuple, Literal
from Pynite import FEModel3D
from Pynite.Rendering import Renderer
import numpy as np
import pandas as pd
from collections import defaultdict
from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real
from skopt.utils import use_named_args
import warnings

import bisect
import heapq


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
    
# Units are mm, N, and MPa (N/mm²)
INPUT_PARAMS = pd.read_csv('data/design_parameters.csv').iloc[0].to_dict()
INPUT_PARAMS = DesignParameters(**INPUT_PARAMS)
MATERIAL_STRENGTHS = pd.read_csv('data/material_strengths.csv').set_index('material').to_dict(orient='index')
MATERIAL_CATALOG = pd.read_csv('data/material_catalog.csv')
MATERIAL_CATALOG['id'] = (MATERIAL_CATALOG['material'] + '_' + MATERIAL_CATALOG['base'].astype(str) + 'x' + MATERIAL_CATALOG['height'].astype(str))

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
        self._catalog_data = MATERIAL_CATALOG[MATERIAL_CATALOG['id'] == self.catalog_id].iloc[0]
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
            G = props['E'] / (2 * (1 + props['nu']))
        else:
            G = props['G']
        frame.add_material(
            mat_name,
            E=props['E'],
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


def define_supports(frame, nodes, wall_thickness, material, walls=False):
    supports = {}
    supports['north'] = sorted([n for n in nodes if n.name.endswith('_N')], key=lambda n: n.X)
    supports['south'] = sorted([n for n in nodes if n.name.endswith('_S')], key=lambda n: n.X)
    supports['east'] = sorted([n for n in nodes if n.name.startswith('E_')], key=lambda n: n.Z)
    supports['west'] = sorted([n for n in nodes if n.name.startswith('W_')], key=lambda n: n.Z)

    if walls:
        floored = []
        for support_side, support_nodes in supports.items():
            for node in support_nodes:
                if node.name not in floored:
                    floored.append(node.name)
                    frame.add_node(f'{node.name}_floor', node.X, 0, node.Z)

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


def apply_loads(frame, members):
    # Dead loads
    for member in members:
        geom = member.spec.get_geometry()
        material = MATERIAL_STRENGTHS[member.spec.material]
        dead_load = -geom.A * material['rho']
        frame.add_member_dist_load(member.name, 'FY', dead_load, dead_load)

    # Live loads
    plank_members = [m for m in members if m.name.startswith('p')]
    plank_z_values = sorted(set(frame.nodes[m.node_i].Z for m in plank_members))
    standard_spacing = plank_z_values[1] - plank_z_values[0]
    
    tail_planks = []
    for m in plank_members:
        if 'tail' in m.node_i or 'tail' in m.node_j:
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
        frame.add_member_dist_load(member.name, 'FY', live_load, live_load)


def calculate_purchase_quantity(required_lengths: List[float], stock_length: float) -> Tuple[int, List[List[float]]]:
    required_lengths.sort(reverse=True)

    bins_cuts: List[List[float]] = []
    bins_remaining: List[float] = []
    for length in required_lengths:
        index = bisect.bisect_left(bins_remaining, length)

        if index < len(bins_remaining):
            old_remaining = bins_remaining.pop(index)
            cuts = bins_cuts.pop(index)
            
            cuts.append(length)
            new_remaining = old_remaining - length
            
            new_index = bisect.bisect_left(bins_remaining, new_remaining)
            bins_remaining.insert(new_index, new_remaining)
            bins_cuts.insert(new_index, cuts)
            
        else:
            new_remaining = stock_length - length
            
            new_index = bisect.bisect_left(bins_remaining, new_remaining)
            bins_remaining.insert(new_index, new_remaining)
            bins_cuts.insert(new_index, [length])

    return len(bins_cuts), bins_cuts


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

    materials = defaultdict(list)
    for member in members:
        pynite_member = frame.members[member.name]
        materials[member.spec.catalog_id].append(pynite_member.L())

    total_cost = 0.0
    for catalog_id, lengths in materials.items():
        catalog_data = MATERIAL_CATALOG[MATERIAL_CATALOG['id'] == catalog_id].iloc[0]
        stock_length = catalog_data['length']
        cost_per_beam = catalog_data['cost_unit']
        
        num_beams_to_buy, cuts = calculate_purchase_quantity(lengths, stock_length)
        total_cost += num_beams_to_buy * cost_per_beam
    
    return frame, nodes, members, total_cost


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



def create_optimization_objective(
    objective_weights: Dict[str, float] = None,
    max_deflection_limit: float = 10.0):
    
    if objective_weights is None:
        objective_weights = {'cost': 0.6, 'deflection': 0.3}
    total_weight = sum(objective_weights.values())
    objective_weights = {k: v/total_weight for k, v in objective_weights.items()}
    
    def objective(params_dict: Dict) -> float:
        try:
            east_joists = MemberSpec(params_dict['east_material'], quantity=params_dict['east_quantity'], padding=params_dict['east_padding'])
            west_joists = MemberSpec(params_dict['west_material'], quantity=params_dict['west_quantity'], padding=params_dict['west_padding'])
            tail_joists = MemberSpec(params_dict['tail_material'], quantity=params_dict['tail_quantity'], padding=params_dict['tail_padding'])
            trimmers = MemberSpec(params_dict['trimmer_material'], quantity=2)
            header = MemberSpec(params_dict['header_material'], quantity=1)
            planks = MemberSpec(params_dict['plank_material'])
            
            frame, _, members, total_cost = create_model(
                east_joists=east_joists,
                west_joists=west_joists,
                tail_joists=tail_joists,
                trimmers=trimmers,
                header=header,
                planks=planks
            )
            frame.analyze(check_statics=True)
            
            results = []
            for member in members:
                pynite_member = frame.members[member.name]
                spec = member.spec
                geom = spec.get_geometry()
                strengths = MATERIAL_STRENGTHS[spec.material]

                # Bending Stress heck
                # Get max moment. For a floor joist, this is typically about the strong axis (Z-axis in Pynite)
                max_moment = max(abs(pynite_member.max_moment('Mz')), abs(pynite_member.min_moment('Mz')))
                section_modulus_z = geom.Iz / (spec.height / 2)
                
                if section_modulus_z > 0:
                    bending_stress = max_moment / section_modulus_z
                else:
                    bending_stress = 0
                    
                bending_strength = strengths['f_mk']
                bending_utilization = bending_stress / bending_strength

                # Shear Stress Check
                # Get max shear force (typically FY for gravity loads)
                max_shear = max(abs(pynite_member.max_shear('Fy')), abs(pynite_member.min_shear('Fy')))

                # Max shear stress for a rectangular section
                if geom.A > 0 and spec.shape == 'rectangular':
                    shear_stress = 1.5 * (max_shear / geom.A)
                else:
                    # Note: Shear stress calculation for I-beams is more complex, often dominated by the web.
                    # This is a simplification. A more detailed analysis would be needed for I-beams.
                    shear_stress = max_shear / geom.A 

                shear_strength = strengths['f_vk']
                shear_utilization = shear_stress / shear_strength if shear_strength > 0 else 0
                
                # Deflection Check
                # Pynite calculates nodal displacements. Getting local member deflection is more involved.
                # A simple proxy is to check the max vertical displacement of the member's nodes.
                node_i_disp = frame.nodes[member.node_i].DY['Combo 1'] # Assuming one load combo
                node_j_disp = frame.nodes[member.node_j].DY['Combo 1']
                max_deflection = min(node_i_disp, node_j_disp) # Downward is negative

                results.append({
                    'Member': member.name,
                    'Type': spec.catalog_id,
                    'Max Moment (N-mm)': max_moment,
                    'Bending Stress (MPa)': bending_stress,
                    'Bending Strength (MPa)': bending_strength,
                    'Bending Utilization': f"{bending_utilization:.2%}",
                    'Max Shear (N)': max_shear,
                    'Shear Stress (MPa)': shear_stress,
                    'Shear Strength (MPa)': shear_strength,
                    'Shear Utilization': f"{shear_utilization:.2%}",
                    'Max Deflection (mm)': max_deflection
                })

            results_df = pd.DataFrame(results)
            print(results_df.to_string())

            # Identify failing members
            failing_bending = results_df[results_df['Bending Utilization'].str.rstrip('%').astype(float) > 100]
            failing_shear = results_df[results_df['Shear Utilization'].str.rstrip('%').astype(float) > 100]

            if not failing_bending.empty:
                print(failing_bending[['Member', 'Type', 'Bending Utilization']])
            if not failing_shear.empty:
                print(failing_shear[['Member', 'Type', 'Shear Utilization']])



            max_deflection = 0
            max_stress_ratio = 0
            for member_name, pynite_member in frame.members.items():
                
                # Deflection
                deflection = abs(member.min_deflection('dy', 'Combo 1'))
                max_deflection = max(max_deflection, deflection)
                
                # Stress check
                max_moment = max(
                    abs(member.max_moment('Mz', 'Combo 1')),
                    abs(member.min_moment('Mz', 'Combo 1'))
                )
                max_shear = max(
                    abs(member.max_shear('Fy', 'Combo 1')),
                    abs(member.min_shear('Fy', 'Combo 1'))
                )
                
                # Calculate stress ratios
                geom = spec.get_geometry()
                mat_props = MATERIAL_STRENGTHS[spec.material]
                
                c = spec.height / 2
                bending_stress = abs(max_moment * c / geom.Iz)
                bending_ratio = bending_stress / mat_props['f_mk']
                
                if spec.shape == 'rectangular':
                    shear_stress = 1.5 * abs(max_shear) / geom.A
                else:  # I-beam
                    catalog_data = MATERIAL_CATALOG[MATERIAL_CATALOG['id'] == spec.catalog_id].iloc[0]
                    web_area = (spec.height - 2*catalog_data['flange_thickness']) * catalog_data['web_thickness']
                    shear_stress = abs(max_shear) / web_area
                
                shear_ratio = shear_stress / mat_props['f_vk']
                
                max_stress_ratio = max(max_stress_ratio, bending_ratio, shear_ratio)
            
            if max_deflection > max_deflection_limit:
                # Penalize configurations that exceed deflection limit
                penalty = 1000 * (max_deflection - max_deflection_limit)
                return penalty + total_cost
            
            if max_stress_ratio > 1.0:
                # Penalize configurations that fail stress checks
                penalty = 5000 * (max_stress_ratio - 1.0)
                return penalty + total_cost
            
            # Normalize metrics for multi-objective optimization
            # Note: These normalization factors should be adjusted based on expected ranges
            normalized_deflection = max_deflection / max_deflection_limit
            
            objective_value = (
                objective_weights['cost'] +
                objective_weights['deflection'] * normalized_deflection
            )
            
            return objective_value
            
        except Exception as e:
            warnings.warn(f"Configuration failed: {str(e)}")
            return 1e6
    
    return objective


def setup_search_space() -> Tuple[list, list]:
    beam_ids = MATERIAL_CATALOG[MATERIAL_CATALOG['type'] == 'beam']['id'].unique().tolist()
    double_ids = MATERIAL_CATALOG[MATERIAL_CATALOG['type'] == 'double']['id'].unique().tolist()
    floor_ids = MATERIAL_CATALOG[MATERIAL_CATALOG['type'] == 'floor']['id'].unique().tolist()
    space = [
        Categorical(beam_ids, name='east_material'),
        Integer(1, 3, name='east_quantity'),
        Integer(0, 400, name='east_padding'),
        
        Categorical(beam_ids, name='west_material'),
        Integer(1, 3, name='west_quantity'),
        Integer(0, 400, name='west_padding'),
        
        Categorical(beam_ids, name='tail_material'),
        Integer(1, 3, name='tail_quantity'),
        Integer(0, 400, name='tail_padding'),
        
        Categorical(beam_ids + double_ids, name='trimmer_material'),
        Categorical(beam_ids + double_ids, name='header_material'),  
        Categorical(floor_ids, name='plank_material'),
    ]
    dimension_names = [dim.name for dim in space]
    
    return space, dimension_names


def run_bayesian_optimization(
    n_calls: int = 50,
    n_initial_points: int = 10,
    objective_weights: Dict[str, float] = None,
    max_deflection_limit: float = 10.0,
    random_state: int = 42) -> Dict:

    space, dimension_names = setup_search_space()
    objective_fn = create_optimization_objective(objective_weights, max_deflection_limit)
    
    # Wrapper for use_named_args decorator
    @use_named_args(space)
    def objective_wrapper(**params):
        return objective_fn(params)
    
    result = gp_minimize(
        objective_wrapper,
        space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=random_state,
        verbose=True,
        n_jobs=-1
    )
    
    optimization_summary = {
        'best_params': {name: val for name, val in zip(dimension_names, result.x)},
        'best_objective_value': result.fun,
        'n_calls': len(result.func_vals),
        'all_objective_values': result.func_vals,
        'all_params': result.x_iters,
        'dimension_names': dimension_names,
        'convergence_data': {
            'iterations': list(range(len(result.func_vals))),
            'objective_values': result.func_vals,
            'best_so_far': np.minimum.accumulate(result.func_vals)
        }
    }
    
    return optimization_summary


def evaluate_configuration(params_dict: Dict) -> Dict:    
    east_joists = MemberSpec(params_dict['east_material'], quantity=params_dict['east_quantity'], padding=params_dict['east_padding'])
    west_joists = MemberSpec(params_dict['west_material'], quantity=params_dict['west_quantity'], padding=params_dict['west_padding'])
    tail_joists = MemberSpec(params_dict['tail_material'], quantity=params_dict['tail_quantity'], padding=params_dict['tail_padding'])
    trimmers = MemberSpec(params_dict['trimmer_material'], quantity=2)
    header = MemberSpec(params_dict['header_material'], quantity=1)
    planks = MemberSpec(params_dict['plank_material'])
    
    frame, nodes, members, total_cost = create_model(
        east_joists=east_joists,
        west_joists=west_joists,
        tail_joists=tail_joists,
        trimmers=trimmers,
        header=header,
        planks=planks,
        walls=True
    )
    frame.analyze(check_statics=True)
    
    metrics = {
        'total_cost': 0,
        'total_volume_m3': 0,
        'max_deflection_mm': 0,
        'max_stress_ratio': 0,
        'passes_all_checks': True,
        'member_details': []
    }
    
    for member_name, member in frame.members.items():
        if member_name.startswith('east'):
            spec = east_joists
        elif member_name.startswith('west'):
            spec = west_joists
        elif member_name.startswith('tail'):
            spec = tail_joists
        elif member_name.startswith('trimmer'):
            spec = trimmers
        elif member_name == 'header':
            spec = header
        elif member_name.startswith('p'):
            spec = planks
        else:
            continue
        
        length = member.L()
        metrics['total_cost'] += spec.get_cost(length)
        metrics['total_volume_m3'] += spec.get_volume(length) / 1e9
        
        deflection = abs(member.min_deflection('dy', 'Combo 1'))
        metrics['max_deflection_mm'] = max(metrics['max_deflection_mm'], deflection)
        
        # Stress ratios
        max_moment = max(
            abs(member.max_moment('Mz', 'Combo 1')),
            abs(member.min_moment('Mz', 'Combo 1'))
        )
        max_shear = max(
            abs(member.max_shear('Fy', 'Combo 1')),
            abs(member.min_shear('Fy', 'Combo 1'))
        )
        
        geom = spec.get_geometry()
        mat_props = MATERIAL_STRENGTHS[spec.material]
        
        c = spec.height / 2
        bending_stress = abs(max_moment * c / geom.Iz)
        bending_ratio = bending_stress / mat_props['f_mk']
        
        if spec.shape == 'rectangular':
            shear_stress = 1.5 * abs(max_shear) / geom.A
        else:
            catalog_data = MATERIAL_CATALOG[MATERIAL_CATALOG['id'] == spec.catalog_id].iloc[0]
            web_area = (spec.height - 2*catalog_data['flange_thickness']) * catalog_data['web_thickness']
            shear_stress = abs(max_shear) / web_area
        
        shear_ratio = shear_stress / mat_props['f_vk']
        
        max_ratio = max(bending_ratio, shear_ratio)
        metrics['max_stress_ratio'] = max(metrics['max_stress_ratio'], max_ratio)
        
        if bending_ratio > 1.0 or shear_ratio > 1.0:
            metrics['passes_all_checks'] = False
        
        metrics['member_details'].append({
            'name': member_name,
            'deflection': deflection,
            'bending_ratio': bending_ratio,
            'shear_ratio': shear_ratio
        })
    
    return metrics


if __name__ == '__main__':
    # Units are mm, N, and MPa (N/mm²)
    params = pd.read_csv('data/design_parameters.csv').iloc[0].to_dict()
    INPUT_PARAMS = DesignParameters(**params)
    MATERIAL_STRENGTHS = pd.read_csv('data/material_strengths.csv').set_index('material').to_dict(orient='index')
    MATERIAL_CATALOG = pd.read_csv('data/material_catalog.csv')
    
    results = run_bayesian_optimization(
        design_params=INPUT_PARAMS,
        material_strengths=MATERIAL_STRENGTHS,
        n_calls=100,
        n_initial_points=20,
        objective_weights={'cost': 0.5, 'deflection': 0.3},
        max_deflection_limit=10.0,
    )
    
    best_metrics = evaluate_configuration(
        results['best_params'],
        MATERIAL_CATALOG,
        INPUT_PARAMS,
        MATERIAL_STRENGTHS
    )
    
    print(f"Total cost: ${best_metrics['total_cost']:.2f}")
    print(f"Total volume: {best_metrics['total_volume_m3']:.4f} m³")
    print(f"Max deflection: {best_metrics['max_deflection_mm']:.3f} mm")
    print(f"Max stress ratio: {best_metrics['max_stress_ratio']:.3f}")
    print(f"Passes all checks: {best_metrics['passes_all_checks']}")
    
    results_df = pd.DataFrame([results['best_params']])
    results_df['best_objective'] = results['best_objective_value']
    results_df.to_csv('optimization_results.csv', index=False)



if __name__ == '__main__':

    east_joists = MemberSpec('c24_60x120', quantity=1, padding=0)
    tail_joists = MemberSpec('c24_60x120', quantity=1, padding=0)
    west_joists = MemberSpec('c24_60x120', quantity=1, padding=0)
    trimmers = MemberSpec('c24_80x160', quantity=2)
    header = MemberSpec('c24_60x120', quantity=1)
    planks = MemberSpec('c14_200x25')

    frame, nodes, members, total_cost = create_model(
        east_joists=east_joists,
        west_joists=west_joists,
        tail_joists=tail_joists,
        trimmers=trimmers,
        header=header,
        planks=planks,
        walls=True
    )
    frame.analyze(check_statics=True)
    render(frame, deformed_scale=100, opacity=0.25)