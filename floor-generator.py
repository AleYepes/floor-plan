from dataclasses import dataclass, replace
from re import L
from typing import List, Dict, Optional, Tuple, Literal
from Pynite import FEModel3D
from Pynite.Rendering import Renderer
import numpy as np
import pandas as pd

from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real
from skopt.utils import use_named_args
import warnings


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
        BEAM_CATALOG['id'] = (BEAM_CATALOG['material'] + BEAM_CATALOG['base'].astype(str) + 'x' + BEAM_CATALOG['height'].astype(str))
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
        frame.add_material(
            mat_name,
            E=props['E'],
            G=props['E'] / (2 * (1 + props['nu'])),
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


def create_model(
    east_joists: MemberSpec,
    west_joists: MemberSpec,
    tail_joists: MemberSpec,
    trimmers: MemberSpec,
    header: MemberSpec,
    planks: MemberSpec,
    walls: bool) -> Tuple:
    
    nodes, members = calculate_nodes_and_members(east_joists, west_joists, tail_joists, trimmers, header, planks)
    frame = assemble_frame(nodes, members)
    define_supports(frame, nodes, INPUT_PARAMS.wall_thickness, 'brick', walls=walls)
    apply_loads(frame, members)
    
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


def create_optimization_objective(
    catalog: pd.DataFrame,
    material_strengths: Dict,
    objective_weights: Dict[str, float] = None,
    max_deflection_limit: float = 10.0):
    
    if objective_weights is None:
        objective_weights = {'cost': 0.6, 'deflection': 0.3, 'volume': 0.1}
    
    # Normalize weights
    total_weight = sum(objective_weights.values())
    objective_weights = {k: v/total_weight for k, v in objective_weights.items()}
    
    def objective(params_dict: Dict) -> float:
        """
        Objective function that creates and analyzes a floor configuration.
        Returns a scalar cost to minimize.
        """
        try:
            # Create MemberSpec objects from parameters
            east_joists = MemberSpec(
                catalog_id=params_dict['east_material'],
                quantity=params_dict.get('east_quantity', 0),
                padding=params_dict.get('east_padding', 0)
            )
            
            west_joists = MemberSpec(
                catalog_id=params_dict['west_material'],
                quantity=params_dict.get('west_quantity', 0),
                padding=params_dict.get('west_padding', 0)
            )
            
            tail_joists = MemberSpec(
                catalog_id=params_dict['tail_material'],
                quantity=params_dict.get('tail_quantity', 0),
                padding=params_dict.get('tail_padding', 0)
            )
            
            trimmers = MemberSpec(
                catalog_id=params_dict['trimmer_material'],
                quantity=2
            )
            
            header = MemberSpec(
                catalog_id=params_dict['header_material'],
                quantity=1
            )
            
            planks = MemberSpec(catalog_id=params_dict['plank_material'])
            
            # Create and analyze model
            frame = create_model(
                east_joists=east_joists,
                west_joists=west_joists,
                tail_joists=tail_joists,
                trimmers=trimmers,
                header=header,
                planks=planks,
                walls=True
            )
            
            # Analyze
            frame.analyze(check_statics=True)
            
            # Calculate metrics
            total_cost = 0
            total_volume = 0
            max_deflection = 0
            max_stress_ratio = 0
            
            for member_name, member in frame.members.items():
                # Get member spec (reconstruct to get properties)
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
                
                # Cost and volume
                length = member.L()
                total_cost += spec.get_cost(length)
                total_volume += spec.get_volume(length)
                
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
                mat_props = material_strengths[spec.material]
                
                c = spec.height / 2
                bending_stress = abs(max_moment * c / geom.Iz)
                bending_ratio = bending_stress / mat_props['f_mk']
                
                if spec.shape == 'rectangular':
                    shear_stress = 1.5 * abs(max_shear) / geom.A
                else:  # I-beam
                    catalog_data = catalog[catalog['id'] == spec.catalog_id].iloc[0]
                    web_area = (spec.height - 2*catalog_data['flange_thickness']) * catalog_data['web_thickness']
                    shear_stress = abs(max_shear) / web_area
                
                shear_ratio = shear_stress / mat_props['f_vk']
                
                max_stress_ratio = max(max_stress_ratio, bending_ratio, shear_ratio)
            
            # Check constraints
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
            normalized_cost = total_cost / 1000  # Assuming costs in range 0-10000
            normalized_deflection = max_deflection / max_deflection_limit
            normalized_volume = (total_volume / 1e9) / 1.0  # Volume in m³, normalized by 1 m³
            
            # Weighted sum of objectives
            objective_value = (
                objective_weights.get('cost', 0) * normalized_cost +
                objective_weights.get('deflection', 0) * normalized_deflection +
                objective_weights.get('volume', 0) * normalized_volume
            )
            
            return objective_value
            
        except Exception as e:
            # Return high penalty for invalid configurations
            warnings.warn(f"Configuration failed: {str(e)}")
            return 1e6
    
    return objective


def setup_search_space(catalog: pd.DataFrame) -> Tuple[list, list]:
    """
    Sets up the search space for Bayesian optimization.
    
    Args:
        catalog: Beam catalog dataframe
    
    Returns:
        Tuple of (space, dimension_names)
    """
    
    # Get available catalog IDs
    available_ids = catalog['id'].unique().tolist()
    
    # Define search space
    space = [
        # East joists
        Categorical(available_ids, name='east_material'),
        Integer(0, 5, name='east_quantity'),
        Real(0, 200, name='east_padding'),
        
        # West joists
        Categorical(available_ids, name='west_material'),
        Integer(0, 5, name='west_quantity'),
        Real(0, 200, name='west_padding'),
        
        # Tail joists
        Categorical(available_ids, name='tail_material'),
        Integer(0, 5, name='tail_quantity'),
        Real(0, 200, name='tail_padding'),
        
        # Trimmers (no quantity/padding optimization)
        Categorical(available_ids, name='trimmer_material'),
        
        # Header (no quantity/padding optimization)
        Categorical(available_ids, name='header_material'),
        
        # Planks
        Categorical(available_ids, name='plank_material'),
    ]
    
    dimension_names = [dim.name for dim in space]
    
    return space, dimension_names


def run_bayesian_optimization(
    catalog: pd.DataFrame,
    design_params: DesignParameters,
    material_strengths: Dict,
    n_calls: int = 50,
    n_initial_points: int = 10,
    objective_weights: Dict[str, float] = None,
    max_deflection_limit: float = 10.0,
    random_state: int = 42
) -> Dict:
    """
    Runs Bayesian optimization to find optimal floor configuration.
    
    Args:
        catalog: Beam catalog dataframe
        design_params: Design parameters instance
        material_strengths: Material properties dictionary
        n_calls: Number of optimization iterations
        n_initial_points: Number of random initial points
        objective_weights: Weights for multi-objective optimization
        max_deflection_limit: Maximum allowable deflection in mm
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary containing optimization results
    """
    
    # Setup search space
    space, dimension_names = setup_search_space(catalog)
    
    # Create objective function
    objective_fn = create_optimization_objective(
        catalog=catalog,
        design_params=design_params,
        material_strengths=material_strengths,
        objective_weights=objective_weights,
        max_deflection_limit=max_deflection_limit
    )
    
    # Wrapper for use_named_args decorator
    @use_named_args(space)
    def objective_wrapper(**params):
        return objective_fn(params)
    
    # Run optimization
    print(f"Starting Bayesian optimization with {n_calls} iterations...")
    print(f"Search space: {len(space)} dimensions")
    print(f"Initial random points: {n_initial_points}")
    
    result = gp_minimize(
        objective_wrapper,
        space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=random_state,
        verbose=True,
        n_jobs=1  # Set to -1 for parallel evaluation if possible
    )
    
    # Extract best parameters
    best_params = {name: val for name, val in zip(dimension_names, result.x)}
    
    # Create summary
    optimization_summary = {
        'best_params': best_params,
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
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nBest objective value: {result.fun:.4f}")
    print("\nBest configuration:")
    for name, value in best_params.items():
        print(f"  {name}: {value}")
    
    return optimization_summary


def evaluate_configuration(
    params_dict: Dict,
    catalog: pd.DataFrame,
    design_params: DesignParameters,
    material_strengths: Dict
) -> Dict:
    """
    Evaluate a specific configuration and return detailed metrics.
    
    Args:
        params_dict: Dictionary of parameter values
        catalog: Beam catalog dataframe
        design_params: Design parameters instance
        material_strengths: Material properties dictionary
    
    Returns:
        Dictionary with detailed performance metrics
    """
    
    # Create MemberSpec objects
    east_joists = MemberSpec(
        catalog_id=params_dict['east_material'],
        quantity=params_dict.get('east_quantity', 0),
        padding=params_dict.get('east_padding', 0)
    )
    
    west_joists = MemberSpec(
        catalog_id=params_dict['west_material'],
        quantity=params_dict.get('west_quantity', 0),
        padding=params_dict.get('west_padding', 0)
    )
    
    tail_joists = MemberSpec(
        catalog_id=params_dict['tail_material'],
        quantity=params_dict.get('tail_quantity', 0),
        padding=params_dict.get('tail_padding', 0)
    )
    
    trimmers = MemberSpec(
        catalog_id=params_dict['trimmer_material'],
        quantity=2
    )
    
    header = MemberSpec(
        catalog_id=params_dict['header_material'],
        quantity=1
    )
    
    planks = MemberSpec(catalog_id=params_dict['plank_material'])
    
    # Create and analyze model
    frame = create_model(
        east_joists=east_joists,
        west_joists=west_joists,
        tail_joists=tail_joists,
        trimmers=trimmers,
        header=header,
        planks=planks,
        walls=True
    )
    
    frame.analyze(check_statics=True)
    
    # Collect detailed metrics
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
        mat_props = material_strengths[spec.material]
        
        c = spec.height / 2
        bending_stress = abs(max_moment * c / geom.Iz)
        bending_ratio = bending_stress / mat_props['f_mk']
        
        if spec.shape == 'rectangular':
            shear_stress = 1.5 * abs(max_shear) / geom.A
        else:
            catalog_data = catalog[catalog['id'] == spec.catalog_id].iloc[0]
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
    BEAM_CATALOG = pd.read_csv('data/material_catalog.csv')
    BEAM_CATALOG['id'] = (BEAM_CATALOG['material'] + BEAM_CATALOG['base'].astype(str) + 'x' + BEAM_CATALOG['height'].astype(str))
    
    results = run_bayesian_optimization(
        catalog=BEAM_CATALOG,
        design_params=INPUT_PARAMS,
        material_strengths=MATERIAL_STRENGTHS,
        n_calls=100,
        n_initial_points=20,
        objective_weights={'cost': 0.5, 'deflection': 0.3, 'volume': 0.2},
        max_deflection_limit=10.0,
        random_state=42
    )
    
    best_metrics = evaluate_configuration(
        results['best_params'],
        BEAM_CATALOG,
        INPUT_PARAMS,
        MATERIAL_STRENGTHS
    )
    
    print("\n" + "="*70)
    print("DETAILED EVALUATION OF BEST CONFIGURATION")
    print("="*70)
    print(f"Total cost: ${best_metrics['total_cost']:.2f}")
    print(f"Total volume: {best_metrics['total_volume_m3']:.4f} m³")
    print(f"Max deflection: {best_metrics['max_deflection_mm']:.3f} mm")
    print(f"Max stress ratio: {best_metrics['max_stress_ratio']:.3f}")
    print(f"Passes all checks: {best_metrics['passes_all_checks']}")
    
    # Save results
    results_df = pd.DataFrame([results['best_params']])
    results_df['best_objective'] = results['best_objective_value']
    results_df.to_csv('optimization_results.csv', index=False)
    print("\nResults saved to 'optimization_results.csv'")



# if __name__ == '__main__':

#     # Units are mm, N, and MPa (N/mm²)
#     params = pd.read_csv('data/design_parameters.csv').iloc[0].to_dict()
#     INPUT_PARAMS = DesignParameters(**params)
#     MATERIAL_STRENGTHS = pd.read_csv('data/material_strengths.csv').set_index('material').to_dict(orient='index')
#     BEAM_CATALOG = pd.read_csv('data/material_catalog.csv')

#     east_joists = MemberSpec('wood60x120', quantity=1, padding=0)
#     tail_joists = MemberSpec('wood60x120', quantity=1, padding=0)
#     west_joists = MemberSpec('wood60x120', quantity=1, padding=0)
#     trimmers = MemberSpec('wood80x160', quantity=2)
#     header = MemberSpec('wood60x120', quantity=1)
#     planks = MemberSpec('wood200x18')

#     frame = create_model(
#         east_joists=east_joists,
#         west_joists=west_joists,
#         tail_joists=tail_joists,
#         trimmers=trimmers,
#         header=header,
#         planks=planks,
#         walls=True
#     )
#     frame.analyze(check_statics=True)
#     # render(frame, deformed_scale=100, opacity=0.25)