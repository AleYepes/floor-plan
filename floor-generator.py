from dataclasses import dataclass, field, replace
from typing import List, Dict, Optional, Tuple
from Pynite import FEModel3D
from Pynite.Rendering import Renderer
import numpy as np
import pandas as pd
from itertools import product
from tqdm.auto import tqdm


# NAMING CONVENTION CLASS
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


# CONSTANTS AND MATERIAL PROPERTIES
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
    def floor_to_floor(self):
        return self.room_height / 2 + self.plank_thickness / 2

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

def load_beam_catalog(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise SystemExit(f"Error: Beam catalog file not found at '{path}'")
    except Exception as e:
        raise SystemExit(f"Error loading beam catalog from '{path}': {e}")

# Units are mm, N, and MPa (N/mm²)
DEFAULT_PARAMS = load_design_parameters('data/design_parameters.csv')
MATERIAL_STRENGTHS = load_material_strengths('data/material_strengths.csv')
BEAM_CATALOG = load_beam_catalog('data/beam_catalog.csv')


# CROSS-SECTION GEOMETRY CALCULATIONS
@dataclass
class CrossSectionGeometry:
    """Calculates geometric properties for different cross-section shapes"""
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
    def from_i_beam(cls, height: float, flange_width: float, 
                    flange_thickness: float, web_thickness: float) -> 'CrossSectionGeometry':
        """Calculate properties for symmetric I-beam"""
        # Area
        A_flanges = 2 * flange_width * flange_thickness
        A_web = (height - 2 * flange_thickness) * web_thickness
        A = A_flanges + A_web
        
        # Iz (strong axis - bending about horizontal axis)
        Iz_flanges = 2 * (flange_width * flange_thickness**3 / 12 + 
                         flange_width * flange_thickness * ((height - flange_thickness)/2)**2)
        web_height = height - 2 * flange_thickness
        Iz_web = web_thickness * web_height**3 / 12
        Iz = Iz_flanges + Iz_web
        
        # Iy (weak axis - bending about vertical axis)
        Iy_flanges = 2 * (flange_thickness * flange_width**3 / 12)
        Iy_web = web_height * web_thickness**3 / 12
        Iy = Iy_flanges + Iy_web
        
        # Torsional constant (approximate for thin-walled I-beam)
        J = (2 * flange_width * flange_thickness**3 + web_height * web_thickness**3) / 3
        
        return cls(A=A, Iy=Iy, Iz=Iz, J=J)


# BEAM SPECIFICATION CLASSES
@dataclass
class BeamSpec:
    """Beam specification using catalog ID"""
    catalog_id: str
    beam_type: str  # 'joist', 'tail', 'trimmer', 'header'
    name: str = ''
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
        """Get cross-section geometry"""
        if self.shape == 'rectangular':
            return CrossSectionGeometry.from_rectangular(self.base, self.height)
        elif self.shape == 'I-beam':
            return CrossSectionGeometry.from_i_beam(
                self.height, 
                self._catalog_data['flange_width'],
                self._catalog_data['flange_thickness'],
                self._catalog_data['web_thickness']
            )
        else:
            raise ValueError(f"Unknown shape: {self.shape}")
    
    def create_section(self, frame: FEModel3D):
        """Add section to frame if not already present"""
        if self.section_name not in frame.sections:
            geom = self.get_geometry()
            frame.add_section(self.section_name, geom.A, geom.Iy, geom.Iz, geom.J)
    
    def get_volume(self, length: float) -> float:
        """Calculate volume in mm^3"""
        geometry = self.get_geometry()
        return geometry.A * length
    
    def get_cost(self, length: float) -> float:
        """Calculate cost for given length (length in mm)"""
        volume_m3 = self.get_volume(length) / 1e9  # Convert mm^3 to m^3
        return volume_m3 * self.cost_per_m3
    
    def copy(self, **kwargs):
        return replace(self, **kwargs)


@dataclass
class BeamPlacement:
    spec: BeamSpec
    x_center: float
    z_start: float = 0
    z_end: float = None

    def add_to_frame(self, frame: FEModel3D, params: DesignParameters, default_z_end: float):
        z_end = self.z_end if self.z_end is not None else default_z_end
        name = self.spec.name

        frame.add_node(Naming.node(name, 'S', floor=True), self.x_center, 0, self.z_start)
        frame.add_node(Naming.node(name, 'S'), self.x_center, params.floor_to_floor, self.z_start)
        
        if self.spec.beam_type != 'tail':
            frame.add_node(Naming.node(name, 'N', floor=True), self.x_center, 0, z_end)
            frame.add_node(Naming.node(name, 'N'), self.x_center, params.floor_to_floor, z_end)
            self.spec.create_section(frame)
            frame.add_member(Naming.member(name), Naming.node(name, 'N'), Naming.node(name, 'S'), self.spec.material, self.spec.section_name)
        else:
            self.spec.create_section(frame)


# GEOMETRY RESOLVER
class GeometryResolver:
    def __init__(
        self,
        east_joists: BeamSpec,
        west_joists: BeamSpec,
        tail_joists: BeamSpec,
        trimmers: BeamSpec,
        header: BeamSpec,
        params: DesignParameters
    ):
        self.east_joists = east_joists
        self.west_joists = west_joists
        self.tail_joists = tail_joists
        self.trimmers = trimmers
        self.header = header
        self.params = params
        self.opening_x_start = params.opening_x_start

        assert self.trimmers.quantity == 2, "Must have exactly 2 trimmers"
        assert self.header.quantity == 1, "Must have exactly 1 header"
        
        # Check padding doesn't exceed available space
        max_east_padding = self.opening_x_start - self.trimmers.base - self.east_joists.base
        assert self.east_joists.padding <= max_east_padding, \
            f"East padding {self.east_joists.padding} exceeds max {max_east_padding}"
        
        opening_x_end = self.opening_x_start + self.params.opening_length
        max_west_padding = self.params.room_length - opening_x_end - self.trimmers.base - self.west_joists.base
        assert self.west_joists.padding <= max_west_padding, \
            f"West padding {self.west_joists.padding} exceeds max {max_west_padding}"

        self.opening_x_end = self.opening_x_start + self.params.opening_length
        
        self.trimmer_E_x_center = self.opening_x_start - (self.trimmers.base / 2)
        self.trimmer_W_x_center = self.opening_x_end + (self.trimmers.base / 2)
        
        self.tail_z_end = self.params.beam_length - self.params.opening_width - self.params.wall_beam_contact_depth/2 - self.header.base/2
        self.header_z_pos = self.tail_z_end
        
        self.opening_z_start = self.params.opening_width + self.params.wall_beam_contact_depth/2

    def _resolve_joist_positions(self, n: int, clear_start: float, clear_end: float, beam_base: float, group: str) -> List[float]:
        def east_positions(n, clear_start, clear_end, beam_base):
            centerline_start = clear_start + beam_base / 2
            centerline_end = clear_end + beam_base / 2
            positions = np.linspace(centerline_start, centerline_end, n+1).tolist()
            return positions[:-1]

        def west_positions(n, clear_start, clear_end, beam_base):
            centerline_start = clear_start - beam_base / 2
            centerline_end = clear_end + beam_base / 2
            positions = np.linspace(centerline_end, centerline_start, n+1).tolist()
            return positions[:-1]

        def tail_positions(n, clear_start, clear_end, beam_base):
            centerline_start = clear_start - beam_base / 2
            centerline_end = clear_end + beam_base / 2
            positions = np.linspace(centerline_start, centerline_end, n+2).tolist()
            return positions[1:-1]

        dispatch = {
            'east': east_positions,
            'west': west_positions,
            'tail': tail_positions
        }

        if group not in dispatch:
            raise ValueError("Invalid group specified")

        return dispatch[group](n, clear_start, clear_end, beam_base)

    def resolve_all_placements(self) -> List[BeamPlacement]:
        all_placements = []
        
        # East joists
        if self.east_joists.quantity > 0:
            joist_spec_east = BeamSpec(
                self.east_joists.catalog_id, 
                'joist',
                quantity=self.east_joists.quantity,
                padding=self.east_joists.padding
            )
            clear_start = self.east_joists.padding
            clear_end = self.trimmer_E_x_center - self.trimmers.base / 2
            east_positions = self._resolve_joist_positions(
                self.east_joists.quantity, 
                clear_start, 
                clear_end, 
                joist_spec_east.base,
                "east"
            )
            for i, x in enumerate(east_positions):
                all_placements.append(BeamPlacement(spec=joist_spec_east.copy(name=f'E{i}'), x_center=x))
        
        # Tail joists
        if self.tail_joists.quantity > 0:
            tail_spec = BeamSpec(
                self.tail_joists.catalog_id, 
                'tail',
                quantity=self.tail_joists.quantity,
                padding=self.tail_joists.padding
            )
            clear_start = self.opening_x_start + self.tail_joists.padding
            clear_end = self.opening_x_end - self.tail_joists.padding
            tail_positions = self._resolve_joist_positions(
                self.tail_joists.quantity, 
                clear_start, 
                clear_end, 
                tail_spec.base,
                "tail"
            )
            for i, x in enumerate(tail_positions):
                all_placements.append(BeamPlacement(spec=tail_spec.copy(name=f'T{i}'), x_center=x, z_end=self.tail_z_end))
        
        # West joists
        if self.west_joists.quantity > 0:
            joist_spec_west = BeamSpec(
                self.west_joists.catalog_id, 
                'joist',
                quantity=self.west_joists.quantity,
                padding=self.west_joists.padding
            )
            clear_start = self.trimmer_W_x_center + self.trimmers.base / 2
            clear_end = self.params.room_length - self.west_joists.padding - self.west_joists.base
            west_positions = self._resolve_joist_positions(
                self.west_joists.quantity, 
                clear_start,
                clear_end,
                joist_spec_west.base,
                "west"
            )
            for i, x in enumerate(west_positions):
                all_placements.append(BeamPlacement(spec=joist_spec_west.copy(name=f'W{i}'), x_center=x))

        # Trimmers
        trimmer_spec = BeamSpec(
            self.trimmers.catalog_id, 
            'trimmer',
            quantity=self.trimmers.quantity,
            padding=self.trimmers.padding
        )
        all_placements.append(BeamPlacement(spec=trimmer_spec.copy(name='trimmer_E'), x_center=self.trimmer_E_x_center))
        all_placements.append(BeamPlacement(spec=trimmer_spec.copy(name='trimmer_W'), x_center=self.trimmer_W_x_center))

        return all_placements

    def add_header_to_frame(self, frame: FEModel3D):
        """Add header member to frame - call this separately"""
        header_spec = self.header
        header_spec.create_section(frame)
        frame.add_node(Naming.header_node('E'), self.trimmer_E_x_center, 
                       self.params.floor_to_floor, self.header_z_pos)
        frame.add_node(Naming.header_node('W'), self.trimmer_W_x_center, 
                       self.params.floor_to_floor, self.header_z_pos)
        frame.add_member(Naming.header_member(), Naming.header_node('W'), 
                         Naming.header_node('E'), header_spec.material, 
                         header_spec.section_name)

    def connect_tails_to_header(self, frame: FEModel3D, layout: 'LayoutManager'):
        # Connect tail joists to header
        for beam_placement in layout.beams:
            if beam_placement.spec.beam_type == 'tail':
                tail_node_name = Naming.tail_header_node(beam_placement.spec.name)
                frame.add_node(tail_node_name, beam_placement.x_center, self.params.floor_to_floor, self.header_z_pos)
                frame.add_member(
                    Naming.member(beam_placement.spec.name),
                    tail_node_name,
                    Naming.node(beam_placement.spec.name, 'S'),
                    beam_placement.spec.material,
                    beam_placement.spec.section_name
                )


# LAYOUT MANAGER
class LayoutManager:
    def __init__(self, params: DesignParameters):
        self.params = params
        self.beams: List[BeamPlacement] = []

    def add_beams(self, placements: List[BeamPlacement]):
        self.beams.extend(placements)

    def get_boundary_beams(self) -> Tuple[BeamPlacement, BeamPlacement]:
        beams_with_position = sorted(self.beams, key=lambda p: p.x_center)
        return beams_with_position[0], beams_with_position[-1]


# LOAD APPLICATOR
class LoadApplicator:
    def __init__(self, beams: List[BeamPlacement], params: DesignParameters):
        self.beams = beams
        self.params = params

    def apply_dead_loads(self, frame: FEModel3D):
        for p in self.beams:
            if p.spec.beam_type == 'header':
                continue
            section_geom = p.spec.get_geometry()
            material = MATERIAL_STRENGTHS[p.spec.material]
            dead_load = -section_geom.A * material['rho']
            frame.add_member_dist_load(p.spec.name, 'FY', dead_load, dead_load)
    
    def apply_live_loads(self, frame: FEModel3D):
        sorted_beams = sorted(self.beams, key=lambda p: p.x_center)
        if len(sorted_beams) < 2:
            return

        for i, p in enumerate(sorted_beams):
            # Calculate tributary boundaries
            left_boundary = sorted_beams[i-1].x_center if i > 0 else 0
            right_boundary = sorted_beams[i+1].x_center if i < len(sorted_beams) - 1 else self.params.room_length
            
            trib_left = (p.x_center - left_boundary) / 2
            trib_right = (right_boundary - p.x_center) / 2
            
            load_left = self.params.live_load_mpa * trib_left
            load_right = self.params.live_load_mpa * trib_right

            if p.spec.beam_type in ['joist', 'tail']:
                total_load = load_left + load_right
                frame.add_member_dist_load(p.spec.name, 'FY', total_load, total_load)
            elif p.spec.beam_type == 'trimmer':
                # For trimmers, one side has a load break at the opening
                is_left_opening = i > 0 and sorted_beams[i-1].spec.beam_type == 'tail'
                is_right_opening = i < len(sorted_beams)-1 and sorted_beams[i+1].spec.beam_type == 'tail'
                
                if is_left_opening:
                    frame.add_member_dist_load(p.spec.name, 'FY', load_left, load_left, 0, self.params.opening_width + self.params.wall_beam_contact_depth/2)
                    frame.add_member_dist_load(p.spec.name, 'FY', load_right, load_right)
                elif is_right_opening:
                    frame.add_member_dist_load(p.spec.name, 'FY', load_left, load_left)
                    frame.add_member_dist_load(p.spec.name, 'FY', load_right, load_right, 0, self.params.opening_width + self.params.wall_beam_contact_depth/2)
                else:
                    frame.add_member_dist_load(p.spec.name, 'FY', load_left + load_right, load_left + load_right)


# WALL GENERATION
def auto_add_walls(frame, layout, wall_thickness, material):
    eastmost_beam, westmost_beam = layout.get_boundary_beams()
    
    frame.add_quad(Naming.side_wall('west'), Naming.node(westmost_beam.spec.name, 'S', floor=True), Naming.node(westmost_beam.spec.name, 'N', floor=True), Naming.node(westmost_beam.spec.name, 'N'), Naming.node(westmost_beam.spec.name, 'S'), wall_thickness, material)
    frame.add_quad(Naming.side_wall('east'), Naming.node(eastmost_beam.spec.name, 'N', floor=True), Naming.node(eastmost_beam.spec.name, 'S', floor=True), Naming.node(eastmost_beam.spec.name, 'S'), Naming.node(eastmost_beam.spec.name, 'N'), wall_thickness, material)
    
    # South wall should be continuous and connect all beams
    sorted_beams = sorted(layout.beams, key=lambda p: p.x_center)
    prev_beam = None
    for beam in sorted_beams:
        if prev_beam is None:
            prev_beam = beam
            continue
        frame.add_quad(Naming.wall(prev_beam.spec.name, beam.spec.name, 'south'), Naming.node(prev_beam.spec.name, 'S', floor=True), Naming.node(beam.spec.name, 'S', floor=True), Naming.node(beam.spec.name, 'S'), Naming.node(prev_beam.spec.name, 'S'), wall_thickness, material)
        prev_beam = beam
    
    # North wall should have a gap for the opening
    north_reaching_beams = [b for b in layout.beams if b.spec.beam_type not in ['tail', 'header']]
    north_reaching_beams.sort(key=lambda p: p.x_center)
    prev_beam = None
    for beam in north_reaching_beams:
        if prev_beam is None:
            prev_beam = beam
            continue
        frame.add_quad(Naming.wall(prev_beam.spec.name, beam.spec.name, 'north'), Naming.node(prev_beam.spec.name, 'N', floor=True), Naming.node(beam.spec.name, 'N', floor=True), Naming.node(beam.spec.name, 'N'), Naming.node(prev_beam.spec.name, 'N'), wall_thickness, material)
        prev_beam = beam


# TELEMETRY CLASSES
@dataclass
class BeamTelemetry:
    name: str
    material: str
    catalog_id: str
    length: float
    max_moment_mz: float
    min_moment_mz: float
    max_shear_fy: float
    min_shear_fy: float
    max_deflection: float
    min_deflection: float
    volume: float
    cost: float
    stress_max: float
    shear_stress_max: float
    passes_bending: bool
    passes_shear: bool

@dataclass
class GroupTelemetry:
    group_name: str
    count: int
    max_deflection: float
    mean_deflection: float
    max_moment: float
    mean_moment: float
    max_shear: float
    mean_shear: float
    total_volume: float
    total_cost: float
    all_pass_bending: bool
    all_pass_shear: bool
    beam_details: List[BeamTelemetry]

@dataclass
class SystemTelemetry:
    max_header_deflection_mm: float
    max_deflection_overall_mm: float
    worst_member: str
    total_volume_m3: float
    total_cost: float
    system_passes: bool

@dataclass
class Telemetry:
    beam_telemetries: Dict[str, BeamTelemetry] = field(default_factory=dict)
    group_telemetries: Dict[str, GroupTelemetry] = field(default_factory=dict)
    system_telemetry: SystemTelemetry = None


class TelemetryCollector:
    """Collects and analyzes telemetry from FE analysis"""
    
    def __init__(self, frame, beam_specs: Dict[str, BeamSpec]):
        self.frame = frame
        self.beam_specs = beam_specs
        
    def calculate_beam_stress(self, member_name: str, moment: float, spec: BeamSpec) -> float:
        """Calculate bending stress: sigma = M * c / I"""
        geometry = spec.get_geometry()
        c = spec.height / 2
        stress = abs(moment * c / geometry.Iz)
        return stress
    
    def calculate_shear_stress(self, member_name: str, shear: float, spec: BeamSpec) -> float:
        """Calculate average shear stress"""
        geometry = spec.get_geometry()
        if spec.shape == 'rectangular':
            tau = 1.5 * abs(shear) / geometry.A
        else:  # I-beam
            catalog_data = spec._catalog_data
            web_area = (spec.height - 2*catalog_data['flange_thickness']) * catalog_data['web_thickness']
            tau = abs(shear) / web_area
        return tau
    
    def collect_beam_telemetry(self, member_name: str, spec: BeamSpec) -> BeamTelemetry:
        member = self.frame.members[member_name]
        
        # Get forces and deflections
        max_moment = member.max_moment('Mz', 'Combo 1')
        min_moment = member.min_moment('Mz', 'Combo 1')
        max_shear = member.max_shear('Fy', 'Combo 1')
        min_shear = member.min_shear('Fy', 'Combo 1')
        max_deflection = member.max_deflection('dy', 'Combo 1')
        min_deflection = member.min_deflection('dy', 'Combo 1')
        
        # Calculate stresses
        moment_for_stress = max(abs(max_moment), abs(min_moment))
        stress_max = self.calculate_beam_stress(member_name, moment_for_stress, spec)
        
        shear_for_stress = max(abs(max_shear), abs(min_shear))
        shear_stress_max = self.calculate_shear_stress(member_name, shear_for_stress, spec)
        
        # Check against material strengths
        material_props = MATERIAL_STRENGTHS[spec.material]
        passes_bending = stress_max <= material_props['f_mk']
        passes_shear = shear_stress_max <= material_props['f_vk']
        
        # Calculate cost and volume
        length = member.L()
        volume = spec.get_volume(length)
        cost = spec.get_cost(length)
        
        return BeamTelemetry(
            name=member_name,
            material=spec.material,
            catalog_id=spec.catalog_id,
            length=length,
            max_moment_mz=max_moment,
            min_moment_mz=min_moment,
            max_shear_fy=max_shear,
            min_shear_fy=min_shear,
            max_deflection=max_deflection,
            min_deflection=min_deflection,
            volume=volume,
            cost=cost,
            stress_max=stress_max,
            shear_stress_max=shear_stress_max,
            passes_bending=passes_bending,
            passes_shear=passes_shear
        )
    
    def collect_group_telemetry(self, group_name: str, member_names: List[str]) -> GroupTelemetry:
        beam_telemetries = [self.collect_beam_telemetry(name, self.beam_specs[name]) for name in member_names]
        
        if not beam_telemetries:
            return GroupTelemetry(
                group_name=group_name, count=0, max_deflection=0, mean_deflection=0,
                max_moment=0, mean_moment=0, max_shear=0, mean_shear=0,
                total_volume=0, total_cost=0, all_pass_bending=True, all_pass_shear=True,
                beam_details=[]
            )
        
        deflections = [abs(b.min_deflection) for b in beam_telemetries]
        moments = [max(abs(b.max_moment_mz), abs(b.min_moment_mz)) for b in beam_telemetries]
        shears = [max(abs(b.max_shear_fy), abs(b.min_shear_fy)) for b in beam_telemetries]
        
        return GroupTelemetry(
            group_name=group_name,
            count=len(beam_telemetries),
            max_deflection=max(deflections),
            mean_deflection=np.mean(deflections),
            max_moment=max(moments),
            mean_moment=np.mean(moments),
            max_shear=max(shears),
            mean_shear=np.mean(shears),
            total_volume=sum(b.volume for b in beam_telemetries),
            total_cost=sum(b.cost for b in beam_telemetries),
            all_pass_bending=all(b.passes_bending for b in beam_telemetries),
            all_pass_shear=all(b.passes_shear for b in beam_telemetries),
            beam_details=beam_telemetries
        )
    
    def collect_system_telemetry(self, beam_groups: Dict[str, List[str]]) -> Telemetry:
        """Collect telemetry for entire system"""
        telemetry = Telemetry()
        
        for group_name, member_names in beam_groups.items():
            if not member_names:
                continue
            group_telem = self.collect_group_telemetry(group_name, member_names)
            telemetry.group_telemetries[group_name] = group_telem
            for beam_telem in group_telem.beam_details:
                telemetry.beam_telemetries[beam_telem.name] = beam_telem
        
        # Overall metrics
        all_beam_stats = list(telemetry.beam_telemetries.values())
        max_deflection_overall = max((abs(b.min_deflection) for b in all_beam_stats), default=0)
        worst_member = max(all_beam_stats, key=lambda b: abs(b.min_deflection), default=None)
        
        total_volume = sum(g.total_volume for g in telemetry.group_telemetries.values())
        total_cost = sum(g.total_cost for g in telemetry.group_telemetries.values())
        
        system_passes = all(g.all_pass_bending and g.all_pass_shear for g in telemetry.group_telemetries.values())
        
        header_deflection = abs(self.frame.members[Naming.header_member()].min_deflection('dy', 'Combo 1'))
        
        telemetry.system_telemetry = SystemTelemetry(
            max_header_deflection_mm=header_deflection,
            max_deflection_overall_mm=max_deflection_overall,
            worst_member=worst_member.name if worst_member else '',
            total_volume_m3=total_volume / 1e9,
            total_cost=total_cost,
            system_passes=system_passes,
        )
        return telemetry

# MAIN ANALYSIS FUNCTION
def generate_and_analyze_floor(
    east_joists: BeamSpec,
    west_joists: BeamSpec,
    tail_joists: BeamSpec,
    trimmers: BeamSpec,
    header: BeamSpec,
) -> tuple:
    """Generate floor model, run analysis, and return frame with results"""
    resolver = GeometryResolver(
        east_joists=east_joists,
        west_joists=west_joists,
        tail_joists=tail_joists,
        trimmers=trimmers,
        header=header,
        params=DEFAULT_PARAMS
    )
    all_placements = resolver.resolve_all_placements()
    frame = FEModel3D()
    
    # Add materials
    for mat_name, props in MATERIAL_STRENGTHS.items():
        frame.add_material(
            mat_name, 
            E=props['E'], 
            G=props['E'] / (2 * (1 + props['nu'])), 
            nu=props['nu'], 
            rho=props['rho']
        )
    
    # Create layout and add all beams
    layout = LayoutManager(params=DEFAULT_PARAMS)
    layout.add_beams(all_placements)
    
    for beam_placement in layout.beams:
        beam_placement.add_to_frame(frame, DEFAULT_PARAMS, DEFAULT_PARAMS.beam_length)

    resolver.add_header_to_frame(frame)
    resolver.connect_tails_to_header(frame, layout)
    
    # Add walls and supports
    auto_add_walls(frame, layout, wall_thickness=DEFAULT_PARAMS.wall_thickness, material='brick')
    for node_name in frame.nodes:
        if node_name.startswith('floor'):
            frame.def_support(node_name, True, True, True, True, True, True)
    
    # Apply loads
    load_applicator = LoadApplicator(layout.beams, layout.params)
    load_applicator.apply_dead_loads(frame)
    load_applicator.apply_live_loads(frame)
    
    # Run analysis
    try:
        frame.analyze(check_statics=True)
    except Exception as e:
        raise RuntimeError(f"Analysis failed: {e}")
    
    # Build beam spec map for telemetry
    beam_spec_map = {}
    for beam_placement in all_placements:
        beam_spec_map[beam_placement.spec.name] = beam_placement.spec
    
    return frame, resolver, beam_spec_map


# OPTIMIZATION FRAMEWORK
class FloorOptimizer:
    """Handles optimization and telemetry collection"""
    
    def __init__(self):
        self.results = []
        
    def run_single_configuration(
        self,
        east_joists: BeamSpec,
        west_joists: BeamSpec,
        tail_joists: BeamSpec,
        trimmers: BeamSpec,
        header: BeamSpec,
        config_id: int = None
    ) -> Dict:
        """Run analysis for a single configuration and collect telemetry"""
        try:
            frame, resolver, beam_spec_map = generate_and_analyze_floor(
                east_joists=east_joists,
                west_joists=west_joists,
                tail_joists=tail_joists,
                trimmers=trimmers,
                header=header,
            )
            
            # Collect telemetry
            collector = TelemetryCollector(frame, beam_spec_map)
            
            # Define beam groups
            beam_groups = {
                'east_joists': [name for name, spec in beam_spec_map.items() if spec.beam_type == 'joist' and name.startswith('E')],
                'west_joists': [name for name, spec in beam_spec_map.items() if spec.beam_type == 'joist' and name.startswith('W')],
                'tail_joists': [name for name, spec in beam_spec_map.items() if spec.beam_type == 'tail'],
                'trimmers': [name for name, spec in beam_spec_map.items() if spec.beam_type == 'trimmer'],
                'header': [name for name, spec in beam_spec_map.items() if spec.beam_type == 'header']
            }
            
            telemetry = collector.collect_system_telemetry(beam_groups)
            
            return frame, {
                'config_id': config_id,
                'east_joists': east_joists,
                'west_joists': west_joists,
                'tail_joists': tail_joists,
                'trimmers': trimmers,
                'header': header,
                'telemetry': telemetry,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return frame, {
                'config_id': config_id,
                'east_joists': east_joists,
                'west_joists': west_joists,
                'tail_joists': tail_joists,
                'trimmers': trimmers,
                'header': header,
                'telemetry': None,
                'success': False,
                'error': str(e)
            }
    
    def run_grid_search(self, param_space: Dict[str, List]) -> pd.DataFrame:
        """
        Run grid search over parameter space
        
        Args:
            param_space: Dictionary defining parameter ranges, e.g.:
                {
                    'east_joist_catalog_id': ['W60x120', 'W80x160'],
                    'east_quantity': [2, 3, 4],
                    'trimmer_catalog_id': ['W80x160', 'W100x200'],
                    'opening_x_start': [800, 820, 840],
                    ...
                }
        """
        keys = list(param_space.keys())
        combinations = list(product(*[param_space[k] for k in keys]))
        print(len(combinations))
        
        print(f"Running grid search over {len(combinations)} configurations...")
        
        for i, combo in tqdm(enumerate(combinations), total=len(combinations)):
            config_params = dict(zip(keys, combo))
            
            # Build configuration
            east_joists_spec = BeamSpec(
                catalog_id=config_params['east_joist_catalog_id'],
                beam_type='joist',
                quantity=config_params['east_quantity'],
                padding=config_params.get('east_padding', 0)
            )
            west_joists_spec = BeamSpec(
                catalog_id=config_params['west_joist_catalog_id'],
                beam_type='joist',
                quantity=config_params['west_quantity'],
                padding=config_params.get('west_padding', 0)
            )
            tail_joists_spec = BeamSpec(
                catalog_id=config_params['tail_joist_catalog_id'],
                beam_type='tail',
                quantity=config_params['tail_quantity'],
                padding=config_params.get('tail_padding', 0)
            )
            trimmers_spec = BeamSpec(
                catalog_id=config_params['trimmer_catalog_id'],
                beam_type='trimmer',
                quantity=2
            )
            header_spec = BeamSpec(
                catalog_id=config_params['header_catalog_id'],
                beam_type='header',
                quantity=1
            )
            
            _, result = self.run_single_configuration(
                east_joists=east_joists_spec,
                west_joists=west_joists_spec,
                tail_joists=tail_joists_spec,
                trimmers=trimmers_spec,
                header=header_spec,
                config_id=i
            )
            self.results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{len(combinations)} configurations")
        
        return self.results_to_dataframe()
    
    def results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis"""
        rows = []
        
        for result in self.results:
            if not result['success']:
                row = {
                    'config_id': result['config_id'],
                    'success': False,
                    'error': result['error']
                }
                rows.append(row)
                continue
            
            telem = result['telemetry']
            
            row = {
                # Configuration
                'config_id': result['config_id'],
                'east_catalog_id': result['east_joists'].catalog_id,
                'east_quantity': result['east_joists'].quantity,
                'west_catalog_id': result['west_joists'].catalog_id,
                'west_quantity': result['west_joists'].quantity,
                'tail_catalog_id': result['tail_joists'].catalog_id,
                'tail_quantity': result['tail_joists'].quantity,
                'trimmer_catalog_id': result['trimmers'].catalog_id,
                'header_catalog_id': result['header'].catalog_id,
                'opening_x_start': DEFAULT_PARAMS.opening_x_start,
                
                # Overall metrics
                'max_deflection_mm': telem.system_telemetry.max_deflection_overall_mm,
                'header_deflection_mm': telem.system_telemetry.max_header_deflection_mm,
                'worst_member': telem.system_telemetry.worst_member,
                'total_volume_m3': telem.system_telemetry.total_volume_m3,
                'total_cost': telem.system_telemetry.total_cost,
                'system_passes': telem.system_telemetry.system_passes,
                
                # Group-level metrics
                **{f'{group}_max_deflection': stats.max_deflection 
                   for group, stats in telem.group_telemetries.items()},
                **{f'{group}_mean_deflection': stats.mean_deflection 
                   for group, stats in telem.group_telemetries.items()},
                **{f'{group}_max_moment': stats.max_moment 
                   for group, stats in telem.group_telemetries.items()},
                **{f'{group}_passes_bending': stats.all_pass_bending 
                   for group, stats in telem.group_telemetries.items()},
                **{f'{group}_passes_shear': stats.all_pass_shear 
                   for group, stats in telem.group_telemetries.items()},
                **{f'{group}_total_cost': stats.total_cost 
                   for group, stats in telem.group_telemetries.items()},
                
                'success': True,
                'error': None
            }
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_pareto_front(self, df: pd.DataFrame, 
                        objectives: List[str] = ['total_cost', 'max_deflection_mm']) -> pd.DataFrame:
        """Find Pareto-optimal solutions"""
        valid = df[(df['success'] == True) & (df['system_passes'] == True)].copy()
        
        if len(valid) == 0:
            print("No valid configurations found!")
            return pd.DataFrame()
        
        # Find Pareto front
        is_pareto = np.ones(len(valid), dtype=bool)
        
        for i, row_i in valid.iterrows():
            for j, row_j in valid.iterrows():
                if i == j:
                    continue
                
                # Check if j dominates i
                dominates = True
                for obj in objectives:
                    if row_j[obj] > row_i[obj]:
                        dominates = False
                        break
                
                if dominates:
                    strictly_better = any(row_j[obj] < row_i[obj] for obj in objectives)
                    if strictly_better:
                        is_pareto[valid.index.get_loc(i)] = False
                        break
        
        pareto_indices = valid.index[is_pareto]
        return valid.loc[pareto_indices]
    
if __name__ == '__main__':
    # Single solution with rendering
    optimizer = FloorOptimizer()
    frame, result = optimizer.run_single_configuration(
        east_joists=BeamSpec(catalog_id='W60x120', beam_type='joist', quantity=1, padding=0),
        west_joists=BeamSpec(catalog_id='W60x120', beam_type='joist', quantity=1, padding=0),
        tail_joists=BeamSpec(catalog_id='W60x120', beam_type='tail', quantity=1, padding=0),
        trimmers=BeamSpec(catalog_id='W80x160', beam_type='trimmer', quantity=2),
        header=BeamSpec(catalog_id='W120x120', beam_type='header', quantity=1),
        config_id=0
    )

    if result['success']:
        telem = result['telemetry']
        print(f"\nSystem passes all checks: {telem.system_telemetry.system_passes}")
        print(f"Total cost: ${telem.system_telemetry.total_cost:.2f}")
        print(f"Total volume: {telem.system_telemetry.total_volume_m3:.4f} m³")
        print(f"Max deflection: {telem.system_telemetry.max_deflection_overall_mm:.3f} mm")
        print(f"Header deflection: {telem.system_telemetry.max_header_deflection_mm:.3f} mm")
        print(f"Worst member: {telem.system_telemetry.worst_member}")
        
        print("\nGroup Statistics:")
        for group_name, stats in telem.group_telemetries.items():
            print(f"\n  {group_name}:")
            print(f"    Count: {stats.count}")
            print(f"    Max deflection: {stats.max_deflection:.3f} mm")
            print(f"    Mean deflection: {stats.mean_deflection:.3f} mm")
            print(f"    Cost: ${stats.total_cost:.2f}")
            print(f"    Passes bending: {stats.all_pass_bending}")
            print(f"    Passes shear: {stats.all_pass_shear}")
    else:
        print(f"Analysis failed: {result['error']}")


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