from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from Pynite import FEModel3D
from Pynite.Rendering import Renderer
import numpy as np
import pandas as pd
from itertools import product
from Pynite.Rendering import Renderer


# CONSTANTS AND MATERIAL PROPERTIES

# Room and opening constants (units: mm, N, MPa)
ROOM_LENGTH = 3000
ROOM_WIDTH = 1870
ROOM_HEIGHT = 5465
PLANK_THICKNESS = 25
OPENING_WIDTH = 630
OPENING_LENGTH = 1420
WALL_BEAM_CONTACT_DEPTH = 40

floor2floor = ROOM_HEIGHT / 2 + PLANK_THICKNESS / 2
beam_length = ROOM_WIDTH + WALL_BEAM_CONTACT_DEPTH

# Material strength properties (MPa for stresses)
MATERIAL_STRENGTHS = {
    'wood': {'f_mk': 24, 'f_vk': 4.0, 'E': 11000, 'nu': 0.3, 'rho': 4.51e-6},
    'aluminum': {'f_mk': 160, 'f_vk': 90, 'E': 69000, 'nu': 0.33, 'rho': 2.7e-6},
    'steel': {'f_mk': 235, 'f_vk': 140, 'E': 200000, 'nu': 0.3, 'rho': 7.85e-6},
    'brick': {'f_mk': 10, 'f_vk': 1.0, 'E': 7000, 'nu': 0.2, 'rho': 5.75e-6}
}

# Beam catalog with all available profiles
BEAM_CATALOG = pd.DataFrame([
    {'id': 'W60x120', 'material': 'wood', 'base': 60, 'height': 120, 'shape': 'rectangular', 'cost_per_m3': 1648.95833, 'flange_width': None, 'flange_thickness': None, 'web_thickness': None},
    {'id': 'W120x120', 'material': 'wood', 'base': 120, 'height': 120, 'shape': 'rectangular', 'cost_per_m3': 3297.91666, 'flange_width': None, 'flange_thickness': None, 'web_thickness': None},
    {'id': 'W80x160', 'material': 'wood', 'base': 80, 'height': 160, 'shape': 'rectangular', 'cost_per_m3': 1358.39844, 'flange_width': None, 'flange_thickness': None, 'web_thickness': None},
    {'id': 'W160x160', 'material': 'wood', 'base': 160, 'height': 160, 'shape': 'rectangular', 'cost_per_m3': 2716.79688, 'flange_width': None, 'flange_thickness': None, 'web_thickness': None},
    
    {'id': 'IPE100', 'material': 'steel', 'base': 55, 'height': 100, 'shape': 'I-beam', 'cost_per_m3': 7850, 'flange_width': 55, 'flange_thickness': 5.7, 'web_thickness': 4.1},
    {'id': 'IPE120', 'material': 'steel', 'base': 64, 'height': 120, 'shape': 'I-beam', 'cost_per_m3': 7850, 'flange_width': 64, 'flange_thickness': 6.3, 'web_thickness': 4.4},
    {'id': 'AL80x40', 'material': 'aluminum', 'base': 40, 'height': 80, 'shape': 'I-beam', 'cost_per_m3': 2500, 'flange_width': 40, 'flange_thickness': 4, 'web_thickness': 2.5},
])


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
        new = BeamSpec(catalog_id=self.catalog_id, beam_type=self.beam_type, name=self.name)
        for k, v in kwargs.items():
            setattr(new, k, v)
        return new


@dataclass
class BeamPlacement:
    spec: BeamSpec
    x_center: float
    z_start: float = 0
    z_end: float = None

    def add_to_frame(self, frame: FEModel3D, floor2floor: float, default_z_end: float):
        z_end = self.z_end if self.z_end is not None else default_z_end
        name = self.spec.name

        if self.spec.beam_type != 'tail':
            frame.add_node(f'floor {name}N', self.x_center, 0, z_end)
        frame.add_node(f'floor {name}S', self.x_center, 0, self.z_start)
        frame.add_node(f'{name}S', self.x_center, floor2floor, self.z_start)
        
        if self.spec.beam_type != 'tail':
            frame.add_node(f'{name}N', self.x_center, floor2floor, z_end)
            self.spec.create_section(frame)
            frame.add_member(name, f'{name}N', f'{name}S', self.spec.material, self.spec.section_name)
        else:
            self.spec.create_section(frame)


@dataclass
class BeamGroupSpec:
    """Specification for a group of beams"""
    catalog_id: str
    quantity: int = 0
    padding: float = 0
    
    @property
    def catalog_data(self):
        return BEAM_CATALOG[BEAM_CATALOG['id'] == self.catalog_id].iloc[0]
    
    @property
    def base(self) -> float:
        return self.catalog_data['base']
    
    @property
    def height(self) -> float:
        return self.catalog_data['height']
    
    @property
    def material(self) -> str:
        return self.catalog_data['material']


# FLOOR PLAN CONFIGURATION

@dataclass
class FloorPlanHyperparameters:
    east_joists: BeamGroupSpec
    west_joists: BeamGroupSpec
    tail_joists: BeamGroupSpec
    trimmers: BeamGroupSpec
    header: BeamGroupSpec
    opening_x_start: float
    
    def validate(self):
        assert self.trimmers.quantity == 2, "Must have exactly 2 trimmers"
        assert self.header.quantity == 1, "Must have exactly 1 header"
        
        # Check padding doesn't exceed available space
        max_east_padding = self.opening_x_start - self.trimmers.base - self.east_joists.base
        assert self.east_joists.padding <= max_east_padding, \
            f"East padding {self.east_joists.padding} exceeds max {max_east_padding}"
        
        opening_x_end = self.opening_x_start + OPENING_LENGTH
        max_west_padding = ROOM_LENGTH - opening_x_end - self.trimmers.base - self.west_joists.base
        assert self.west_joists.padding <= max_west_padding, \
            f"West padding {self.west_joists.padding} exceeds max {max_west_padding}"


# GEOMETRY RESOLVER

class GeometryResolver:
    def __init__(self, params: FloorPlanHyperparameters):
        self.params = params
        params.validate()
        
        self.opening_x_start = params.opening_x_start
        self.opening_x_end = self.opening_x_start + OPENING_LENGTH
        
        self.trimmer_E_x_center = self.opening_x_start - (self.params.trimmers.base / 2)
        self.trimmer_W_x_center = self.opening_x_end + (self.params.trimmers.base / 2)
        
        self.tail_z_end = beam_length - OPENING_WIDTH - WALL_BEAM_CONTACT_DEPTH/2 - params.header.base/2
        self.header_z_pos = self.tail_z_end
        
        self.opening_z_start = OPENING_WIDTH + WALL_BEAM_CONTACT_DEPTH/2

    def _resolve_joist_positions(self, n: int, clear_start: float, clear_end: float, beam_base: float, group: str) -> List[float]:
        if group == 'east':
            centerline_start = clear_start + beam_base / 2
            centerline_end = clear_end + beam_base / 2
            positions = np.linspace(centerline_start, centerline_end, n+1).tolist()
            return positions[:-1]
        elif group == 'west':
            centerline_start = clear_start - beam_base / 2
            centerline_end = clear_end + beam_base / 2
            positions = np.linspace(centerline_end, centerline_start, n+1).tolist()
            return positions[:-1]
        elif group == 'tail':
            centerline_start = clear_start - beam_base / 2
            centerline_end = clear_end + beam_base / 2
            positions = np.linspace(centerline_start, centerline_end, n+2).tolist()
            return positions[1:-1]
        else:
            raise ValueError("Invalid group specified")

    def resolve_all_placements(self) -> Dict[str, List[BeamPlacement]]:
        placements = {
            'east': [],
            'west': [],
            'tail': [],
            'trimmers': []
        }
        
        # East joists
        if self.params.east_joists.quantity > 0:
            joist_spec_east = BeamSpec(self.params.east_joists.catalog_id, 'joist')
            clear_start = self.params.east_joists.padding
            clear_end = self.trimmer_E_x_center - self.params.trimmers.base / 2
            east_positions = self._resolve_joist_positions(
                self.params.east_joists.quantity, 
                clear_start, 
                clear_end, 
                joist_spec_east.base,
                "east"
            )
            for i, x in enumerate(east_positions):
                placements['east'].append(BeamPlacement(spec=joist_spec_east.copy(name=f'E{i}'), x_center=x))
        
        # Tail joists
        if self.params.tail_joists.quantity > 0:
            tail_spec = BeamSpec(self.params.tail_joists.catalog_id, 'tail')
            clear_start = self.opening_x_start + self.params.tail_joists.padding
            clear_end = self.opening_x_end - self.params.tail_joists.padding
            tail_positions = self._resolve_joist_positions(
                self.params.tail_joists.quantity, 
                clear_start, 
                clear_end, 
                tail_spec.base,
                "tail"
            )
            for i, x in enumerate(tail_positions):
                placements['tail'].append(BeamPlacement(spec=tail_spec.copy(name=f'T{i}'), x_center=x, z_end=self.tail_z_end))
        
        # West joists
        if self.params.west_joists.quantity > 0:
            joist_spec_west = BeamSpec(self.params.west_joists.catalog_id, 'joist')
            clear_start = self.trimmer_W_x_center + self.params.trimmers.base / 2
            clear_end = ROOM_LENGTH - self.params.west_joists.padding - self.params.west_joists.base
            west_positions = self._resolve_joist_positions(
                self.params.west_joists.quantity, 
                clear_start,
                clear_end,
                joist_spec_west.base,
                "west"
            )
            for i, x in enumerate(west_positions):
                placements['west'].append(BeamPlacement(spec=joist_spec_west.copy(name=f'W{i}'), x_center=x))

        # Trimmers
        trimmer_spec = BeamSpec(self.params.trimmers.catalog_id, 'trimmer')
        placements['trimmers'].append(BeamPlacement(spec=trimmer_spec.copy(name='trimmer_E'), x_center=self.trimmer_E_x_center))
        placements['trimmers'].append(BeamPlacement(spec=trimmer_spec.copy(name='trimmer_W'), x_center=self.trimmer_W_x_center))
        
        return placements


# LAYOUT MANAGER

class LayoutManager:
    def __init__(self, room_length: float):
        self.room_length = room_length
        self.beams: List[BeamPlacement] = []
        self._is_sorted = False

    def add_beams(self, placements: List[BeamPlacement]):
        self.beams.extend(placements)
        self._is_sorted = False

    def sort_beams(self):
        self.beams.sort(key=lambda p: p.x_center)
        self._is_sorted = True
    
    def apply_dead_loads(self, frame: FEModel3D):
        for p in self.beams:
            section_geom = p.spec.get_geometry()
            material = MATERIAL_STRENGTHS[p.spec.material]
            dead_load = -section_geom.A * material['rho']
            frame.add_member_dist_load(p.spec.name, 'FY', dead_load, dead_load)
    
    def apply_live_loads(self, frame: FEModel3D, live_load_mpa: float, opening_z_start: float):
        if not self._is_sorted or len(self.beams) < 2:
            return

        for i, p in enumerate(self.beams):
            # Calculate tributary boundaries
            left_boundary = self.beams[i-1].x_center if i > 0 else 0
            right_boundary = self.beams[i+1].x_center if i < len(self.beams) - 1 else self.room_length
            
            trib_left = (p.x_center - left_boundary) / 2
            trib_right = (right_boundary - p.x_center) / 2
            
            load_left = live_load_mpa * trib_left
            load_right = live_load_mpa * trib_right

            if p.spec.beam_type in ['joist', 'tail']:
                total_load = load_left + load_right
                frame.add_member_dist_load(p.spec.name, 'FY', total_load, total_load)
            elif p.spec.beam_type == 'trimmer':
                # For trimmers, one side has a load break at the opening
                is_left_opening = i > 0 and self.beams[i-1].spec.beam_type == 'tail'
                is_right_opening = i < len(self.beams)-1 and self.beams[i+1].spec.beam_type == 'tail'
                
                if is_left_opening:
                    frame.add_member_dist_load(p.spec.name, 'FY', load_left, load_left, 0, opening_z_start)
                    frame.add_member_dist_load(p.spec.name, 'FY', load_right, load_right)
                elif is_right_opening:
                    frame.add_member_dist_load(p.spec.name, 'FY', load_left, load_left)
                    frame.add_member_dist_load(p.spec.name, 'FY', load_right, load_right, 0, opening_z_start)
                else:
                    frame.add_member_dist_load(p.spec.name, 'FY', load_left + load_right, load_left + load_right)


# WALL GENERATION

def auto_add_walls(frame, layout, wall_thickness, material):
    layout.sort_beams()
    
    eastmost_beam = layout.beams[0]
    westmost_beam = layout.beams[-1]
    
    def node(floor, beam, end):
        return f"{'floor ' if floor else ''}{beam.spec.name}{end}"
    
    frame.add_quad(
        'west wall',
        node(True, westmost_beam, 'S'),
        node(True, westmost_beam, 'N'),
        node(False, westmost_beam, 'N'),
        node(False, westmost_beam, 'S'),
        wall_thickness,
        material
    )
    frame.add_quad(
        'east wall',
        node(True, eastmost_beam, 'N'),
        node(True, eastmost_beam, 'S'),
        node(False, eastmost_beam, 'S'),
        node(False, eastmost_beam, 'N'),
        wall_thickness,
        material
    )
    
    prev_beam = None
    for beam in layout.beams:
        if prev_beam is None:
            prev_beam = beam
            continue
        frame.add_quad(
            f'south wall {prev_beam.spec.name}-{beam.spec.name}',
            node(True, prev_beam, 'S'),
            node(True, beam, 'S'),
            node(False, beam, 'S'),
            node(False, prev_beam, 'S'),
            wall_thickness,
            material
        )
        prev_beam = beam
    
    north_reaching_beams = [b for b in layout.beams if b.spec.beam_type != 'tail']
    prev_beam = None
    for beam in north_reaching_beams:
        if prev_beam is None:
            prev_beam = beam
            continue
        frame.add_quad(
            f'north wall {prev_beam.spec.name}-{beam.spec.name}',
            node(True, prev_beam, 'N'),
            node(True, beam, 'N'),
            node(False, beam, 'N'),
            node(False, prev_beam, 'N'),
            wall_thickness,
            material
        )
        prev_beam = beam


# TELEMETRY CLASSES

@dataclass
class BeamTelemetry:
    """Telemetry data for a single beam"""
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
    """Aggregated telemetry for a group of beams"""
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
    beam_details: List[BeamTelemetry] = field(default_factory=list)


@dataclass
class SystemTelemetry:
    """Complete system telemetry"""
    max_header_deflection_mm: float
    max_deflection_overall_mm: float
    worst_member: str
    total_volume_m3: float
    total_cost: float
    system_passes: bool
    group_stats: Dict[str, GroupTelemetry] = field(default_factory=dict)
    beam_stats: List[BeamTelemetry] = field(default_factory=list)


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
                total_volume=0, total_cost=0, all_pass_bending=True, all_pass_shear=True
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
    
    def collect_system_telemetry(self, beam_groups: Dict[str, List[str]]) -> SystemTelemetry:
        """Collect telemetry for entire system"""
        group_stats = {}
        all_beam_stats = []
        
        for group_name, member_names in beam_groups.items():
            if not member_names:
                continue
            group_telem = self.collect_group_telemetry(group_name, member_names)
            group_stats[group_name] = group_telem
            all_beam_stats.extend(group_telem.beam_details)
        
        # Overall metrics
        max_deflection_overall = max((abs(b.min_deflection) for b in all_beam_stats), default=0)
        worst_member = max(all_beam_stats, key=lambda b: abs(b.min_deflection), default=None)
        
        total_volume = sum(g.total_volume for g in group_stats.values())
        total_cost = sum(g.total_cost for g in group_stats.values())
        
        system_passes = all(g.all_pass_bending and g.all_pass_shear for g in group_stats.values())
        
        header_deflection = abs(self.frame.members['header'].min_deflection('dy', 'Combo 1'))
        
        return SystemTelemetry(
            max_header_deflection_mm=header_deflection,
            max_deflection_overall_mm=max_deflection_overall,
            worst_member=worst_member.name if worst_member else '',
            total_volume_m3=total_volume / 1e9,
            total_cost=total_cost,
            system_passes=system_passes,
            group_stats=group_stats,
            beam_stats=all_beam_stats
        )


# MAIN ANALYSIS FUNCTION

def generate_and_analyze_floor(params: FloorPlanHyperparameters) -> tuple:
    """Generate floor model, run analysis, and return frame with results"""
    resolver = GeometryResolver(params)
    placement_groups = resolver.resolve_all_placements()
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
    layout = LayoutManager(room_length=ROOM_LENGTH)
    for group_placements in placement_groups.values():
        layout.add_beams(group_placements)
    layout.sort_beams()
    
    for beam_placement in layout.beams:
        beam_placement.add_to_frame(frame, floor2floor, beam_length)
    
    # Add header
    header_spec = BeamSpec(params.header.catalog_id, 'header', name='header')
    header_spec.create_section(frame)
    frame.add_node('header_E', resolver.trimmer_E_x_center, floor2floor, resolver.header_z_pos)
    frame.add_node('header_W', resolver.trimmer_W_x_center, floor2floor, resolver.header_z_pos)
    frame.add_member('header', 'header_W', 'header_E', header_spec.material, header_spec.section_name)
    
    # Connect tail joists to header
    for tail_placement in placement_groups['tail']:
        tail_node_name = f"{tail_placement.spec.name}_header"
        frame.add_node(tail_node_name, tail_placement.x_center, floor2floor, resolver.header_z_pos)
        frame.add_member(
            tail_placement.spec.name,
            tail_node_name,
            f"{tail_placement.spec.name}S",
            tail_placement.spec.material,
            tail_placement.spec.section_name
        )
    
    # Add walls and supports
    auto_add_walls(frame, layout, wall_thickness=80, material='brick')
    for node_name in frame.nodes:
        if node_name.startswith('floor'):
            frame.def_support(node_name, True, True, True, True, True, True)
    
    # Apply loads
    layout.apply_dead_loads(frame)
    header_geom = header_spec.get_geometry()
    header_material = MATERIAL_STRENGTHS[header_spec.material]
    header_dead_load = -header_geom.A * header_material['rho']
    frame.add_member_dist_load('header', 'FY', header_dead_load, header_dead_load)
    layout.apply_live_loads(frame, live_load_mpa=-0.003, opening_z_start=resolver.opening_z_start)
    
    # Run analysis
    frame.analyze(check_statics=True)
    
    # Build beam spec map for telemetry
    beam_spec_map = {}
    for beam_placement in layout.beams:
        beam_spec_map[beam_placement.spec.name] = beam_placement.spec
    beam_spec_map['header'] = header_spec
    
    return frame, resolver, beam_spec_map


# OPTIMIZATION FRAMEWORK

class FloorOptimizer:
    """Handles optimization and telemetry collection"""
    
    def __init__(self):
        self.results = []
        
    def run_single_configuration(self, params: FloorPlanHyperparameters, config_id: int = None) -> Dict:
        """Run analysis for a single configuration and collect telemetry"""
        try:
            frame, resolver, beam_spec_map = generate_and_analyze_floor(params)
            
            # Collect telemetry
            collector = TelemetryCollector(frame, beam_spec_map)
            
            # Define beam groups
            beam_groups = {
                'east_joists': [f'E{i}' for i in range(params.east_joists.quantity)],
                'west_joists': [f'W{i}' for i in range(params.west_joists.quantity)],
                'tail_joists': [f'T{i}' for i in range(params.tail_joists.quantity)],
                'trimmers': ['trimmer_E', 'trimmer_W'],
                'header': ['header']
            }
            
            telemetry = collector.collect_system_telemetry(beam_groups)
            
            return frame, {
                'config_id': config_id,
                'params': params,
                'telemetry': telemetry,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return frame, {
                'config_id': config_id,
                'params': params,
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
        
        print(f"Running grid search over {len(combinations)} configurations...")
        
        for i, combo in enumerate(combinations):
            config_params = dict(zip(keys, combo))
            
            # Build configuration
            params = FloorPlanHyperparameters(
                east_joists=BeamGroupSpec(
                    catalog_id=config_params['east_joist_catalog_id'],
                    quantity=config_params['east_quantity'],
                    padding=config_params.get('east_padding', 0)
                ),
                west_joists=BeamGroupSpec(
                    catalog_id=config_params['west_joist_catalog_id'],
                    quantity=config_params['west_quantity'],
                    padding=config_params.get('west_padding', 0)
                ),
                tail_joists=BeamGroupSpec(
                    catalog_id=config_params['tail_joist_catalog_id'],
                    quantity=config_params['tail_quantity'],
                    padding=config_params.get('tail_padding', 0)
                ),
                trimmers=BeamGroupSpec(
                    catalog_id=config_params['trimmer_catalog_id'],
                    quantity=2
                ),
                header=BeamGroupSpec(
                    catalog_id=config_params['header_catalog_id'],
                    quantity=1
                ),
                opening_x_start=config_params['opening_x_start']
            )
            
            frame, result = self.run_single_configuration(params, config_id=i)
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
            
            params = result['params']
            telem = result['telemetry']
            
            row = {
                # Configuration
                'config_id': result['config_id'],
                'east_catalog_id': params.east_joists.catalog_id,
                'east_quantity': params.east_joists.quantity,
                'west_catalog_id': params.west_joists.catalog_id,
                'west_quantity': params.west_joists.quantity,
                'tail_catalog_id': params.tail_joists.catalog_id,
                'tail_quantity': params.tail_joists.quantity,
                'trimmer_catalog_id': params.trimmers.catalog_id,
                'header_catalog_id': params.header.catalog_id,
                'opening_x_start': params.opening_x_start,
                
                # Overall metrics
                'max_deflection_mm': telem.max_deflection_overall_mm,
                'header_deflection_mm': telem.max_header_deflection_mm,
                'worst_member': telem.worst_member,
                'total_volume_m3': telem.total_volume_m3,
                'total_cost': telem.total_cost,
                'system_passes': telem.system_passes,
                
                # Group-level metrics
                **{f'{group}_max_deflection': stats.max_deflection 
                   for group, stats in telem.group_stats.items()},
                **{f'{group}_mean_deflection': stats.mean_deflection 
                   for group, stats in telem.group_stats.items()},
                **{f'{group}_max_moment': stats.max_moment 
                   for group, stats in telem.group_stats.items()},
                **{f'{group}_passes_bending': stats.all_pass_bending 
                   for group, stats in telem.group_stats.items()},
                **{f'{group}_passes_shear': stats.all_pass_shear 
                   for group, stats in telem.group_stats.items()},
                **{f'{group}_total_cost': stats.total_cost 
                   for group, stats in telem.group_stats.items()},
                
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


# EXAMPLE USAGE

if __name__ == '__main__':
    # Example 1: Single configuration analysis
    print("=" * 70)
    print("EXAMPLE 1: Single Configuration Analysis")
    print("=" * 70)
    
    single_config = FloorPlanHyperparameters(
        east_joists=BeamGroupSpec(catalog_id='W60x120', quantity=1, padding=0),
        west_joists=BeamGroupSpec(catalog_id='W60x120', quantity=1, padding=0),
        tail_joists=BeamGroupSpec(catalog_id='W60x120', quantity=1, padding=0),
        trimmers=BeamGroupSpec(catalog_id='W80x160', quantity=2),
        header=BeamGroupSpec(catalog_id='W60x120', quantity=1),
        opening_x_start=900
    )
    
    optimizer = FloorOptimizer()
    frame, result = optimizer.run_single_configuration(single_config, config_id=0)
    
    if result['success']:
        telem = result['telemetry']
        print(f"\nSystem passes all checks: {telem.system_passes}")
        print(f"Total cost: ${telem.total_cost:.2f}")
        print(f"Total volume: {telem.total_volume_m3:.4f} m³")
        print(f"Max deflection: {telem.max_deflection_overall_mm:.3f} mm")
        print(f"Header deflection: {telem.max_header_deflection_mm:.3f} mm")
        print(f"Worst member: {telem.worst_member}")
        
        print("\nGroup Statistics:")
        for group_name, stats in telem.group_stats.items():
            print(f"\n  {group_name}:")
            print(f"    Count: {stats.count}")
            print(f"    Max deflection: {stats.max_deflection:.3f} mm")
            print(f"    Mean deflection: {stats.mean_deflection:.3f} mm")
            print(f"    Cost: ${stats.total_cost:.2f}")
            print(f"    Passes bending: {stats.all_pass_bending}")
            print(f"    Passes shear: {stats.all_pass_shear}")
    else:
        print(f"Analysis failed: {result['error']}")
    
    # Example 2: Grid search
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Grid Search Optimization")
    print("=" * 70)
    
    param_space = {
        'east_joist_catalog_id': ['W60x120', 'W80x160'],
        'east_quantity': [0, 1, 2, 3],
        'west_joist_catalog_id': ['W60x120', 'W80x160'],
        'west_quantity': [0, 1, 2, 3],
        'tail_joist_catalog_id': ['W60x120', 'W80x160'],
        'tail_quantity': [0, 1, 2, 3],
        'trimmer_catalog_id': ['W60x120', 'W80x160', 'W120x120', 'W160x160'],
        'header_catalog_id': ['W60x120', 'W80x160', 'W120x120', 'W160x160'],
        'opening_x_start': [820],
        'east_padding': list(range(0, 200, 10)),
        'west_padding': list(range(0, 200, 10)),
        'tail_padding': list(range(0, 200, 10)),
    }
    
    optimizer = FloorOptimizer()
    results_df = optimizer.run_grid_search(param_space)
    
    # Save results
    results_df.to_csv('floor_optimization_results.csv', index=False)
    print(f"\nResults saved to 'floor_optimization_results.csv'")