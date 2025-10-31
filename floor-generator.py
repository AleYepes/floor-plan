from dataclasses import dataclass
from typing import List, Dict
from Pynite import FEModel3D
import numpy as np

# Room and opening constants (units: mm, N, MPa)
ROOM_WIDTH = 3000
ROOM_LENGTH = 1870
ROOM_HEIGHT = 5465
PLANK_THICKNESS = 25
OPENING_CLEAR_WIDTH = 630  # Clear span between trimmer inner faces
OPENING_CLEAR_DEPTH = 1420  # Clear span from south wall to header
OPENING_X_START = 820  # X-position of east trimmer's WEST face (inner face)

floor2floor = ROOM_HEIGHT / 2 + PLANK_THICKNESS / 2
wall_beam_contact_depth = 40
beam_length = ROOM_LENGTH + wall_beam_contact_depth

# Material properties database
MATERIALS = {
    'wood': {'E': 11000, 'nu': 0.3, 'rho': 4.51e-6},
    'aluminum': {'E': 69000, 'nu': 0.33, 'rho': 2.7e-6},
    'steel': {'E': 200000, 'nu': 0.3, 'rho': 7.85e-6}
}

@dataclass
class BeamSpec:
    template_name: str
    base: int
    height: int
    material: str
    beam_type: str
    name: str = ''

    @property
    def section_name(self) -> str:
        return f"sec_{self.base}x{self.height}"

    def create_section(self, frame: FEModel3D):
        if self.section_name not in frame.sections:
            A = self.base * self.height
            b, h = min(self.base, self.height), max(self.base, self.height)
            J = (b**3 * h) * (1/3 - 0.21 * (b/h) * (1 - (b**4)/(12*h**4)))
            Iy = (self.height * self.base**3) / 12
            Iz = (self.base * self.height**3) / 12
            frame.add_section(self.section_name, A, Iy, Iz, J)

    def copy(self, **kwargs):
        new = BeamSpec(**self.__dict__)
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

        # Create floor nodes (supports)
        frame.add_node(f'floor {name}S', self.x_center, 0, self.z_start)
        frame.add_node(f'floor {name}N', self.x_center, 0, z_end)
        
        # Create upper nodes
        frame.add_node(f'{name}S', self.x_center, floor2floor, self.z_start)
        
        if self.spec.beam_type != 'tail':
            frame.add_node(f'{name}N', self.x_center, floor2floor, z_end)
            self.spec.create_section(frame)
            frame.add_member(name, f'{name}N', f'{name}S', self.spec.material, self.spec.section_name)
        else:
            # Tail joists end at header - node will be created when header is added
            self.spec.create_section(frame)
            # Member will be created after header nodes exist


@dataclass
class BeamGroupSpec:
    base: int
    height: int
    material: str
    quantity: int = 0
    padding: float = 0  # Clear distance from wall to beam face


@dataclass
class FloorPlanHyperparameters:
    east_joists: BeamGroupSpec
    west_joists: BeamGroupSpec
    tail_joists: BeamGroupSpec
    trimmers: BeamGroupSpec
    header: BeamGroupSpec
    
    def validate(self):
        """Validate hyperparameter constraints"""
        assert self.trimmers.quantity == 2, "Must have exactly 2 trimmers"
        assert self.header.quantity == 1, "Must have exactly 1 header"
        
        # Check padding doesn't exceed available space
        max_east_padding = OPENING_X_START - self.trimmers.base - self.east_joists.base
        assert self.east_joists.padding <= max_east_padding, \
            f"East padding {self.east_joists.padding} exceeds max {max_east_padding}"
        
        opening_x_end = OPENING_X_START + OPENING_CLEAR_WIDTH
        max_west_padding = ROOM_WIDTH - opening_x_end - self.trimmers.base - self.west_joists.base
        assert self.west_joists.padding <= max_west_padding, \
            f"West padding {self.west_joists.padding} exceeds max {max_west_padding}"


class GeometryResolver:
    def __init__(self, params: FloorPlanHyperparameters):
        self.params = params
        params.validate()
        
        # Opening geometry - inner faces of trimmers define the clear opening
        self.opening_x_start = OPENING_X_START  # East trimmer west face
        self.opening_x_end = OPENING_X_START + OPENING_CLEAR_WIDTH  # West trimmer east face
        
        # Trimmer centerlines: offset by half their width from their inner faces
        self.trimmer_E_x_center = self.opening_x_start - (self.params.trimmers.base / 2)
        self.trimmer_W_x_center = self.opening_x_end + (self.params.trimmers.base / 2)
        
        # Header position (north end of opening)
        self.header_z_pos = beam_length - OPENING_CLEAR_DEPTH - wall_beam_contact_depth / 2

    def _resolve_joist_positions(self, n: int, clear_start: float, clear_end: float, 
                                  beam_base: float) -> List[float]:
        """
        Calculate centerline positions for n joists in a clear span.
        clear_start and clear_end define the boundaries (walls or adjacent beam faces).
        """
        if n == 0:
            return []
        
        # Convert clear span to centerline span
        centerline_start = clear_start + beam_base / 2
        centerline_end = clear_end - beam_base / 2
        
        if n == 1:
            return [(centerline_start + centerline_end) / 2]
        
        return np.linspace(centerline_start, centerline_end, n).tolist()

    def resolve_all_placements(self) -> Dict[str, List[BeamPlacement]]:
        """Returns dict with keys: 'east', 'west', 'tail', 'trimmers' for organized access"""
        placements = {
            'east': [],
            'west': [],
            'tail': [],
            'trimmers': []
        }
        
        # East joists: from padding to east trimmer's east face
        if self.params.east_joists.quantity > 0:
            joist_spec_east = BeamSpec('joist', self.params.east_joists.base, 
                                      self.params.east_joists.height, 
                                      self.params.east_joists.material, 'joist')
            
            clear_start = self.params.east_joists.padding
            clear_end = self.trimmer_E_x_center - self.params.trimmers.base / 2
            
            east_positions = self._resolve_joist_positions(
                self.params.east_joists.quantity, clear_start, clear_end, 
                joist_spec_east.base
            )
            
            for i, x in enumerate(east_positions):
                placements['east'].append(
                    BeamPlacement(spec=joist_spec_east.copy(name=f'E{i}'), x_center=x)
                )
        
        # Trimmers
        trimmer_spec = BeamSpec('trimmer', self.params.trimmers.base, 
                               self.params.trimmers.height, 
                               self.params.trimmers.material, 'trimmer')
        
        placements['trimmers'].append(
            BeamPlacement(spec=trimmer_spec.copy(name='trimmer_E'), x_center=self.trimmer_E_x_center)
        )
        placements['trimmers'].append(
            BeamPlacement(spec=trimmer_spec.copy(name='trimmer_W'), x_center=self.trimmer_W_x_center)
        )
        
        # Tail joists: between trimmer inner faces
        if self.params.tail_joists.quantity > 0:
            tail_spec = BeamSpec('tail', self.params.tail_joists.base, 
                               self.params.tail_joists.height, 
                               self.params.tail_joists.material, 'tail')
            
            clear_start = self.opening_x_start
            clear_end = self.opening_x_end
            
            tail_positions = self._resolve_joist_positions(
                self.params.tail_joists.quantity, clear_start, clear_end, 
                tail_spec.base
            )
            
            for i, x in enumerate(tail_positions):
                placements['tail'].append(
                    BeamPlacement(spec=tail_spec.copy(name=f'T{i}'), x_center=x, 
                                z_end=self.header_z_pos)
                )
        
        # West joists: from west trimmer's west face to padding
        if self.params.west_joists.quantity > 0:
            joist_spec_west = BeamSpec('joist', self.params.west_joists.base, 
                                      self.params.west_joists.height, 
                                      self.params.west_joists.material, 'joist')
            
            clear_start = self.trimmer_W_x_center + self.params.trimmers.base / 2
            clear_end = ROOM_WIDTH - self.params.west_joists.padding
            
            west_positions = self._resolve_joist_positions(
                self.params.west_joists.quantity, clear_start, clear_end, 
                joist_spec_west.base
            )
            
            for i, x in enumerate(west_positions):
                placements['west'].append(
                    BeamPlacement(spec=joist_spec_west.copy(name=f'W{i}'), x_center=x)
                )
        
        return placements


class LayoutManager:
    def __init__(self, room_width: float):
        self.room_width = room_width
        self.beams: List[BeamPlacement] = []
        self._is_sorted = False

    def add_beams(self, placements: List[BeamPlacement]):
        """Add multiple beam placements"""
        self.beams.extend(placements)
        self._is_sorted = False

    def sort_beams(self):
        self.beams.sort(key=lambda p: p.x_center)
        self._is_sorted = True
    
    def apply_dead_loads(self, frame: FEModel3D):
        """Apply self-weight dead loads to all members"""
        for p in self.beams:
            section = frame.sections[p.spec.section_name]
            material = frame.materials[p.spec.material]
            dead_load = -section.A * material.rho
            frame.add_member_dist_load(p.spec.name, 'FY', dead_load, dead_load)
    
    def apply_live_loads(self, frame: FEModel3D, live_load_mpa: float):
        """Apply tributary area live loads to joists and tail joists"""
        if not self._is_sorted or len(self.beams) < 2:
            return

        for i, p in enumerate(self.beams):
            # Skip headers and trimmers - they get loads from reactions
            if p.spec.beam_type in ['header', 'trimmer']:
                continue
            
            # Calculate tributary boundaries
            left_boundary = self.beams[i-1].x_center if i > 0 else 0
            right_boundary = self.beams[i+1].x_center if i < len(self.beams) - 1 else self.room_width
            
            trib_left = (p.x_center - left_boundary) / 2
            trib_right = (right_boundary - p.x_center) / 2
            total_trib_width = trib_left + trib_right
            
            load = -live_load_mpa * total_trib_width
            frame.add_member_dist_load(p.spec.name, 'FY', load, load)


def generate_and_analyze_floor(params: FloorPlanHyperparameters) -> Dict:
    """
    Main function to generate floor model and run FEA analysis.
    Returns dict with analysis results.
    """
    resolver = GeometryResolver(params)
    placement_groups = resolver.resolve_all_placements()
    
    # Initialize FE model
    frame = FEModel3D()
    
    # Add all materials
    for mat_name, props in MATERIALS.items():
        frame.add_material(
            mat_name, 
            E=props['E'], 
            G=props['E'] / (2 * (1 + props['nu'])), 
            nu=props['nu'], 
            rho=props['rho']
        )
    
    # Create layout and add all beams
    layout = LayoutManager(room_width=ROOM_WIDTH)
    for group_placements in placement_groups.values():
        layout.add_beams(group_placements)
    
    layout.sort_beams()
    
    # Add beams to frame (creates nodes and members)
    for beam_placement in layout.beams:
        beam_placement.add_to_frame(frame, floor2floor, beam_length)
    
    # Add header with nodes at trimmer positions
    header_spec = BeamSpec('header', params.header.base, params.header.height, 
                          params.header.material, 'header', name='header')
    header_spec.create_section(frame)
    
    frame.add_node('header_E', resolver.trimmer_E_x_center, floor2floor, resolver.header_z_pos)
    frame.add_node('header_W', resolver.trimmer_W_x_center, floor2floor, resolver.header_z_pos)
    frame.add_member('header', 'header_W', 'header_E', header_spec.material, header_spec.section_name)
    
    # Now create tail joist members that connect to header
    for tail_placement in placement_groups['tail']:
        tail_node_name = f"{tail_placement.spec.name}_header"
        # Node at tail joist x-position, on the header
        frame.add_node(tail_node_name, tail_placement.x_center, floor2floor, resolver.header_z_pos)
        frame.add_member(
            tail_placement.spec.name,
            tail_node_name,
            f"{tail_placement.spec.name}S",
            tail_placement.spec.material,
            tail_placement.spec.section_name
        )
    
    # Define supports at all floor nodes
    for node_name in frame.nodes:
        if node_name.startswith('floor'):
            frame.def_support(node_name, True, True, True, True, True, True)
    
    # Apply loads
    layout.apply_dead_loads(frame)
    
    # Header dead load
    header_section = frame.sections[header_spec.section_name]
    header_material = frame.materials[header_spec.material]
    header_dead_load = -header_section.A * header_material.rho
    frame.add_member_dist_load('header', 'FY', header_dead_load, header_dead_load)
    
    layout.apply_live_loads(frame, live_load_mpa=0.003)
    
    # Analyze
    frame.analyze(check_statics=True)
    
    # Extract results
    max_header_deflection = abs(min(frame.members['header'].deflection['uy']))
    
    total_mass = sum(
        m.L * frame.sections[m.section_name].A * frame.materials[m.material_name].rho 
        for m in frame.members.values()
    )
    
    # Find maximum deflection across all members
    max_deflection_overall = 0
    worst_member = None
    for member_name, member in frame.members.items():
        deflection = abs(min(member.deflection['uy']))
        if deflection > max_deflection_overall:
            max_deflection_overall = deflection
            worst_member = member_name
    
    return {
        'max_header_deflection_mm': max_header_deflection,
        'max_deflection_overall_mm': max_deflection_overall,
        'worst_member': worst_member,
        'total_mass_kg': total_mass,
        'frame': frame,
        'resolver': resolver
    }


if __name__ == '__main__':
    # Test with default configuration
    default_plan = FloorPlanHyperparameters(
        east_joists=BeamGroupSpec(base=60, height=120, material='wood', quantity=2, padding=50),
        west_joists=BeamGroupSpec(base=60, height=120, material='wood', quantity=1, padding=50),
        tail_joists=BeamGroupSpec(base=60, height=120, material='wood', quantity=3),
        trimmers=BeamGroupSpec(base=80, height=160, material='wood', quantity=2),
        header=BeamGroupSpec(base=120, height=120, material='wood', quantity=1)
    )
    
    results = generate_and_analyze_floor(default_plan)
    
    print(f"Analysis Complete:")
    print(f"  Max Header Deflection: {results['max_header_deflection_mm']:.3f} mm")
    print(f"  Max Overall Deflection: {results['max_deflection_overall_mm']:.3f} mm (at {results['worst_member']})")
    print(f"  Total Floor Mass: {results['total_mass_kg']:.2f} kg")
    print(f"\nOpening verification:")
    print(f"  East trimmer center: {results['resolver'].trimmer_E_x_center:.1f} mm")
    print(f"  West trimmer center: {results['resolver'].trimmer_W_x_center:.1f} mm")
    print(f"  Clear opening width: {results['resolver'].opening_x_end - results['resolver'].opening_x_start:.1f} mm")