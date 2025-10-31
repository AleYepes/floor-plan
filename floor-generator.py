from dataclasses import dataclass
from typing import List, Dict
from Pynite import FEModel3D
import numpy as np
from Pynite.Rendering import Renderer

# Room and opening constants (units: mm, N, MPa)
ROOM_LENGTH = 3000
ROOM_wid = 1870
ROOM_HEIGHT = 5465
PLANK_THICKNESS = 25
OPENING_WIDTH = 630
OPENING_LENGTH = 1420 # Clear span between trimmer inner faces
WALL_BEAM_CONTACT_DEPTH = 40

floor2floor = ROOM_HEIGHT / 2 + PLANK_THICKNESS / 2
beam_length = ROOM_wid + WALL_BEAM_CONTACT_DEPTH

# Material properties database
MATERIALS = {
    'wood': {'E': 11000, 'nu': 0.3, 'rho': 4.51e-6},
    'aluminum': {'E': 69000, 'nu': 0.33, 'rho': 2.7e-6},
    'steel': {'E': 200000, 'nu': 0.3, 'rho': 7.85e-6},
    'brick': {'E': 7000, 'nu': 0.2, 'rho': 5.75e-6}
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
    opening_x_start: float  # X-position where opening starts (east trimmer INNER face)
    
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
            joist_spec_east = BeamSpec('joist', self.params.east_joists.base, self.params.east_joists.height, self.params.east_joists.material, 'joist')
            clear_start = self.params.east_joists.padding
            clear_end = self.trimmer_E_x_center - self.params.trimmers.base / 2
            east_positions = self._resolve_joist_positions(self.params.east_joists.quantity, 
                                                           clear_start, 
                                                           clear_end, 
                                                           joist_spec_east.base,
                                                           "east"
                                                           )
            for i, x in enumerate(east_positions):
                placements['east'].append(BeamPlacement(spec=joist_spec_east.copy(name=f'E{i}'), x_center=x))
        
        # Tail joints
        if self.params.tail_joists.quantity > 0:
            tail_spec = BeamSpec('tail', self.params.tail_joists.base, self.params.tail_joists.height, self.params.tail_joists.material, 'tail')
            clear_start = self.opening_x_start + self.params.tail_joists.padding
            clear_end = self.opening_x_end - self.params.tail_joists.padding
            tail_positions = self._resolve_joist_positions(self.params.tail_joists.quantity, 
                                                           clear_start, 
                                                           clear_end, 
                                                           tail_spec.base,
                                                           "tail"
                                                           )
            for i, x in enumerate(tail_positions):
                placements['tail'].append(BeamPlacement(spec=tail_spec.copy(name=f'T{i}'), x_center=x, z_end=self.tail_z_end))
        
        # West joists
        if self.params.west_joists.quantity > 0:
            joist_spec_west = BeamSpec('joist', self.params.west_joists.base, self.params.west_joists.height, self.params.west_joists.material, 'joist')
            clear_start = self.trimmer_W_x_center + self.params.trimmers.base / 2
            clear_end = ROOM_LENGTH - self.params.west_joists.padding
            west_positions = self._resolve_joist_positions(self.params.west_joists.quantity, 
                                                           clear_start,
                                                           clear_end,
                                                           joist_spec_west.base,
                                                           "west"
                                                           )
            for i, x in enumerate(west_positions):
                placements['west'].append(BeamPlacement(spec=joist_spec_west.copy(name=f'W{i}'), x_center=x))

        # Trimmers
        trimmer_spec = BeamSpec('trimmer', self.params.trimmers.base, self.params.trimmers.height, self.params.trimmers.material, 'trimmer')
        placements['trimmers'].append(BeamPlacement(spec=trimmer_spec.copy(name='trimmer_E'), x_center=self.trimmer_E_x_center))
        placements['trimmers'].append(BeamPlacement(spec=trimmer_spec.copy(name='trimmer_W'), x_center=self.trimmer_W_x_center))
        
        return placements


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
            section = frame.sections[p.spec.section_name]
            material = frame.materials[p.spec.material]
            dead_load = -section.A * material.rho
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


def set_wall_opacity(plotter, opacity=0.5):
    for actor in plotter.renderer.actors.values():
        if (hasattr(actor, 'mapper') and
            hasattr(actor.mapper, 'dataset') and
            actor.mapper.dataset.n_faces_strict > 0):
            actor.prop.opacity = opacity


def generate_and_analyze_floor(params: FloorPlanHyperparameters) -> tuple:
    resolver = GeometryResolver(params)
    placement_groups = resolver.resolve_all_placements()
    frame = FEModel3D()
    
    for mat_name, props in MATERIALS.items():
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
    
    header_spec = BeamSpec('header', params.header.base, params.header.height, params.header.material, 'header', name='header')
    header_spec.create_section(frame)
    frame.add_node('header_E', resolver.trimmer_E_x_center, floor2floor, resolver.header_z_pos)
    frame.add_node('header_W', resolver.trimmer_W_x_center, floor2floor, resolver.header_z_pos)
    frame.add_member('header', 'header_W', 'header_E', header_spec.material, header_spec.section_name)
    
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
    
    auto_add_walls(frame, layout, wall_thickness=80, material='brick')
    for node_name in frame.nodes:
        if node_name.startswith('floor'):
            frame.def_support(node_name, True, True, True, True, True, True)
    
    layout.apply_dead_loads(frame)
    header_section = frame.sections[header_spec.section_name]
    header_material = frame.materials[header_spec.material]
    header_dead_load = -header_section.A * header_material.rho
    frame.add_member_dist_load('header', 'FY', header_dead_load, header_dead_load)
    
    # FIXED: Use negative live load and pass opening_z_start
    layout.apply_live_loads(frame, live_load_mpa=-0.003, opening_z_start=resolver.opening_z_start)
    
    frame.analyze(check_statics=True)
    max_header_deflection = abs(frame.members['header'].min_deflection('dy', 'Combo 1'))
    
    total_mass = sum(
        m.L() * m.section.A * m.material.rho
        for m in frame.members.values()
    )
    
    max_deflection_overall = 0
    worst_member = None
    for member_name, member in frame.members.items():
        deflection = abs(member.min_deflection('dy', 'Combo 1'))
        if deflection > max_deflection_overall:
            max_deflection_overall = deflection
            worst_member = member_name
    
    return frame, {
        'max_header_deflection_mm': max_header_deflection,
        'max_deflection_overall_mm': max_deflection_overall,
        'worst_member': worst_member,
        'total_mass_kg': total_mass,
        'resolver': resolver
    }


if __name__ == '__main__':
    default_plan = FloorPlanHyperparameters(
        east_joists=BeamGroupSpec(base=60, height=120, material='wood', quantity=2, padding=0),
        west_joists=BeamGroupSpec(base=60, height=120, material='wood', quantity=2, padding=0),
        tail_joists=BeamGroupSpec(base=60, height=120, material='wood', quantity=2, padding=0),
        trimmers=BeamGroupSpec(base=80, height=160, material='wood', quantity=2),
        header=BeamGroupSpec(base=120, height=120, material='wood', quantity=1),
        opening_x_start=820
    )
    
    frame, results = generate_and_analyze_floor(default_plan)
    
    print(f"Analysis Complete:")
    print(f"  Max Header Deflection: {results['max_header_deflection_mm']:.3f} mm")
    print(f"  Max Overall Deflection: {results['max_deflection_overall_mm']:.3f} mm (at {results['worst_member']})")
    print(f"  Total Floor Mass: {results['total_mass_kg']:.2f} kg")
    print(f"\nGeometry verification:")
    print(f"  East trimmer center: {results['resolver'].trimmer_E_x_center:.1f} mm")
    print(f"  West trimmer center: {results['resolver'].trimmer_W_x_center:.1f} mm")
    print(f"  Clear opening width: {results['resolver'].opening_x_end - results['resolver'].opening_x_start:.1f} mm")
    print(f"  Tail joist end (header) Z: {results['resolver'].tail_z_end:.1f} mm")
    print(f"  Beam length: {beam_length:.1f} mm")
    
    # Print member statistics
    for beam in frame.members:
        print(f"\n--- {beam} Stats ---")
        print(f"Max Moment (Mz): {frame.members[beam].max_moment('Mz', 'Combo 1'):.3f} N-mm")
        print(f"Min Moment (Mz): {frame.members[beam].min_moment('Mz', 'Combo 1'):.3f} N-mm")
        print(f"Max Deflection (dy): {frame.members[beam].max_deflection('dy', 'Combo 1'):.3f} mm")
        print(f"Min Deflection (dy): {frame.members[beam].min_deflection('dy', 'Combo 1'):.3f} mm")

    # Render
    rndr = Renderer(frame)
    rndr.annotation_size = 5
    rndr.render_loads = False
    rndr.deformed_shape = True
    rndr.deformed_scale = 1000
    opacity = 0.25
    rndr.post_update_callbacks.append(lambda plotter: set_wall_opacity(plotter, opacity=opacity))
    rndr.render_model()