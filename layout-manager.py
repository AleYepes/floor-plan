# Units are mm, N, and MPa (N/mm²)
from dataclasses import dataclass
from typing import List
from Pynite import FEModel3D
from Pynite.Rendering import Renderer


ROOM_WIDTH = 3000  # Along X-axis
ROOM_LENGTH = 1870 # Along Z-axis
ROOM_HEIGHT = 5465 # Along Y-axis
PLANK_THICKNESS = 25

floor2floor = ROOM_HEIGHT/2 + PLANK_THICKNESS/2
wall_beam_contact_depth = 40
beam_length = ROOM_LENGTH + wall_beam_contact_depth

opening_width = 630 # Along the Z-axis
opening_z_start = opening_width + wall_beam_contact_depth/2

frame = FEModel3D()
E = 11000
nu = 0.3
rho = 4.51e-6
frame.add_material('wood', E=E, G=(E / (2 * (1 + nu))), nu=nu, rho=rho)

E = 7000
nu = 0.2
rho = 5.75e-6
frame.add_material('brick', E=E, G=(E / (2 * (1 + nu))), nu=nu, rho=rho)


@dataclass
class BeamSpec:
    template_name: str
    base: int
    height: int
    material: str
    beam_type: str  # 'joist', 'trimmer', 'tail', 'header'
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
        
        if not name.startswith('tail'):
            frame.add_node(f'floor {name}N', self.x_center, 0, z_end)
        frame.add_node(f'floor {name}S', self.x_center, 0, self.z_start)
        frame.add_node(f'{name}S', self.x_center, floor2floor, self.z_start)
        frame.add_node(f'{name}N', self.x_center, floor2floor, z_end)
        
        self.spec.create_section(frame)
        frame.add_member(name, f'{name}N', f'{name}S', self.spec.material, self.spec.section_name)

class LayoutManager:
    def __init__(self, room_width: float):
        self.room_width = room_width
        self.beams: List[BeamPlacement] = []
        self._is_sorted = False

    def add_beam(self, spec: BeamSpec, x_center: float, z_start: float = 0, z_end: float = None) -> BeamPlacement:
        placement = BeamPlacement(spec, x_center, z_start, z_end)
        self.beams.append(placement)
        self._is_sorted = False
        return placement

    def add_beam_at_offset(self, spec: BeamSpec, x_offset: float, **kwargs) -> BeamPlacement:
        """
        Adds a beam based on the clear offset from the east wall (x=0) to its nearest face.
        
        Example: An offset of 820mm for an 80mm wide beam places its east face at
        x=820 and its centerline at x=860.
        """
        x_center = x_offset + spec.base / 2
        return self.add_beam(spec, x_center, **kwargs)

    def add_beam_between(self, spec: BeamSpec, east_beam: BeamPlacement, west_beam: BeamPlacement, **kwargs) -> BeamPlacement:
        east_beam_west_edge = east_beam.x_center + east_beam.spec.base / 2
        west_beam_east_edge = west_beam.x_center - west_beam.spec.base / 2
        
        clear_span = west_beam_east_edge - east_beam_west_edge
        if spec.base > clear_span:
            raise ValueError(f"Cannot place beam '{spec.name}' ({spec.base}mm wide) in a clear span of only {clear_span:.1f}mm.")
            
        x_center = (east_beam_west_edge + west_beam_east_edge) / 2
        return self.add_beam(spec, x_center, **kwargs)

    def sort_beams(self):
        self.beams.sort(key=lambda p: p.x_center)
        self._is_sorted = True
    
    def apply_dead_loads(self, frame: FEModel3D):
        for p in self.beams:
            section = frame.sections[p.spec.section_name]
            material = frame.materials[p.spec.material]
            dead_load = -section.A * material.rho
            frame.add_member_dist_load(p.spec.name, 'FY', dead_load, dead_load)

    def apply_live_loads(self, frame: FEModel3D, live_load_mpa: float, opening_z: float):
        if not self._is_sorted:
            raise RuntimeError("LayoutManager must be finalized before applying loads.")
            
        for i, beam_placement in enumerate(self.beams):
            pos_left = self.beams[i-1].x_center if i > 0 else 0
            pos_right = self.beams[i+1].x_center if i < len(self.beams)-1 else self.room_width
            
            trib_width_left = (beam_placement.x_center - pos_left) / 2
            trib_width_right = (pos_right - beam_placement.x_center) / 2
            
            load_left = live_load_mpa * trib_width_left
            load_right = live_load_mpa * trib_width_right

            if beam_placement.spec.beam_type in ['joist', 'tail']:
                total_load = load_left + load_right
                frame.add_member_dist_load(beam_placement.spec.name, 'FY', total_load, total_load)
            elif beam_placement.spec.beam_type == 'trimmer':
                # For trimmers, one side has a load break at the opening.
                is_left_opening = i > 0 and self.beams[i-1].spec.beam_type == 'tail'
                is_right_opening = i < len(self.beams)-1 and self.beams[i+1].spec.beam_type == 'tail'
                
                if is_left_opening:
                    frame.add_member_dist_load(beam_placement.spec.name, 'FY', load_left, load_left, 0, opening_z)
                    frame.add_member_dist_load(beam_placement.spec.name, 'FY', load_right, load_right)
                elif is_right_opening:
                    frame.add_member_dist_load(beam_placement.spec.name, 'FY', load_left, load_left)
                    frame.add_member_dist_load(beam_placement.spec.name, 'FY', load_right, load_right, 0, opening_z)
                else:
                    frame.add_member_dist_load(beam_placement.spec.name, 'FY', load_left + load_right, load_left + load_right)


joist_spec = BeamSpec('joist', base=60, height=120, material='wood', beam_type='joist')
trimmer_spec = BeamSpec('trimmer', base=80, height=160, material='wood', beam_type='trimmer')
header_spec = BeamSpec('header', base=120, height=120, material='wood', beam_type='header')

layout = LayoutManager(room_width=ROOM_WIDTH)
beam_A = layout.add_beam_at_offset(joist_spec.copy(name='A'), x_offset=0)
trimmer_E = layout.add_beam_at_offset(trimmer_spec.copy(name='trimmer E'), x_offset=820)
beam_B = layout.add_beam_between(joist_spec.copy(name='B'), beam_A, trimmer_E)

tail_length = beam_length - opening_width - wall_beam_contact_depth/2 - joist_spec.base
tail_spec = joist_spec.copy(beam_type='tail')
tail_C = layout.add_beam_at_offset(tail_spec.copy(name='tail E'), x_offset=1295, z_end=tail_length)
tail_D = layout.add_beam_at_offset(tail_spec.copy(name='tail W'), x_offset=1645, z_end=tail_length)

trimmer_W = layout.add_beam_at_offset(trimmer_spec.copy(name='trimmer W'), x_offset=2140)
beam_E = layout.add_beam_at_offset(joist_spec.copy(name='C'), x_offset=ROOM_WIDTH - joist_spec.base)


# Build the Frame Geometry
layout.sort_beams()
for beam_placement in layout.beams:
    beam_placement.add_to_frame(frame, floor2floor, beam_length)

header_spec.create_section(frame)
frame.add_node('header E', trimmer_E.x_center, floor2floor, tail_length)
frame.add_node('header W', trimmer_W.x_center, floor2floor, tail_length)
frame.add_member('header', 'header W', 'header E', header_spec.material, header_spec.section_name)

for node_name, node in frame.nodes.items():
    if node_name.startswith('floor'):
        frame.def_support(node_name, True, True, True, True, True, True)


## WALL SECTION

def auto_add_walls(frame, layout, wall_thickness, material):
    layout.sort_beams()

    def node(floor, beam, end):
        return f"{'floor ' if floor else ''}{beam.spec.name}{end}"

    frame.add_quad(
        'east wall',
        node(True, layout.beams[-1], 'S'),
        node(True, layout.beams[-1], 'N'),
        node(False, layout.beams[-1], 'N'),
        node(False, layout.beams[-1], 'S'),
        wall_thickness,
        material
    )

    frame.add_quad(
        'west wall',
        node(True, layout.beams[0], 'S'),
        node(True, layout.beams[0], 'N'),
        node(False, layout.beams[0], 'N'),
        node(False, layout.beams[0], 'S'),
        wall_thickness,
        material
    )

auto_add_walls(frame, layout, wall_thickness=80, material='brick')

### WALL SECTION ENDs


# layout.apply_dead_loads(frame)
layout.apply_live_loads(frame, live_load_mpa=-0.003, opening_z=opening_z_start)
frame.analyze(check_statics=True)


def set_wall_opacity(plotter, opacity=0.5):  
  for actor in plotter.renderer.actors.values():
    if (hasattr(actor, 'mapper') and
        hasattr(actor.mapper, 'dataset') and
        actor.mapper.dataset.n_faces_strict > 0):
      actor.prop.opacity = opacity

rndr = Renderer(frame)
rndr.annotation_size = 5
rndr.render_loads = True
rndr.deformed_shape = True
rndr.deformed_scale = 10000
# opacity = .25
# rndr.post_update_callbacks.append(lambda plotter: set_wall_opacity(plotter, opacity=opacity))
rndr.render_model()
