# Units are mm, N, and MPa (N/mm²)
from Pynite import FEModel3D
from Pynite.Rendering import Renderer
# import pandas as pd

frame = FEModel3D()

# Constants
ROOM_WIDTH = 1870
ROOM_LENGTH = 3000
ROOM_HEIGHT = 5465
PLANK_THICKNESS = 25
floor2floor = ROOM_HEIGHT/2 + PLANK_THICKNESS/2
floor2ceiling = ROOM_HEIGHT/2 - PLANK_THICKNESS/2
beam_base = 60
double_base = beam_base * 2
beam_height = 120
trimmer_base = 80
trimmer_height = 160
floor2beam = floor2ceiling - trimmer_height
wall_beam_contact_depth = 40
beam_length = ROOM_WIDTH + wall_beam_contact_depth # NOT real beam length. This calculates wall center to wall center.

# Materials
E = 11000 # MPa (N/mm²)
nu = 0.3
G = E / (2 * (1 + nu))  # MPa (N/mm²)
rho = 4.51e-6
frame.add_material('wood', E=E, G=G, nu=nu, rho=rho)

E = 7000 # MPa (N/mm²)
nu = 0.2
G = E / (2 * (1 + nu))  # MPa (N/mm²)
rho = 5.75e-6
frame.add_material('brick', E=E, G=G, nu=nu, rho=rho)


# Beam A
frame.add_node('floor AS', beam_base/2, 0, 0)
frame.add_node('AS', frame.nodes['floor AS'].X, frame.nodes['floor AS'].Y + floor2floor, frame.nodes['floor AS'].Z)
frame.add_node('floor AN', beam_base/2, 0, beam_length)
frame.add_node('AN', frame.nodes['floor AN'].X, frame.nodes['floor AN'].Y + floor2floor, frame.nodes['floor AN'].Z)

# East trimmer
EAST_TRIMMER_DISTANCE = 820 + trimmer_base/2
frame.add_node('floor trimmer ES', EAST_TRIMMER_DISTANCE, frame.nodes['floor AS'].Y, frame.nodes['floor AS'].Z)
frame.add_node('trimmer ES', EAST_TRIMMER_DISTANCE, frame.nodes['AS'].Y, frame.nodes['AS'].Z)
frame.add_node('floor trimmer EN',  EAST_TRIMMER_DISTANCE, frame.nodes['floor AN'].Y, frame.nodes['floor AN'].Z)
frame.add_node('trimmer EN', EAST_TRIMMER_DISTANCE, frame.nodes['AN'].Y, frame.nodes['AN'].Z)

# Beam B
B_distance = (EAST_TRIMMER_DISTANCE + abs(trimmer_base - beam_base)) / 2
frame.add_node('floor BS', B_distance, frame.nodes['floor AS'].Y, frame.nodes['floor AS'].Z)
frame.add_node('BS', B_distance, frame.nodes['AS'].Y, frame.nodes['AS'].Z)
frame.add_node('floor BN', B_distance, frame.nodes['floor AN'].Y, frame.nodes['floor AN'].Z)
frame.add_node('BN', B_distance, frame.nodes['AN'].Y, frame.nodes['AN'].Z)

# West trimmer
OPENING_LENGTH = 1420
west_trimmer_distance = OPENING_LENGTH + EAST_TRIMMER_DISTANCE + trimmer_base # to center beams
frame.add_node('floor trimmer WS', west_trimmer_distance, frame.nodes['floor trimmer ES'].Y, frame.nodes['floor trimmer ES'].Z)
frame.add_node('trimmer WS', west_trimmer_distance, frame.nodes['trimmer ES'].Y, frame.nodes['trimmer ES'].Z)
frame.add_node('floor trimmer WN', west_trimmer_distance, frame.nodes['floor trimmer EN'].Y, frame.nodes['floor trimmer EN'].Z)
frame.add_node('trimmer WN',  west_trimmer_distance, frame.nodes['trimmer EN'].Y, frame.nodes['trimmer EN'].Z)

# Tail C
stair_width = 630
tail_length = beam_length - stair_width - wall_beam_contact_depth/2 - beam_base
custom_adjustment_to_round_measures = - 70 ########### Write a function to find closest int
CD_distance = (west_trimmer_distance - EAST_TRIMMER_DISTANCE - trimmer_base + custom_adjustment_to_round_measures) / 3
frame.add_node('floor CS', frame.nodes['floor trimmer ES'].X + CD_distance, frame.nodes['floor trimmer ES'].Y, frame.nodes['floor trimmer ES'].Z)
frame.add_node('CS', frame.nodes['trimmer ES'].X + CD_distance, frame.nodes['trimmer ES'].Y, frame.nodes['trimmer ES'].Z)
frame.add_node('CN', frame.nodes['trimmer EN'].X + CD_distance, frame.nodes['trimmer EN'].Y, tail_length)

# Tail D
frame.add_node('floor DS', frame.nodes['floor trimmer WS'].X - CD_distance, frame.nodes['floor trimmer WS'].Y, frame.nodes['floor trimmer WS'].Z)
frame.add_node('DS', frame.nodes['trimmer WS'].X - CD_distance, frame.nodes['trimmer WS'].Y, frame.nodes['trimmer WS'].Z)
frame.add_node('DN', frame.nodes['trimmer WN'].X - CD_distance, frame.nodes['trimmer WN'].Y, tail_length)

# Header
frame.add_node('header E', frame.nodes['trimmer ES'].X, frame.nodes['trimmer ES'].Y, tail_length)
frame.add_node('header W', frame.nodes['trimmer WS'].X, frame.nodes['trimmer WS'].Y, tail_length)

# Beam E
frame.add_node('floor ES', ROOM_LENGTH - beam_base/2, 0, 0)
frame.add_node('floor EN', ROOM_LENGTH - beam_base/2, 0, beam_length)
frame.add_node('ES', frame.nodes['floor ES'].X, frame.nodes['floor ES'].Y + floor2floor, frame.nodes['floor ES'].Z)
frame.add_node('EN', frame.nodes['floor EN'].X, frame.nodes['floor EN'].Y + floor2floor, frame.nodes['floor EN'].Z)


# Walls
wall_thickness = 80
frame.add_quad('east wall', 'floor AS', 'floor AN', 'AN', 'AS', wall_thickness, 'brick')
frame.add_quad('west wall', 'floor ES', 'floor EN', 'EN', 'ES', wall_thickness, 'brick')

frame.add_quad('south wall AB', 'floor AS', 'floor BS', 'BS', 'AS', wall_thickness, 'brick')
frame.add_quad('south wall Btrimmer', 'floor trimmer ES', 'floor BS', 'BS', 'trimmer ES', wall_thickness, 'brick')
frame.add_quad('south wall trimmerC', 'floor trimmer ES', 'floor CS', 'CS', 'trimmer ES', wall_thickness, 'brick')
frame.add_quad('south wall CD', 'floor DS', 'floor CS', 'CS', 'DS', wall_thickness, 'brick')
frame.add_quad('south wall Dtrimmer', 'floor trimmer WS', 'floor DS', 'DS', 'trimmer WS', wall_thickness, 'brick')
frame.add_quad('south wall trimmerE', 'floor trimmer WS', 'floor ES', 'ES', 'trimmer WS', wall_thickness, 'brick')

frame.add_quad('north wall AB', 'floor AN', 'floor BN', 'BN', 'AN', wall_thickness, 'brick')
frame.add_quad('north wall Btrimmer', 'floor trimmer EN', 'floor BN', 'BN', 'trimmer EN', wall_thickness, 'brick')
frame.add_quad('north wall trimmer', 'floor trimmer EN', 'floor trimmer WN', 'trimmer WN', 'trimmer EN', wall_thickness, 'brick')
frame.add_quad('north wall trimmerE', 'floor EN', 'floor trimmer WN', 'trimmer WN', 'EN', wall_thickness, 'brick')

for node in frame.nodes:
    if node.startswith('floor'):
        frame.def_support(node, True, True, True, True, True, True)


# Beam cross-section
A = beam_base * beam_height
J = beam_base ** 3 * beam_height * ((1/3) - (0.21 * beam_base/beam_height * (1 - (beam_base ** 4 / (12 * beam_height ** 4)))))
Iy = (beam_base ** 3 * beam_height) / 12
Iz = (beam_base * beam_height ** 3) / 12
frame.add_section('beam', A, Iy, Iz, J)

# Beam cross-section
A = double_base * beam_height
J = double_base ** 3 * beam_height * ((1/3) - (0.21 * double_base/beam_height * (1 - (double_base ** 4 / (12 * beam_height ** 4)))))
Iy = (double_base ** 3 * beam_height) / 12
Iz = (double_base * beam_height ** 3) / 12
frame.add_section('double beam', A, Iy, Iz, J)

# Trimmer cross-section
A = trimmer_base * trimmer_height
J = trimmer_base ** 3 * trimmer_height * ((1/3) - (0.21 * trimmer_base/trimmer_height * (1 - (trimmer_base ** 4 / (12 * trimmer_height ** 4)))))
Iy = (trimmer_base ** 3 * trimmer_height) / 12
Iz = (trimmer_base * trimmer_height ** 3) / 12
frame.add_section('trimmer', A, Iy, Iz, J)

# Joist members
frame.add_member('A', 'AN', 'AS', 'wood', 'beam')
frame.add_member('B', 'BN', 'BS', 'wood', 'beam')
frame.add_member('trimmer E', 'trimmer EN', 'trimmer ES', 'wood', 'trimmer')
frame.add_member('trimmer W', 'trimmer WN', 'trimmer WS', 'wood', 'trimmer')
frame.add_member('header', 'header W', 'header E', 'wood', 'double beam')
frame.add_member('C', 'CN', 'CS', 'wood', 'beam')
frame.add_member('D', 'DN', 'DS', 'wood', 'beam')
frame.add_member('E', 'EN', 'ES', 'wood', 'beam')

# Joist dead loads
for member in frame.members:
    dead_line_load = (-frame.materials['wood'].rho * frame.sections[frame.members[member].section.name].A)
    print(f'{member}: {dead_line_load}')
    frame.add_member_dist_load(member, 'FY', dead_line_load, dead_line_load)

# Tributary areas
A_trib_width = (beam_base / 2) + (frame.nodes['BS'].X - frame.nodes['AS'].X) / 2
B_trib_width = (frame.nodes['BS'].X - frame.nodes['AS'].X) / 2 + (frame.nodes['trimmer ES'].X - frame.nodes['BS'].X) / 2
trimmerE_trib_widthE = (frame.nodes['trimmer ES'].X - frame.nodes['BS'].X) / 2
trimmerE_trib_widthW = (frame.nodes['CS'].X - frame.nodes['trimmer ES'].X) / 2
C_trib_width = (frame.nodes['CS'].X - frame.nodes['trimmer ES'].X) / 2 + (frame.nodes['DS'].X - frame.nodes['CS'].X) / 2
D_trib_width = (frame.nodes['DS'].X - frame.nodes['CS'].X) / 2 + (frame.nodes['trimmer WS'].X - frame.nodes['DS'].X) / 2
trimmerW_trib_widthE = (frame.nodes['trimmer WS'].X - frame.nodes['DS'].X) / 2
trimmerW_trib_widthW = (frame.nodes['ES'].X - frame.nodes['trimmer WS'].X) / 2
E_trib_width = (frame.nodes['ES'].X - frame.nodes['trimmer WS'].X) / 2 + (beam_base / 2)

print(f'''
TRIBUTARY AREAS:
    A: {A_trib_width}
    B: {B_trib_width}
    trimmer EE: {trimmerE_trib_widthE}
    trimmer EW: {trimmerE_trib_widthW}
    C: {C_trib_width}
    D: {D_trib_width}
    trimmer WE: {trimmerW_trib_widthE}
    trimmer WW: {trimmerW_trib_widthW}
    E: {E_trib_width}
      ''')

# Add live loads
live_load = -0.003

load_start = stair_width + wall_beam_contact_depth/2 - beam_base/2
load_end = beam_length - wall_beam_contact_depth/2

frame.add_member_dist_load('A', 'FY', live_load * A_trib_width, live_load * A_trib_width)
frame.add_member_dist_load('B', 'FY', live_load * B_trib_width, live_load * B_trib_width)
frame.add_member_dist_load('trimmer E', 'FY', live_load * trimmerE_trib_widthE, live_load * trimmerE_trib_widthE)
frame.add_member_dist_load('trimmer E', 'FY', live_load * trimmerE_trib_widthW, live_load * trimmerE_trib_widthW, load_start, load_end)
frame.add_member_dist_load('C', 'FY', live_load * C_trib_width, live_load * C_trib_width)
frame.add_member_dist_load('D', 'FY', live_load * D_trib_width, live_load * D_trib_width)
frame.add_member_dist_load('trimmer W', 'FY', live_load * trimmerE_trib_widthE, live_load * trimmerE_trib_widthE, load_start, load_end)
frame.add_member_dist_load('trimmer W', 'FY', live_load * trimmerE_trib_widthW, live_load * trimmerE_trib_widthW)
frame.add_member_dist_load('E', 'FY', live_load * E_trib_width, live_load * E_trib_width)

frame.analyze(check_statics=True)

for beam in frame.members:
    print(f"\n--- {beam} Stats ---")
    print(f"Max Moment (Mz): {frame.members[beam].max_moment('Mz', 'Combo 1'):.3f} N-mm")
    print(f"Min Moment (Mz): {frame.members[beam].min_moment('Mz', 'Combo 1'):.3f} N-mm")
    print(f"Max Shear (Fy): {frame.members[beam].max_shear('Fy', 'Combo 1'):.3f} N")
    print(f"Min Shear (Fy): {frame.members[beam].min_shear('Fy', 'Combo 1'):.3f} N")
    print(f"Max Deflection (dy): {frame.members[beam].max_deflection('dy', 'Combo 1'):.3f} mm")
    print(f"Min Deflection (dy): {frame.members[beam].min_deflection('dy', 'Combo 1'):.3f} mm")


# def set_wall_opacity(plotter, opacity=0.5):  
#   for actor in plotter.renderer.actors.values():
#     if (hasattr(actor, 'mapper') and
#         hasattr(actor.mapper, 'dataset') and
#         actor.mapper.dataset.n_faces_strict > 0):
#       actor.prop.opacity = opacity

# rndr = Renderer(frame)
# rndr.annotation_size = 5
# rndr.render_loads = True
# rndr.deformed_shape = True
# rndr.deformed_scale = 1000
# opacity = .25
# rndr.post_update_callbacks.append(lambda plotter: set_wall_opacity(plotter, opacity=opacity))
# rndr.render_model()