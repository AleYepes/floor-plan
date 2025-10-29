# Units are mm, N, and MPa (N/mm²)
from Pynite import FEModel3D
from Pynite.Rendering import Renderer
import math

frame = FEModel3D()

# Constants
room_width = 1870
room_length = 3017
room_height = 5465
plank_thickness = 25
floor2floor = room_height/2 + plank_thickness/2
floor2ceiling = room_height/2 - plank_thickness/2
beam_base = 60
beam_height = 120
trimmer_base = 80
trimmer_height = 160
floor2beam = floor2ceiling - trimmer_height
wall_beam_contact_area = 40
beam_length = room_width + wall_beam_contact_area # NOT real beam length. This calculates wall center to wall center.

# Materials
E = 11000 # MPa (N/mm²)
nu = 0.3
G = E / (2 * (1 + nu))  # MPa (N/mm²)
rho = 6.0e-7  # kg/mm3
frame.add_material('wood', E=E, G=G, nu=nu, rho=rho)

E = 7000 # MPa (N/mm²)
nu = 0.2
G = E / (2 * (1 + nu))  # MPa (N/mm²)
rho = 5.45e-7 # kg/mm3
frame.add_material('brick', E=E, G=G, nu=nu, rho=rho)

# Floor nodes and supports
# frame.add_node('floor HS', room_length - beam_base/2, 0, 0)
# frame.add_node('floor HN', room_length - beam_base/2, 0, room_width)
# frame.add_node('HS', frame.nodes['floor HS'].X, frame.nodes['floor HS'].Y + floor2floor, frame.nodes['floor HS'].Z)
# frame.add_node('HN', frame.nodes['floor HN'].X, frame.nodes['floor HN'].Y + floor2floor, frame.nodes['floor HN'].Z)

# A
frame.add_node('floor AS', beam_base/2, 0, 0)
frame.add_node('AS', frame.nodes['floor AS'].X, frame.nodes['floor AS'].Y + floor2floor, frame.nodes['floor AS'].Z)
frame.add_node('floor AN', beam_base/2, 0, room_width)
frame.add_node('AN', frame.nodes['floor AN'].X, frame.nodes['floor AN'].Y + floor2floor, frame.nodes['floor AN'].Z)

# East trimmer
EAST_TRIMMER_DISTANCE = 912.3 - trimmer_base/2 - beam_base/2 # to center beams
frame.add_node('floor trimmer ES', frame.nodes['floor AS'].X + EAST_TRIMMER_DISTANCE, frame.nodes['floor AS'].Y, frame.nodes['floor AS'].Z)
frame.add_node('trimmer ES', frame.nodes['AS'].X + EAST_TRIMMER_DISTANCE, frame.nodes['AS'].Y, frame.nodes['AS'].Z)
frame.add_node('floor trimmer EN', frame.nodes['floor AN'].X + EAST_TRIMMER_DISTANCE, frame.nodes['floor AN'].Y, frame.nodes['floor AN'].Z)
frame.add_node('trimmer EN', frame.nodes['AN'].X + EAST_TRIMMER_DISTANCE, frame.nodes['AN'].Y, frame.nodes['AN'].Z)

# B
B_distance = (EAST_TRIMMER_DISTANCE - abs((trimmer_base - beam_base)/2)) / 2
frame.add_node('floor BS', frame.nodes['floor AS'].X + B_distance, frame.nodes['floor AS'].Y, frame.nodes['floor AS'].Z)
frame.add_node('BS', frame.nodes['AS'].X + B_distance, frame.nodes['AS'].Y, frame.nodes['AS'].Z)
frame.add_node('floor BN', frame.nodes['floor AN'].X + B_distance, frame.nodes['floor AN'].Y, frame.nodes['floor AN'].Z)
frame.add_node('BN', frame.nodes['AN'].X + B_distance, frame.nodes['AN'].Y, frame.nodes['AN'].Z)

# West trimmer
west_trimmer_distance = 1423.7 + trimmer_base # to center beams
frame.add_node('floor trimmer WS', frame.nodes['floor trimmer ES'].X + west_trimmer_distance, frame.nodes['floor trimmer ES'].Y, frame.nodes['floor trimmer ES'].Z)
frame.add_node('trimmer WS', frame.nodes['trimmer ES'].X + west_trimmer_distance, frame.nodes['trimmer ES'].Y, frame.nodes['trimmer ES'].Z)
frame.add_node('floor trimmer WN', frame.nodes['floor trimmer EN'].X + west_trimmer_distance, frame.nodes['floor trimmer EN'].Y, frame.nodes['floor trimmer EN'].Z)
frame.add_node('trimmer WN', frame.nodes['trimmer EN'].X + west_trimmer_distance, frame.nodes['trimmer EN'].Y, frame.nodes['trimmer EN'].Z)

# C and D
CD_distance = (west_trimmer_distance - trimmer_base) / 3 + trimmer_base/2
frame.add_node('floor CS', frame.nodes['floor trimmer ES'].X + CD_distance, frame.nodes['floor trimmer ES'].Y, frame.nodes['floor trimmer ES'].Z)
frame.add_node('CS', frame.nodes['trimmer ES'].X + CD_distance, frame.nodes['trimmer ES'].Y, frame.nodes['trimmer ES'].Z)
frame.add_node('floor CN', frame.nodes['floor trimmer EN'].X + CD_distance, frame.nodes['floor trimmer EN'].Y, frame.nodes['floor trimmer EN'].Z)
frame.add_node('CN', frame.nodes['trimmer EN'].X + CD_distance, frame.nodes['trimmer EN'].Y, frame.nodes['trimmer EN'].Z)

frame.add_node('floor DS', frame.nodes['floor trimmer ES'].X + CD_distance * 2, frame.nodes['floor trimmer ES'].Y, frame.nodes['floor trimmer ES'].Z)
frame.add_node('DS', frame.nodes['trimmer ES'].X + CD_distance * 2, frame.nodes['trimmer ES'].Y, frame.nodes['trimmer ES'].Z)
frame.add_node('floor DN', frame.nodes['floor trimmer EN'].X + CD_distance * 2, frame.nodes['floor trimmer EN'].Y, frame.nodes['floor trimmer EN'].Z)
frame.add_node('DN', frame.nodes['trimmer EN'].X + CD_distance * 2, frame.nodes['trimmer EN'].Y, frame.nodes['trimmer EN'].Z)



# Wall members
wall_thickness = 80
frame.add_quad('east wall', 'floor AS', 'floor AN', 'AN', 'AS', wall_thickness, 'brick')

frame.add_quad('south wall AB', 'floor AS', 'floor BS', 'BS', 'AS', wall_thickness, 'brick')
frame.add_quad('south wall Btrimmer', 'floor trimmer ES', 'floor BS', 'BS', 'trimmer ES', wall_thickness, 'brick')
frame.add_quad('south wall trimmerC', 'floor trimmer ES', 'floor CS', 'CS', 'trimmer ES', wall_thickness, 'brick')
frame.add_quad('south wall CD', 'floor DS', 'floor CS', 'CS', 'DS', wall_thickness, 'brick')
frame.add_quad('south wall Dtrimmer', 'floor trimmer WS', 'floor CS', 'CS', 'trimmer WS', wall_thickness, 'brick')

frame.add_quad('north wall AB', 'floor AN', 'floor BN', 'BN', 'AN', wall_thickness, 'brick')
frame.add_quad('north wall Btrimmer', 'floor trimmer EN', 'floor BN', 'BN', 'trimmer EN', wall_thickness, 'brick')
frame.add_quad('north wall trimmerC', 'floor trimmer EN', 'floor CN', 'CN', 'trimmer EN', wall_thickness, 'brick')
frame.add_quad('north wall CD', 'floor DN', 'floor CN', 'CN', 'DN', wall_thickness, 'brick')
frame.add_quad('north wall Dtrimmer', 'floor trimmer WN', 'floor CN', 'CN', 'trimmer WN', wall_thickness, 'brick')

# frame.add_quad('north wall Btrimmer', 'floor trimmer EN', 'floor BN', 'BN', 'trimmer EN', wall_thickness, 'brick')
# frame.add_quad('north wall trimmerC', 'floor trimmer EN', 'floor trimmer WN', 'trimmer WN', 'trimmer EN', wall_thickness, 'brick')
# frame.add_quad('north wall CD', 'floor trimmer EN', 'floor trimmer WN', 'trimmer WN', 'trimmer EN', wall_thickness, 'brick')
# frame.add_quad('north wall Dtrimmer', 'floor trimmer EN', 'floor trimmer WN', 'trimmer WN', 'trimmer EN', wall_thickness, 'brick')


for node in frame.nodes:
    if node.startswith('floor'):
        frame.def_support(node, True, True, True, True, True, True)


# Beam cross-section
A = beam_base * beam_height
J = beam_base ** 3 * beam_height * ((1/3) - (0.21 * beam_base/beam_height * (1 - (beam_base ** 4 / (12 * beam_height ** 4)))))
Iy = (beam_base ** 3 * beam_height) / 12
Iz = (beam_base * beam_height ** 3) / 12
frame.add_section('beam', A, Iy, Iz, J)
# Trimmer cross-section
A = trimmer_base * trimmer_height
J = trimmer_base ** 3 * trimmer_height * ((1/3) - (0.21 * trimmer_base/trimmer_height * (1 - (trimmer_base ** 4 / (12 * trimmer_height ** 4)))))
Iy = (trimmer_base ** 3 * trimmer_height) / 12
Iz = (trimmer_base * trimmer_height ** 3) / 12
frame.add_section('trimmer', A, Iy, Iz, J)


# # Beam members
frame.add_member('A', 'AN', 'AS', 'wood', 'beam')
frame.add_member('B', 'BN', 'BS', 'wood', 'beam')
frame.add_member('trimmer E', 'trimmer EN', 'trimmer ES', 'wood', 'trimmer')
frame.add_member('trimmer W', 'trimmer WN', 'trimmer WS', 'wood', 'trimmer')


# Add loads
load = -0.003 # N/mm^2  Equivalent to 3000N/m^2
spacing = 1000
line_load = load * spacing
frame.add_member_dist_load('A', 'FY', line_load, line_load)
frame.add_member_dist_load('B', 'FY', line_load, line_load)
frame.add_member_dist_load('trimmer E', 'FY', line_load, line_load)
frame.add_member_dist_load('trimmer W', 'FY', line_load, line_load)

frame.analyze(check_statics=True)

# # Check the displacement of the top beam node
# print("Displacement at TR node (DX, DY, DZ):")
# print(f"({frame.nodes['HN'].DX['Combo 1']:.3f}, {frame.nodes['HN'].DY['Combo 1']:.3f}, {frame.nodes['HN'].DZ['Combo 1']:.3f})")
# print("\nRotation at TR node (RX, RY, RZ):")
# print(f"({frame.nodes['HN'].RX['Combo 1']:.3f}, {frame.nodes['HN'].RY['Combo 1']:.3f}, {frame.nodes['HN'].RZ['Combo 1']:.3f})")

beam = frame.members['A']
print("\n--- BeamBTR Stats ---")
print(f"Max Moment (Mz): {beam.max_moment('Mz', 'Combo 1'):.3f} N-mm")
print(f"Min Moment (Mz): {beam.min_moment('Mz', 'Combo 1'):.3f} N-mm")
print(f"Max Shear (Fy): {beam.max_shear('Fy', 'Combo 1'):.3f} N")
print(f"Min Shear (Fy): {beam.min_shear('Fy', 'Combo 1'):.3f} N")
print(f"Max Deflection (dy): {beam.max_deflection('dy', 'Combo 1'):.3f} mm")
print(f"Min Deflection (dy): {beam.min_deflection('dy', 'Combo 1'):.3f} mm")

rndr = Renderer(frame)
rndr.annotation_size = 5
rndr.render_loads = True
rndr.deformed_shape = True
rndr.deformed_scale = 100
rndr.render_model()