# Units are mm, N, and MPa (N/mm²)
from Pynite import FEModel3D
from Pynite.Rendering import Renderer

frame = FEModel3D()

# Constants
room_width = 1870
room_length = 3000
floor_height = 2750
beam_base = 60
beam_height = 120
wall_beam_contact_area = 40
beam_length = room_width + wall_beam_contact_area # NOT full length, calculated brick' center. Full length == room_width + wall_beam_contact_area * 2

# Materials
E = 11000 # MPa (N/mm²)
nu = 0.3
G = E / (2 * (1 + nu))  # MPa (N/mm²)
rho = 6.0e-7  # kg/mm3
frame.add_material('Wood', E=E, G=G, nu=nu, rho=rho)

E = 7000 # MPa (N/mm²)
nu = 0.2
G = E / (2 * (1 + nu))  # MPa (N/mm²)
rho = 5.45e-7 # kg/mm3
frame.add_material('Brick', E=E, G=G, nu=nu, rho=rho)

# Floor nodes and supports
frame.add_node('FBL', beam_base/2, -floor_height, 0)
frame.add_node('FBR', room_length - beam_base/2, -floor_height, 0)
frame.add_node('FTL', beam_base/2, -floor_height, room_width)
frame.add_node('FTR', room_length - beam_base/2, -floor_height, room_width)

frame.def_support('FBL', True, True, True, True, True, True)
frame.def_support('FBR', True, True, True, True, True, True)
frame.def_support('FTL', True, True, True, True, True, True)
frame.def_support('FTR', True, True, True, True, True, True)

# Wall nodes and supports
frame.add_node('BL', beam_base/2, 0, 0)
frame.add_node('BR', room_length - beam_base/2, 0, 0)
frame.add_node('TL', beam_base/2, 0, room_width)
frame.add_node('TR', room_length - beam_base/2, 0, room_width)


# Beam cross-section
A = beam_base * beam_height
J = beam_base ** 3 * beam_height * ((1/3) - (0.21 * beam_base/beam_height * (1 - (beam_base ** 4 / (12 * beam_height ** 4)))))
Iy = (beam_base ** 3 * beam_height) / 12
Iz = (beam_base * beam_height ** 3) / 12
frame.add_section('Beam', A, Iy, Iz, J)

# Column cross-section
A = beam_base * wall_beam_contact_area
brick_width = beam_base if beam_base <= wall_beam_contact_area else wall_beam_contact_area
brick_length = wall_beam_contact_area if beam_base <= wall_beam_contact_area else beam_base
J = brick_width ** 3 * brick_length * ((1/3) - (0.21 * brick_width/brick_length * (1 - (brick_width ** 4 / (12 * brick_length ** 4)))))
Iy = (brick_width ** 3 * brick_length) / 12
Iz = (brick_width * brick_length ** 3) / 12
frame.add_section('Column', A, Iy, Iz, J)

# Wall cross-section
A = beam_height * wall_beam_contact_area
brick_width = beam_height if beam_height <= wall_beam_contact_area else wall_beam_contact_area
brick_length = wall_beam_contact_area if beam_height <= wall_beam_contact_area else beam_base
J = brick_width ** 3 * brick_length * ((1/3) - (0.21 * brick_width/brick_length * (1 - (brick_width ** 4 / (12 * brick_length ** 4)))))
Iy = (brick_width ** 3 * brick_length) / 12
Iz = (brick_width * brick_length ** 3) / 12
frame.add_section('Wall', A, Iy, Iz, J)


# Wall members
frame.add_member('ColBL', 'FBL', 'BL', 'Brick', 'Column')
frame.add_member('ColTL', 'FTL', 'TL', 'Brick', 'Column')
frame.add_member('ColBR', 'FBR', 'BR', 'Brick', 'Column')
frame.add_member('ColTR', 'FTR', 'TR', 'Brick', 'Column')

frame.add_member('WallBot', 'BL', 'BR', 'Brick', 'Wall')
frame.add_member('WallTop', 'TL', 'TR', 'Brick', 'Wall')
frame.add_member('FloorBot', 'FBL', 'FBR', 'Brick', 'Wall')
frame.add_member('FloorTop', 'FTL', 'FTR', 'Brick', 'Wall')

frame.add_member('CrossBot', 'BL', 'FBR', 'Brick', 'Wall')
frame.add_member('CrossBot2', 'FBL', 'BR', 'Brick', 'Wall')
frame.add_member('CrossTop', 'TL', 'FTR', 'Brick', 'Wall')
frame.add_member('CrossTop2', 'FTL', 'TR', 'Brick', 'Wall')

# Beam members
frame.add_member('BeamBTL', 'BL', 'TL', 'Wood', 'Beam')
frame.add_member('BeamBTR', 'BR', 'TR', 'Wood', 'Beam')


# Add nodal loads
load = -0.003 # N/mm^2  Equivalent to 3000N/m^2
spacing = 1000
line_load = load * spacing

frame.add_member_dist_load('BeamBTR', 'FY', line_load, line_load)


# Analyze the model
frame.analyze(check_statics=True)

# Print node displacements (actual numeric values, in mm)
print("Node displacements (mm):")
for name, node in frame.nodes.items():
    dx = node.DX.get('Combo 1', 0.0)
    dy = node.DY.get('Combo 1', 0.0)
    dz = node.DZ.get('Combo 1', 0.0)
    print(f"{name}: DX={dx:.6f}, DY={dy:.6f}, DZ={dz:.6f}")

max_dz = max(abs(node.DZ.get('Combo 1', 0.0)) for node in frame.nodes.values())
print(f"Max |DZ| (mm): {max_dz:.6f}")

# Render with realistic scale for verification (not exaggerated)
rndr = Renderer(frame)
rndr.annotation_size = 5
rndr.render_loads = True
rndr.deformed_shape = True
rndr.deformed_scale = 100   # IMPORTANT: use 1 to see true displacement
rndr.render_model()