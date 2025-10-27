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
frame.add_material('wood', E=E, G=G, nu=nu, rho=rho)

E = 7000 # MPa (N/mm²)
nu = 0.2
G = E / (2 * (1 + nu))  # MPa (N/mm²)
rho = 5.45e-7 # kg/mm3
frame.add_material('brick', E=E, G=G, nu=nu, rho=rho)

# Floor nodes and supports
frame.add_node('floor SE', beam_base/2, 0, 0)
frame.add_node('floor NE', beam_base/2, 0, room_width)
frame.add_node('floor SW', room_length - beam_base/2, 0, 0)
frame.add_node('floor NW', room_length - beam_base/2, 0, room_width)

frame.def_support('floor SE', True, True, True, True, True, True)
frame.def_support('floor SW', True, True, True, True, True, True)
frame.def_support('floor NE', True, True, True, True, True, True)
frame.def_support('floor NW', True, True, True, True, True, True)

# Wall nodes
frame.add_node('SE', frame.nodes['floor SE'].X, frame.nodes['floor SE'].Y + floor_height, frame.nodes['floor SE'].Z)
frame.add_node('NE', frame.nodes['floor NE'].X, frame.nodes['floor NE'].Y + floor_height, frame.nodes['floor NE'].Z)
frame.add_node('SW', frame.nodes['floor SW'].X, frame.nodes['floor SW'].Y + floor_height, frame.nodes['floor SW'].Z)
frame.add_node('NW', frame.nodes['floor NW'].X, frame.nodes['floor NW'].Y + floor_height, frame.nodes['floor NW'].Z)

# Tail joist
frame.add_node('tail S', frame.nodes['SW'].X - 680, frame.nodes['SW'].Y, frame.nodes['SW'].Z)
frame.add_node('floor tail S', frame.nodes['floor SW'].X - 680, frame.nodes['floor SW'].Y, frame.nodes['floor SW'].Z)
frame.add_node('tail N', frame.nodes['NW'].X - 680, frame.nodes['NW'].Y, frame.nodes['NW'].Z)
frame.add_node('floor tail N', frame.nodes['floor NW'].X - 680, frame.nodes['floor NW'].Y, frame.nodes['floor NW'].Z)

## Double trimmer
frame.add_node('double trimmer S', frame.nodes['SW'].X - 1390, frame.nodes['SW'].Y, frame.nodes['SW'].Z)
frame.add_node('floor double trimmer S', frame.nodes['floor SW'].X - 1390, frame.nodes['floor SW'].Y, frame.nodes['floor SW'].Z)
frame.add_node('double trimmer N', frame.nodes['NW'].X - 1390, frame.nodes['NW'].Y, frame.nodes['NW'].Z)
frame.add_node('floor double trimmer N', frame.nodes['floor NW'].X - 1390, frame.nodes['floor NW'].Y, frame.nodes['floor NW'].Z)

# Header
frame.add_node('header S', frame.nodes['SW'].X, frame.nodes['SW'].Y, frame.nodes['SW'].Z + 555)
frame.add_node('floor header S', frame.nodes['floor SW'].X, frame.nodes['floor SW'].Y, frame.nodes['floor SW'].Z + 555)
frame.add_node('header N', frame.nodes['NW'].X, frame.nodes['NW'].Y, frame.nodes['NW'].Z - 555)
frame.add_node('floor header N', frame.nodes['floor NW'].X, frame.nodes['floor NW'].Y, frame.nodes['floor NW'].Z - 555)

frame.add_node('header-trimmer S', frame.nodes['double trimmer S'].X, frame.nodes['double trimmer S'].Y, frame.nodes['header S'].Z)
frame.add_node('header-trimmer N', frame.nodes['double trimmer N'].X, frame.nodes['double trimmer N'].Y, frame.nodes['header N'].Z)

frame.add_node('header-tail S', frame.nodes['tail S'].X, frame.nodes['tail S'].Y, frame.nodes['header S'].Z)
frame.add_node('header-tail N', frame.nodes['tail N'].X, frame.nodes['tail N'].Y, frame.nodes['header N'].Z)

# Wall members
wall_thickness = 90
frame.add_quad('south wall 3', 'floor SE', 'floor double trimmer S', 'double trimmer S', 'SE', wall_thickness, 'brick')
frame.add_quad('south wall 4', 'floor tail S', 'floor double trimmer S', 'double trimmer S', 'tail S', wall_thickness, 'brick')
frame.add_quad('south wall 5', 'floor SW', 'floor tail S', 'tail S', 'SW', wall_thickness, 'brick')

frame.add_quad('north wall 3', 'floor NE', 'floor double trimmer N', 'double trimmer N', 'NE', wall_thickness, 'brick')
frame.add_quad('north wall 4', 'floor tail N', 'floor double trimmer N', 'double trimmer N', 'tail N', wall_thickness, 'brick')
frame.add_quad('north wall 5', 'floor NW', 'floor tail N', 'tail N', 'NW', wall_thickness, 'brick')

frame.add_quad('east wall', 'floor SE', 'floor NE', 'NE', 'SE', wall_thickness, 'brick')

# frame.add_quad('west wall', 'floor SW', 'floor NW', 'NW', 'SW', wall_thickness, 'brick')
frame.add_quad('west wall 1', 'floor SW', 'floor header S', 'header S', 'SW', wall_thickness, 'brick')
frame.add_quad('west wall 2', 'floor header N', 'floor header S', 'header S', 'header N', wall_thickness, 'brick')
frame.add_quad('west wall 3', 'floor header N', 'floor NW', 'NW', 'header N', wall_thickness, 'brick')


# Beam cross-section
A = beam_base * beam_height
J = beam_base ** 3 * beam_height * ((1/3) - (0.21 * beam_base/beam_height * (1 - (beam_base ** 4 / (12 * beam_height ** 4)))))
Iy = (beam_base ** 3 * beam_height) / 12
Iz = (beam_base * beam_height ** 3) / 12
frame.add_section('beam', A, Iy, Iz, J)

# Beam members
frame.add_member('trimmer', 'double trimmer N', 'double trimmer S', 'wood', 'beam')
frame.add_member('header S', 'header-trimmer S', 'header S', 'wood', 'beam')
frame.add_member('header N', 'header-trimmer N', 'header N', 'wood', 'beam')
frame.add_member('tail S', 'header-tail S', 'tail S', 'wood', 'beam')
frame.add_member('tail N', 'header-tail N', 'tail N', 'wood', 'beam')


# Add loads
load = -0.003 # N/mm^2  Equivalent to 3000N/m^2
spacing = 1000
line_load = load * spacing
frame.add_member_dist_load('trimmer', 'FY', line_load, line_load)
frame.add_member_dist_load('header N', 'FY', line_load, line_load)
frame.add_member_dist_load('header S', 'FY', line_load, line_load)


frame.analyze(check_statics=True)

# # Check the displacement of the top beam node
# print("Displacement at TR node (DX, DY, DZ):")
# print(f"({frame.nodes['NW'].DX['Combo 1']:.3f}, {frame.nodes['NW'].DY['Combo 1']:.3f}, {frame.nodes['NW'].DZ['Combo 1']:.3f})")
# print("\nRotation at TR node (RX, RY, RZ):")
# print(f"({frame.nodes['NW'].RX['Combo 1']:.3f}, {frame.nodes['NW'].RY['Combo 1']:.3f}, {frame.nodes['NW'].RZ['Combo 1']:.3f})")

beam = frame.members['trimmer']
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