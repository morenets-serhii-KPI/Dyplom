import gdstk
lib  = gdstk.Library()
cell = lib.new_cell('TOP')

# Group 1: close but not overlapping (gap=5, gap=10)
cell.add(gdstk.rectangle((0,   0), (40, 40), layer=0))
cell.add(gdstk.rectangle((45,  0), (85, 40), layer=0))   # gap=5
cell.add(gdstk.rectangle((100, 0), (140,40), layer=0))   # gap=15
cell.add(gdstk.rectangle((155, 0), (195,40), layer=0))   # gap=15

# Group 2: overlapping (distance=0, actual intersection)
cell.add(gdstk.rectangle((250,  0), (300, 40), layer=0))
cell.add(gdstk.rectangle((280, 10), (330, 50), layer=0))  # overlap 20x30

cell.add(gdstk.rectangle((350,  0), (400, 40), layer=0))
cell.add(gdstk.rectangle((370,  0), (420, 40), layer=0))  # overlap 30x40

lib.write_gds('input.gds')
print(f'done — {len(cell.polygons)} polygons')