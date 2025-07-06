import meshio
from collections import Counter

msh = meshio.read("two_jets.msh")

print("Available cell types:", [cell.type for cell in msh.cells])

for cell_type in msh.cells_dict:
    tags = msh.get_cell_data("gmsh:physical", cell_type)
    counts = Counter(tags)
    print(f"\nCell type: {cell_type}")
    print("  Physical tags:", set(tags))
    print("  Tag counts:")
    for tag, count in counts.items():
        print(f"    Tag {tag}: {count} elements")
