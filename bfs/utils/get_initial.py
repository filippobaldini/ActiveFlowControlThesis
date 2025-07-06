from dolfin import *

# Paths to the input .xdmf files
u_input_file_path = "../../simulations/u_Re100.xdmf"
p_input_file_path = "../../simulations/p_Re100.xdmf"

# Paths to the output .xdmf files
u_output_file_path = "output/u_Re100.xdmf"
p_output_file_path = "output/p_Re100.xdmf"

# Timestep to extract (0-based index)
timestep_to_extract = 2500  # Example: Extract the 51st timestep

# Open the input .xdmf files
with XDMFFile(u_input_file_path) as u_infile, XDMFFile(p_input_file_path) as p_infile:
    # Read the mesh from the velocity file (assuming both files share the same mesh)
    # Importing mesh
    mesh = Mesh()
    f = HDF5File(mesh.mpi_comm(), "mesh/our_mesh.h5", "r")
    f.read(mesh, "mesh", False)

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)

    # Create functions to store the data
    u = Function(V)
    p = Function(Q)

    # Iterate through the timesteps
    for i in range(3000):  # Replace 200 with the total number of timesteps
        # Read velocity and pressure fields at the current timestep
        u_infile.read_checkpoint(u, "velocity", i)
        p_infile.read_checkpoint(p, "pressure", i)

        # If this is the desired timestep, save it to new files
        if i == timestep_to_extract:
            # Save velocity field with the name "u0"
            with XDMFFile(u_output_file_path) as u_outfile:
                u_outfile.write_checkpoint(
                    u, "u0", 0, XDMFFile.Encoding.HDF5, append=False
                )
            # Save pressure field with the name "p0"
            with XDMFFile(p_output_file_path) as p_outfile:
                p_outfile.write_checkpoint(
                    p, "p0", 0, XDMFFile.Encoding.HDF5, append=False
                )
            print(
                f"Timestep {i} saved to {u_output_file_path} and {p_output_file_path}"
            )
            break
