import os, subprocess


"""The function generate_mesh is called in EnvBackward_step.py, as first arg are passed the geometry_params, the template is passed"""
"""The dimensionality is in-line initialized always here"""


def generate_mesh(
    g_parameters, template="backward_facing_step.template_geo", dim=2
):  # dim stands for the dimensionality of the mesh (2D).
    """Modify template according args and make gmsh generate the mesh"""

    # ensures that the template exists and copy the arguments passed
    assert os.path.exists(template)
    params_cp = g_parameters.copy()

    # flag parameter to change the geometry_params
    change_parameters = 0

    with open(template, "r") as f:
        template_lines = f.readlines()

    # keep apart the geometry_params from the rest of the template file
    template_params_code = list(
        map(lambda g: g.startswith("DefineConstant"), template_lines)
    ).index(True)

    # if we want to change some parameters before meshing,
    # we can do it here without changing the .geo file
    # (e.g. here it is provided the code for changing the 'control width' which of course goes along with a change in 'length_before_control')
    if change_parameters == 1:
        line_to_change = list(
            map(lambda g: g.startswith("control_width"), template_lines)
        ).index(True)
        template_lines.pop(line_to_change)
        control_width = list(map(float, params_cp.pop("control_width")))

        line_to_change = list(
            map(lambda g: g.startswith("length_before_control"), template_lines)
        ).index(True)
        template_lines.pop(line_to_change)
        length_before_control = list(map(float, params_cp.pop("length_before_control")))

        control_width = "control_width[] = {%s};\n" % (
            ", ".join(list(map(str, control_width)))
        )
        length_before_control = "length_before_control [] = {%s};\n" % (
            ", ".join(list(map(str, length_before_control)))
        )
        new_geometry_params = "".join(
            [control_width, length_before_control]
            + template_lines[template_params_code:]
        )

    # changes if necessary the 'output' (i.e. file_root) in the copy of geometry_params with the template passed here and verify that is a .geo file
    output = params_cp.pop("output")
    if not output:
        output = template
    assert os.path.splitext(output)[1] == ".geo"

    # open the template file through the variable output and change it with the changed parameters through the copy of the parameters passed in the function
    if change_parameters == 1:
        with open(output, "w") as f:
            f.write(new_geometry_params)
    else:
        with open(output, "w") as f:
            f.write(" ".join(template_lines[template_params_code:]))

    # preparing the commands to run for building the mesh
    cmd = "gmsh -0 -format msh2 %s " % output
    scale = params_cp.pop("clscale")
    subprocess.call(cmd, shell=True)

    unrolled = "_".join([output, "unrolled"])
    assert os.path.exists(unrolled)

    # call gmesh to compute the mesh
    return subprocess.call(
        ["gmsh -%d -format msh2 -clscale %g %s" % (dim, scale, unrolled)], shell=True
    )


# main omitted because already shown that it works.
