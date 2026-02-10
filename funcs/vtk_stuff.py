import numpy as np
import struct


def write_binary_mesh_vtk(filename,
                          xyz,
                          e2n,
                          nodal_solutions=None,
                          cell_solutions=None):
    """
    Write a binary VTK file with mesh and optional solution fields.
    
    Parameters:
    -----------
    filename : str
        Output VTK filename
    xyz : np.ndarray
        Points array of shape (n_points, 3)
    e2n : np.ndarray
        Element to node connectivity array of shape (n_elements, nodes_per_element)
        For tetrahedra: (n_elements, 4)
    nodal_solutions : list of tuples, optional
        List of (name, data, is_vector) for nodal fields
        - name: field name string
        - data: np.ndarray - if vector, shape is (3*n_points,) with [all_x, all_y, all_z]
        - is_vector: bool indicating if this is a vector field
    cell_solutions : list of tuples, optional
        List of (name, data, is_vector) for cell fields
        - name: field name string  
        - data: np.ndarray - if vector, shape is (3*n_cells,) with [all_x, all_y, all_z]
        - is_vector: bool indicating if this is a vector field
    """

    def write_header(f):
        f.write(b'# vtk DataFile Version 3.0\n')
        f.write(b'Mesh with solution fields\n')
        f.write(b'BINARY\n')
        f.write(b'DATASET UNSTRUCTURED_GRID\n')

    def write_points(f, points):
        n_points = points.shape[0]
        f.write(f'POINTS {n_points} float\n'.encode())
        # Ensure points are in big-endian float32 format
        f.write(points.astype('>f4').tobytes())

    def write_cells(f, elements):
        n_cells, nodes_per_element = elements.shape
        size = n_cells * (nodes_per_element + 1)  # +1 for the count prefix

        f.write(f'\nCELLS {n_cells} {size}\n'.encode())

        # For each element, write [count, node1, node2, ...]
        for i in range(n_cells):
            # Create array with count followed by node indices
            data = np.empty(nodes_per_element + 1, dtype='>i4')
            data[0] = nodes_per_element
            data[1:] = elements[i]
            f.write(data.tobytes())

        # Write cell types
        f.write(f'\nCELL_TYPES {n_cells}\n'.encode())

        # Determine cell type based on nodes per element
        if nodes_per_element == 4:
            cell_type = 10  # VTK_TETRA
        elif nodes_per_element == 8:
            cell_type = 12  # VTK_HEXAHEDRON
        elif nodes_per_element == 3:
            cell_type = 5  # VTK_TRIANGLE
        elif nodes_per_element == 6:
            cell_type = 13  # VTK_WEDGE
        else:
            cell_type = 10  # Default to tetra

        types = np.full(n_cells, cell_type, dtype='>i4')
        f.write(types.tobytes())

    def write_scalar_field(f, name, data, data_type='float'):
        """Write a scalar field."""
        f.write(f'SCALARS {name} {data_type} 1\n'.encode())
        f.write(b'LOOKUP_TABLE default\n')
        if data_type == 'float':
            f.write(data.astype('>f4').tobytes())
        else:  # int
            f.write(data.astype('>i4').tobytes())

    def write_vector_field(f, name, data, n_entities):
        """Write a vector field, converting from [all_x, all_y, all_z] to interleaved."""
        f.write(f'VECTORS {name} float\n'.encode())

        # Convert from flattened [all_x, all_y, all_z] to interleaved format
        # data has shape (3*n_entities,)
        n = n_entities
        x_components = data[:n]
        y_components = data[n:2 * n]
        z_components = data[2 * n:3 * n]

        # Interleave the components for VTK format
        interleaved = np.empty((n, 3), dtype='>f4')
        interleaved[:, 0] = x_components
        interleaved[:, 1] = y_components
        interleaved[:, 2] = z_components

        f.write(interleaved.tobytes())

    def write_point_data(f, solutions, n_points):
        """Write nodal (point) data."""
        if not solutions:
            return

        f.write(f'\nPOINT_DATA {n_points}\n'.encode())

        for name, data, is_vector in solutions:
            if is_vector:
                write_vector_field(f, name, data, n_points)
            else:
                # Determine data type
                data_type = 'float' if data.dtype.kind == 'f' else 'int'
                write_scalar_field(f, name, data, data_type)

    def write_cell_data(f, solutions, n_cells):
        """Write cell data."""
        if not solutions:
            return

        f.write(f'\nCELL_DATA {n_cells}\n'.encode())

        for name, data, is_vector in solutions:
            if is_vector:
                write_vector_field(f, name, data, n_cells)
            else:
                # Determine data type
                data_type = 'float' if data.dtype.kind == 'f' else 'int'
                write_scalar_field(f, name, data, data_type)

    # Main writing process
    n_points = xyz.shape[0]
    n_cells = e2n.shape[0]

    with open(filename, 'wb') as f:
        write_header(f)
        write_points(f, xyz)
        write_cells(f, e2n)
        write_point_data(f, nodal_solutions, n_points)
        write_cell_data(f, cell_solutions, n_cells)

    print(f"Written mesh to {filename}")

    # Print summary of what was written
    if nodal_solutions:
        print(f"  - {len(nodal_solutions)} nodal field(s)")
    if cell_solutions:
        print(f"  - {len(cell_solutions)} cell field(s)")


def write_binary_grid_vtk_hexmesh(filename,
                                  voxel_centers,
                                  hx,
                                  hy,
                                  hz,
                                  x_values,
                                  y_values,
                                  mag_values,
                                  z_values=None,
                                  labels_values=None):
    """
    Write a binary VTK file with hexahedral elements representing the grid voxels.
    Each voxel will be colored with a scalar value passed in `values`.

    Args:
        filename: Path to the output VTK file
        voxel_centers: (N, 3) array of voxel center coordinates
        hx, hy, hz: size of the voxel along x, y, z
        x_values:   (N,) array of scalar values to assign to each voxel
        y_values:   (N,) array of scalar values to assign to each voxel
        mag_values: (N,) array of scalar values to assign to each voxel
    """
    if len(voxel_centers) == 0:
        print(f"Warning: No voxel centers provided to write to {filename}")
        return

    assert len(x_values) == len(
        voxel_centers), "Length of values must match number of voxels"
    assert len(y_values) == len(
        voxel_centers), "Length of values must match number of voxels"
    assert len(mag_values) == len(
        voxel_centers), "Length of values must match number of voxels"
    if labels_values is not None:
        assert len(labels_values) == len(
            voxel_centers), "Length of values must match number of voxels"

    half_x, half_y, half_z = hx / 2, hy / 2, hz / 2
    num_voxels = len(voxel_centers)
    points = np.zeros((num_voxels * 8, 3), dtype=np.float32)
    hex_cells = np.zeros((num_voxels, 8), dtype=np.int32)
    voxel_ids = np.zeros(num_voxels, dtype=np.int32)

    for i, center in enumerate(voxel_centers):
        x, y, z = center
        base_idx = i * 8

        corners = [[x - half_x, y - half_y, z - half_z],
                   [x + half_x, y - half_y, z - half_z],
                   [x + half_x, y + half_y, z - half_z],
                   [x - half_x, y + half_y, z - half_z],
                   [x - half_x, y - half_y, z + half_z],
                   [x + half_x, y - half_y, z + half_z],
                   [x + half_x, y + half_y, z + half_z],
                   [x - half_x, y + half_y, z + half_z]]

        for j, corner in enumerate(corners):
            points[base_idx + j] = corner

        hex_cells[i] = np.array([
            base_idx, base_idx + 1, base_idx + 2, base_idx + 3, base_idx + 4,
            base_idx + 5, base_idx + 6, base_idx + 7
        ])

        ix = int(round((x - half_x) / hx))
        iy = int(round((y - half_y) / hy))
        iz = int(round((z - half_z) / hz))
        voxel_ids[i] = (iz * 10000) + (iy * 100) + ix

    with open(filename, 'wb') as f:
        header = "# vtk DataFile Version 2.0\nGrid Hexahedral Mesh\nBINARY\nDATASET UNSTRUCTURED_GRID\n"
        f.write(header.encode('ascii'))

        # Points
        f.write(f"POINTS {len(points)} float\n".encode('ascii'))
        for point in points:
            f.write(struct.pack('>fff', *point))

        # Cells
        num_cells = len(hex_cells)
        num_cell_values = num_cells * 9  # 8 nodes + 1 size prefix
        f.write(f"\nCELLS {num_cells} {num_cell_values}\n".encode('ascii'))
        for cell in hex_cells:
            f.write(struct.pack('>iiiiiiiii', 8, *cell))

        # Cell types
        f.write(f"\nCELL_TYPES {num_cells}\n".encode('ascii'))
        for _ in range(num_cells):
            f.write(struct.pack('>i', 12))  # VTK_HEXAHEDRON

        # Cell Data: voxel ID
        f.write(f"\nCELL_DATA {num_cells}\n".encode('ascii'))
        f.write("SCALARS voxel_id int 1\n".encode('ascii'))
        f.write("LOOKUP_TABLE default\n".encode('ascii'))
        for voxel_id in voxel_ids:
            f.write(struct.pack('>i', voxel_id))

        # Cell Data: x values
        f.write("SCALARS x_value float 1\n".encode('ascii'))
        f.write("LOOKUP_TABLE default\n".encode('ascii'))
        for val in x_values:
            f.write(struct.pack('>f', float(val)))
        # Cell Data: y values
        f.write("SCALARS y_value float 1\n".encode('ascii'))
        f.write("LOOKUP_TABLE default\n".encode('ascii'))
        for val in y_values:
            f.write(struct.pack('>f', float(val)))
        # Cell Data: magnitude values
        f.write("SCALARS mag_value float 1\n".encode('ascii'))
        f.write("LOOKUP_TABLE default\n".encode('ascii'))
        for val in mag_values:
            f.write(struct.pack('>f', float(val)))

        # Cell Data: z values
        if z_values is not None:
            f.write("SCALARS z_value float 1\n".encode('ascii'))
            f.write("LOOKUP_TABLE default\n".encode('ascii'))
            for val in z_values:
                f.write(struct.pack('>f', float(val)))
        # Cell Data: labels values
        if labels_values is not None:
            f.write("SCALARS label_value float 1\n".encode('ascii'))
            f.write("LOOKUP_TABLE default\n".encode('ascii'))
            for val in labels_values:
                f.write(struct.pack('>f', float(val)))

    print(f"Binary VTK file written to: {filename} with {num_voxels} voxels")
