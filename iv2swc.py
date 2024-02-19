import numpy as np


def iv2dic(file_text):
    # Write the content of the iv file into a dictionary

    filaments_coord = {}
    filaments_struct_identifier = {}

    pre_last_point = np.array([0.0, 0.0, 0.0])
    coordinate_idx = 0
    starting_point = True
    filament_id = 0
    end_idx = 0
    while True:
        # Find the instances of 'Coordinate3 {' and '}' pairs which enclose the values for the coordinates
        start_idx = file_text.find('Coordinate3 {', end_idx)
        end_idx = file_text.find('}', start_idx)
        # Terminate if no more 'Coordinate3 {' and '}' pairs found
        if start_idx == -1:
            break

        # Extract the point values
        coord_section = file_text[start_idx:end_idx + 1]
        point_start_idx = coord_section.find('point [') + len('point [')
        point_end_idx = coord_section.find(']', point_start_idx)
        point_str = coord_section[point_start_idx:point_end_idx].strip()

        # replace unwanted characters with empty string and split into individual values
        point_str = point_str.replace('\n', ' ').replace(',', '')
        values = point_str.split()
        arr = np.array(values, dtype=float)
        arr = arr.reshape((-1, 3))

        # Create a new dictionary entry
        filament_name = 'Filament_' + str(filament_id).zfill(5)
        filament_id += 1
        filaments_coord[filament_name] = arr

        if coordinate_idx == 0:
            filaments_struct_identifier[filament_name] = 2
        elif starting_point and (arr[0] == pre_last_point).any():
            filaments_struct_identifier[filament_name] = 2
        else:
            # While first time starting_point becomes False, we assume continues coordinates are not axon part
            starting_point = False
            filaments_struct_identifier[filament_name] = 3
        pre_last_point = arr[-1]
        coordinate_idx += 1

    return filaments_coord, filaments_struct_identifier


def fill_swc_recursive(filament_props, current_filament_idx, parent_coord_idx, starting_coord_idx, swc_txt):
    # Starting from the global parent filament, recursively fill the text string for swc file
    # Add content to the swc text string
    filament_identity = filament_props["identity"][filament_props["names"][current_filament_idx]]
    is_first_coord = True
    for coord in filament_props["coords"][filament_props["names"][current_filament_idx]]:
        if is_first_coord:
            new_line = f'{starting_coord_idx} {filament_identity} {coord[0]} {coord[1]} {coord[2]} {0.1} {parent_coord_idx}\n'
            is_first_coord = False
        else:
            new_line = f'{starting_coord_idx} {filament_identity} {coord[0]} {coord[1]} {coord[2]} {0.1} {starting_coord_idx - 1}\n'
        starting_coord_idx += 1
        swc_txt += new_line

    # Find the children filaments
    current_filament_tail = filament_props["ends"][current_filament_idx]
    child_filament_list = np.where((filament_props["starts"] == current_filament_tail).all(axis=1))[0]

    if np.size(child_filament_list) == 0:
        # If there is no child, terminate the recursive loop and return the swc_txt
        # print('Loose End Found: ' + filament_names[current_filament_idx])
        return swc_txt, starting_coord_idx
    else:
        parent_coord_idx = starting_coord_idx - 1
        for child_filament_idx in child_filament_list:
            swc_txt, starting_coord_idx = fill_swc_recursive(filament_props, child_filament_idx, parent_coord_idx,
                                                             starting_coord_idx, swc_txt)
        return swc_txt, starting_coord_idx


def dic2swc(filaments_coord, filaments_struct_identifier):
    # Make filament properties to a dictionary
    filament_props = {}
    filament_props["names"] = list(filaments_coord.keys())
    filament_props["identity"] = filaments_struct_identifier
    filament_props["coords"] = filaments_coord
    filament_props["starts"] = []
    filament_props["ends"] = []

    # Make a list of filament starts and ends
    for filament_name in filament_props["names"]:
        filament_props["starts"] = np.append(filament_props["starts"], filament_props["coords"][filament_name][0])
        filament_props["ends"] = np.append(filament_props["ends"], filament_props["coords"][filament_name][-1])
    filament_props["starts"] = filament_props["starts"].reshape((-1, 3))
    filament_props["ends"] = filament_props["ends"].reshape((-1, 3))

    # Find the global parent filament of all filaments
    parent_filament_idx = 0
    while True:
        child_filament_head = filament_props["starts"][parent_filament_idx]
        parent_filament_list = np.where((filament_props["ends"] == child_filament_head).all(axis=1))[0]
        if np.size(parent_filament_list) == 0:
            # print('Global parent filament found: ' + filament_props["names"][parent_filament_idx])
            break
        else:
            # print('Searching for global parent filament... Now at ' + filament_props["names"][parent_filament_idx])
            parent_filament_idx = parent_filament_list[0]

    swc_txt, _ = fill_swc_recursive(filament_props, parent_filament_idx, -1, 1, '')

    return swc_txt


def iv2swc(iv_file_path):
    # Convert neural-traces in iv format to the swc format.
    # Input: iv_file_path: path to your iv file
    # An swc file, with the same name as the original iv file but with an *.swc extension will be created

    print('Converting ', iv_file_path, ' to *.swc format...', end="", flush=True)
    # read iv data
    with open(iv_file_path, 'r') as f:
        file_text = f.read()
    # convert iv file content to a dictionary
    filaments_coord, filaments_struct_identifier = iv2dic(file_text)
    # convert iv dic to swc text
    swc_txt = dic2swc(filaments_coord, filaments_struct_identifier)
    print("[DONE]")
    swc_file_path = iv_file_path[:-3] + '.swc'

    # Open a file for writing and write the file_contents string to it
    print('Writing ', swc_file_path)
    with open(swc_file_path, 'w') as f:
        f.write(swc_txt)
    return swc_file_path
