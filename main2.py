########################################################################################################################
# Import libraries
from skimage.io import imread, imsave
from utility import find_files, get_pixel_size, scale_image, get_axon_dendrite_for_napari, read_coord_csv, \
    um_to_pixel, get_transform_matrix, linear_transform_image, linear_transform_coord, linear_transform_swc, \
    make_directory, pixel_to_um, write_coord_csv, downsample, tps_transform_image, tps_transform_swc, save_image
from iv2swc import iv2swc
import numpy as np
import napari
import os

########################################################################################################################
# User inputs
home_directory = r"D:\drosophila_visual_trace_data\Toll 9_7_RV"
target_coord_path = r"D:\drosophila_visual_trace_data\target_coordinates\target_coordinates_RV.csv"
transform_type = 'affine'  # Must be 'affine' or 'rigid'
do_tps = True
image_bin_factor = 2  # Factors for binning the image for transformation
align_target_coord = True
napari_display = True

########################################################################################################################
# Data Loading and Preprocessing

# Create a directory to save all the transformed results
transformed_results_dir = os.path.join(home_directory, 'transformed_results')
make_directory(transformed_results_dir)

# Find all the necessary files
ics_file_path = find_files(home_directory, '.ics')
csv_file_path = find_files(home_directory, '.csv')
iv_file_path = find_files(home_directory, '.iv')
swc_file_path = find_files(home_directory, '.swc')
ids_file_path = find_files(home_directory, '.ids')
tif_file_path = find_files(home_directory, '.tif')

# Read image file
# if tif does not exist, it will assume that ids exists, and will convert ids to tif and save
if tif_file_path is None:
    from read_ids import read_ids
    print("Converting IDS to TIF ...")
    image = read_ids(ids_file_path)
    # Save the ids file as tif
    imsave(ids_file_path[:-4] + '.tif', image)
    tif_file_path = find_files(home_directory, '.tif')
else:
    # Read tif file
    image = imread(tif_file_path)
    # Image J saves image in the format ZCYX however in scikit image processed as ZYXC
    image = np.transpose(image, (0, 2, 3, 1))

# Get pixel size, and downsample if needed (Optional)
# Get pixel size
pixel_size = get_pixel_size(ics_file_path)
# Scale pixel size by the downsample bin factor
pixel_size = pixel_size * image_bin_factor
# Down sample the image
image = downsample(image, image_bin_factor)
# Scale the original image such that all axis have the save pixel size
image_scaled, pixel_size = scale_image(image, pixel_size)

# if swc file does not exist, convert iv to swc
if swc_file_path is None:
    swc_file_path = iv2swc(iv_file_path)

# Read control and target coordinates in csv file, and convert both from um to pixels
control_coord = read_coord_csv(csv_file_path)
target_coord = read_coord_csv(target_coord_path)
control_coord_pixels = um_to_pixel(control_coord, pixel_size)
target_coord_pixels = um_to_pixel(target_coord, pixel_size)

########################################################################################################################
# Target coordinates alignment

if align_target_coord:
    # Get transform matrix
    transform_matrix = get_transform_matrix(control_coord_pixels, target_coord_pixels, "rigid")
    # Perform transformation
    target_coord_pixels = linear_transform_coord(target_coord_pixels, transform_matrix)
    # Save the new target coordinates results
    target_coord = pixel_to_um(target_coord_pixels, pixel_size)
    transformed_target_csv_file_path = os.path.join(transformed_results_dir, 'aligned_target_coord.csv')
    write_coord_csv(transformed_target_csv_file_path, target_coord)

########################################################################################################################
#   Linear Transformation

# Transforming image volumes
# Get transform matrix for image transformation
transform_matrix = get_transform_matrix(control_coord_pixels, target_coord_pixels, transform_type)
# Transform image data and save results
image_LT = linear_transform_image(image_scaled, transform_matrix)
save_image(os.path.join(transformed_results_dir, transform_type + '_transformed_image.tif'), image_LT)

# Transforming coordinates
# Get transform matrix for coordinates transformation
transform_matrix = get_transform_matrix(target_coord_pixels, control_coord_pixels, transform_type)
# Transform neurites data and save results
linear_transformed_swc_path = os.path.join(transformed_results_dir, transform_type + '_transformed_neurites.swc')
linear_transform_swc(swc_file_path, transform_matrix, pixel_size, linear_transformed_swc_path)

# Transform control coord data and save results
linear_transformed_control_coord_pixels = linear_transform_coord(control_coord_pixels, transform_matrix)
linear_transformed_control_coord = pixel_to_um(linear_transformed_control_coord_pixels, pixel_size)
write_coord_csv(os.path.join(transformed_results_dir, transform_type + '_transformed_control_coord.csv'),
                linear_transformed_control_coord)

########################################################################################################################
# Non-linear Transform (TPS)

# TPS transform the image
image_TPS, tps_fun = tps_transform_image(image_LT, linear_transformed_control_coord_pixels,
                                         target_coord_pixels)
save_image(os.path.join(transformed_results_dir, 'tps' + '_transformed_image.tif'), image_TPS)
# TPS  transform the SWC
tps_transformed_swc_path = os.path.join(transformed_results_dir, 'tps' + '_transformed_neurites.swc')
tps_transform_swc(linear_transformed_swc_path, linear_transformed_control_coord_pixels, target_coord_pixels,
                  pixel_size, tps_transformed_swc_path)

########################################################################################################################
# Display results

if napari_display:
    # Convert swc files to Napari displayable format
    axon, dendrite = get_axon_dendrite_for_napari(swc_file_path, pixel_size)
    axon_LT, dendrite_LT = get_axon_dendrite_for_napari(linear_transformed_swc_path, pixel_size)
    axon_TPS, dendrite_TPS = get_axon_dendrite_for_napari(tps_transformed_swc_path, pixel_size)

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(image_scaled[:, :, :, 0])
    viewer.add_image(image_scaled[:, :, :, 1])
    viewer.add_shapes(axon, shape_type='path', edge_width=1.0, edge_color=["#ca6e38"])
    viewer.add_shapes(dendrite, shape_type='path', edge_width=1.0, edge_color=["#a7993f"])
    viewer.add_points(control_coord_pixels, size=10, face_color=["#cb5462"])

    viewer.add_image(image_LT[:, :, :, 0])
    viewer.add_image(image_LT[:, :, :, 1])
    viewer.add_shapes(axon_LT, shape_type='path', edge_width=1.0, edge_color=["#44bdc1"])
    viewer.add_shapes(dendrite_LT, shape_type='path', edge_width=1.0, edge_color=["#62b64c"])
    viewer.add_points(linear_transformed_control_coord_pixels, size=10, face_color=["#559961"])

    if do_tps is True:
        viewer.add_image(image_TPS[:, :, :, 0])
        viewer.add_image(image_TPS[:, :, :, 1])
        viewer.add_shapes(axon_TPS, shape_type='path', edge_width=1.0, edge_color=["#c65ea0"])
        viewer.add_shapes(dendrite_TPS, shape_type='path', edge_width=1.0, edge_color=["#6e87cc"])
        viewer.add_points(target_coord_pixels, size=10, face_color=["#9061cb"])
    napari.run()

print("DONE")
