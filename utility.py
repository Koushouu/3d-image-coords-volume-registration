import csv
import os
import numpy as np
import neurom as nm
from scipy.ndimage import zoom
from transforms3d._gohlketransforms import affine_matrix_from_points
from scipy import ndimage as ndi
from skimage.measure import block_reduce
from tps import ThinPlateSpline
from scipy.ndimage import map_coordinates
from tifffile import imwrite


def make_directory(new_directory_path):
    is_exists = os.path.exists(new_directory_path)
    if not is_exists:
        os.mkdir(new_directory_path)
        print(new_directory_path, " is created")
    else:
        print(new_directory_path, " already exists")


def find_files(home_directory, file_extension):
    file_found = 0
    for file in os.listdir(home_directory):
        if file.endswith(file_extension):
            file_found = file_found + 1
            file_name = file
            print(file_extension, " file found: ", file_name)
            file_path = os.path.join(home_directory, file_name)
    if file_found == 0:
        print("WARNING: ", file_extension, " file not found in ", home_directory)
        file_path = None
    return file_path


def get_pixel_size(ics_path):
    with open(ics_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        ics = list(reader)
    pixel_size_x = float(ics[13][3])
    pixel_size_y = float(ics[13][4])
    pixel_size_z = float(ics[13][5])
    pixel_size = [pixel_size_z, pixel_size_y, pixel_size_x]
    return np.array(pixel_size)


def scale_image(image, pixel_size):
    # Scale the original image such that all axis have the save pixel size (pixel_size_x)
    print('Scaling images so that pixel sizes of all dimension = ', pixel_size[2], 'um (equivalent to x pixel size)')
    zoom_z = pixel_size[0] / pixel_size[2]
    zoom_y = pixel_size[1] / pixel_size[2]
    zoom_x = pixel_size[2] / pixel_size[2]
    print("\tScaling for ch0...", end="", flush=True)
    image_scaled_ch0 = zoom(image[:, :, :, 0], (zoom_z, zoom_y, zoom_x))
    print("[DONE]")
    print("\tScaling for ch1...", end="", flush=True)
    image_scaled_ch1 = zoom(image[:, :, :, 1], (zoom_z, zoom_y, zoom_x))
    print("[DONE]")
    image_scaled = np.stack((image_scaled_ch0, image_scaled_ch1), axis=3)
    # Get new pixel size
    pixel_size = [pixel_size[2], pixel_size[2], pixel_size[2]]
    return image_scaled, pixel_size


def um_to_pixel(um_data, pixel_size):
    pixel_data = np.zeros_like(um_data)
    pixel_data[:, 0] = um_data[:, 0] / pixel_size[0]
    pixel_data[:, 1] = um_data[:, 1] / pixel_size[1]
    pixel_data[:, 2] = um_data[:, 2] / pixel_size[2]
    return pixel_data


def pixel_to_um(pixel_data, pixel_size):
    um_data = np.zeros_like(pixel_data)
    um_data[:, 0] = pixel_data[:, 0] * pixel_size[0]
    um_data[:, 1] = pixel_data[:, 1] * pixel_size[1]
    um_data[:, 2] = pixel_data[:, 2] * pixel_size[2]
    return um_data


def get_axon_dendrite_for_napari(swc_path, pixel_size):
    m = nm.load_morphology(swc_path)
    axon = []
    dendrite = []
    for n in m.neurites:
        for section in n.iter_sections():
            coord = section.points[:, :3]
            coord = um_to_pixel(coord, pixel_size)
            coord = np.flip(coord, axis=1)
            if section.type == 2:
                axon.append(coord)
            elif section.type == 3:
                dendrite.append(coord)
    return axon, dendrite


def read_coord_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=",")
        csv_data = list(reader)
    # Get rid of the last column data
    csv_data = np.array(csv_data)
    csv_data = csv_data[:, 0:3]
    csv_data = np.array(csv_data, dtype=float)
    # Re-order the axis to zyx
    csv_data = np.flip(csv_data, axis=1)
    return csv_data


def write_coord_csv(csv_path, csv_data):
    csv_data = np.flip(csv_data, axis=1)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerows(csv_data)


def get_transform_matrix(control_coord, target_coord, transform_type):
    print("Calculating the transformation matrix... ", end="", flush=True)
    if transform_type == "affine":
        transform_matrix = affine_matrix_from_points(target_coord.T, control_coord.T)
    elif transform_type == "rigid":
        transform_matrix = affine_matrix_from_points(target_coord.T, control_coord.T, shear=False, scale=False)
    else:
        print("Unknown transform_type (must be affine or rigid)")
    print("[DONE]")
    return transform_matrix


def linear_transform_image(image, transform_matrix):
    print("Performing linear transformation of a given image based on the transformation matrix: ")
    image_after_transform = np.zeros_like(image)
    print("\tLinearly transforming ch0... ", end="", flush=True)
    image_after_transform[:, :, :, 0] = ndi.affine_transform(image[:, :, :, 0], transform_matrix)
    print("[DONE]")
    print("\tLinearly transforming ch1... ", end="", flush=True)
    image_after_transform[:, :, :, 1] = ndi.affine_transform(image[:, :, :, 1], transform_matrix)
    print("[DONE]")
    return image_after_transform


def linear_transform_coord(coord, transform_matrix):
    print("Performing linear transformation of the given coordinates based on the transformation matrix... ", end="",
          flush=True)
    coord_for_transformation = np.copy(coord.T)
    coord_for_transformation = np.append(coord_for_transformation, np.ones((1, coord_for_transformation.shape[1])),
                                         axis=0)
    transformed_coord_transposed = np.matmul(transform_matrix, coord_for_transformation)
    print("[DONE]")
    return transformed_coord_transposed[:3].T


def linear_transform_swc(swc_file_path, transform_matrix, pixel_size, transformed_swc_path):
    print("Performing linear transformation of the given neurite data based on the transformation matrix... ", end="",
          flush=True)
    # Read the neurite coordinates
    swc_data = np.loadtxt(swc_file_path, delimiter=' ')
    # extract only the coordinates part
    swc_coord = swc_data[:, 2:5]
    # Convert to ZYX format for processing
    swc_coord = np.flip(swc_coord, axis=1)
    # Convert to pixel unit
    swc_coord = um_to_pixel(swc_coord, pixel_size)
    # Perform affine transform
    swc_coord_transformed = linear_transform_coord(swc_coord, transform_matrix)
    # Convert it back to um unit
    swc_coord_transformed = pixel_to_um(swc_coord_transformed, pixel_size)
    # Convert it back to XYZ format for saving
    swc_coord_transformed = np.flip(swc_coord_transformed, axis=1)
    # Put it back to the swc_data
    swc_data[:, 2:5] = swc_coord_transformed
    # Save the result as a swc file
    fmt = '%d', '%d', '%1.6f', '%1.6f', '%1.6f', '%1.1f', '%d'
    print('Writing ', transformed_swc_path)
    np.savetxt(transformed_swc_path, swc_data, fmt=fmt, delimiter=' ')
    print("[DONE]")
    return transformed_swc_path


def downsample(image, bin_factor):
    # Downsample image with dimensions ZYXC
    print('Downsampling the image...', end="", flush=True)
    image = block_reduce(image, block_size=(bin_factor, bin_factor, bin_factor, 1), func=np.mean)
    print('[DONE]')
    return image


def tps_transform_image(image, control_coord, target_coord):
    # Create the 3d meshgrid of indices of output image; Shape: (Z, Y, X, 3)
    z_size, y_size, x_size = image.shape[0:3]
    output_indices = np.indices((z_size, y_size, x_size), dtype=np.float64).transpose(1, 2, 3, 0)
    # Transform it into the input indices
    tps_fun = ThinPlateSpline(0.5)
    tps_fun.fit(target_coord, control_coord)
    input_indices = tps_fun.transform(output_indices.reshape(-1, 3)).reshape(z_size, y_size, x_size, 3)
    # Interpolate the resulting image
    img_transformed = np.concatenate(
        [
            map_coordinates(np.array(image)[..., channel], input_indices.transpose(3, 0, 1, 2))[..., None]
            for channel in (0, 1)
        ],
        axis=-1,
    )
    return img_transformed, tps_fun


def tps_transform_swc(swc_file_path, control_coord, target_coord, pixel_size, transformed_swc_path):
    # NB: pre-process and post-process procedure basically the same as affine_transform_swc - consider combine the
    # two later
    print("Performing TPS transformation of the given neurite data based on the tps object... ", end=""
          , flush=True)
    # Read the neurite coordinates
    swc_data = np.loadtxt(swc_file_path, delimiter=' ')
    # extract only the coordinates part
    swc_coord = swc_data[:, 2:5]
    # Convert to ZYX format for processing
    swc_coord = np.flip(swc_coord, axis=1)
    # Convert to pixel unit
    swc_coord = um_to_pixel(swc_coord, pixel_size)
    # Perform transformation
    tps_fun = ThinPlateSpline(0.5)
    tps_fun.fit(control_coord, target_coord)
    swc_coord_transformed = tps_fun.transform(swc_coord)
    # Convert it back to um unit
    swc_coord_transformed = pixel_to_um(swc_coord_transformed, pixel_size)
    # Convert it back to XYZ format for saving
    swc_coord_transformed = np.flip(swc_coord_transformed, axis=1)
    # Put it back to the swc_data
    swc_data[:, 2:5] = swc_coord_transformed
    # Save the result as a swc file
    fmt = '%d', '%d', '%1.6f', '%1.6f', '%1.6f', '%1.1f', '%d'
    print('Writing ', transformed_swc_path)
    np.savetxt(transformed_swc_path, swc_data, fmt=fmt, delimiter=' ')
    print("[DONE]")
    return transformed_swc_path


def save_image(image_path, image):
    # Convert image to UINT16 format
    data_min = np.min(image)
    data_max = np.max(image)
    data_range = data_max - data_min
    data_scale = 65535.0 / data_range
    image_scaled = (image - data_min) * data_scale
    image_scaled_uint8 = image_scaled.astype('uint16')
    # Convert image to ZCYX format (ImageJ)
    image_scaled_uint8_zcyx = image_scaled_uint8.transpose(0, 3, 1, 2)
    # Write image
    imwrite(image_path, image_scaled_uint8_zcyx, imagej=True, metadata={'axes': 'ZCYX'})
