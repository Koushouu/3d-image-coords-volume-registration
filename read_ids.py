import javabridge
import bioformats


def read_ids(ids_path):
    print('Reading ', ids_path, '...', end="", flush=True)
    # Start the Java virtual machine
    javabridge.start_vm(class_path=bioformats.JARS)

    # Import metadata
    omexmlstr = bioformats.get_omexml_metadata(path=ids_path)
    o = bioformats.OMEXML(omexmlstr)
    pixels = o.image().Pixels

    # Initialize array: image will be saved in zyxc format (Scikit image convention)
    result_array = np.empty([pixels.SizeZ, pixels.SizeY, pixels.SizeX, pixels.SizeC])

    # Import data
    for ch in range(0, pixels.SizeC):
        for pln in range(0, pixels.SizeZ):
            img = bioformats.load_image(ids_path, c=ch, z=pln)
            result_array[pln, :, :, ch] = img
    # Terminate the Java virtual machine
    javabridge.kill_vm()
    print("[DONE]")
    return result_array