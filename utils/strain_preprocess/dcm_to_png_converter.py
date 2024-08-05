import os
import pydicom
from PIL import Image
import glob


def convert_dicom_to_png(dicom_file_path, output_folder):
    try:
        # Read the DICOM file
        ds = pydicom.dcmread(dicom_file_path, force=True)
        
        # Ensure that the file metadata includes a TransferSyntaxUID
        if not hasattr(ds, 'file_meta') or 'TransferSyntaxUID' not in ds.file_meta:
            if not hasattr(ds, 'file_meta'):
                ds.file_meta = pydicom.dataset.FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            
        # Check if the DICOM file contains pixel data
        if not hasattr(ds, 'PixelData'):
            print(f"No Pixel Data in {dicom_file_path}")
            return

        # Get the pixel data and determine the image mode
        pixel_array = ds.pixel_array
        mode = 'L'

        if pixel_array.ndim == 2:
            mode = 'L'
        elif pixel_array.ndim == 3 and pixel_array.shape[2] == 3:
            mode = 'RGB'
        else:
            raise ValueError(f"Unsupported number of dimensions or color channels in {dicom_file_path}")

        # Convert the pixel data to an image
        image = Image.fromarray(pixel_array, mode)
        base_name = os.path.basename(dicom_file_path)
        output_path = os.path.join(output_folder, base_name.replace('.dcm', '.png'))
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        image.save(output_path)
        print(f"Image saved at {output_path}")

    except Exception as e:
        print(f"Failed to process {dicom_file_path}: {e}")


def process_directory(input_base_directory, output_base_directory):
    # Use glob to find all DICOM files recursively
    dicom_files = glob.glob(os.path.join(input_base_directory, '**', '*.dcm'), recursive=True)
    
    for dicom_file_path in dicom_files:
        # Determine the output path relative to the input base directory
        relative_path = os.path.relpath(dicom_file_path, input_base_directory)
        output_folder = os.path.join(output_base_directory, os.path.dirname(relative_path))
        
        # Convert DICOM to PNG
        convert_dicom_to_png(dicom_file_path, output_folder)


input_base_directory = './Strain'
output_base_directory = './strain_png'
process_directory(input_base_directory, output_base_directory)