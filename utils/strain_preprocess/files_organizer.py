import os
import glob
import re
import shutil


def organize_files_by_alpha(base_dir):
    """
    Organizes files into directories based on the first alphabetical character in their filenames.
    """
    categories = ['G', 'S']
    scores = ['score1', 'score2', 'score3', 'score4', 'score5']
    
    for category in categories:
        for score in scores:
            os.makedirs(os.path.join(base_dir, category, score), exist_ok=True)
    
    for score_dir in scores:
        current_dir = os.path.join(base_dir, score_dir)
        
        if os.path.isdir(current_dir):
            for filename in os.listdir(current_dir):
                if filename.lower().endswith('.png'):
                    match = re.search("[a-zA-Z]", filename)
                    if match:
                        first_alpha = match.group(0).upper()
                        source_path = os.path.join(current_dir, filename)
                        
                        if first_alpha == 'G' or first_alpha == 'S':
                            dest_path = os.path.join(base_dir, first_alpha, score_dir, filename)
                            shutil.copy(source_path, dest_path)
                            print(f"Copied {source_path} to {dest_path}")


def move_files_containing_E(base_dir):
    """
    Moves files with 'E' in their filenames from SCORE 1 to SCORE 5 to their respective directories.
    """
    output_dir = os.path.join(base_dir, "S/score")

    for score in range(1, 6):
        output_path = os.path.join(output_dir + str(score))
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        input_path = os.path.join(base_dir, f"SCORE {score}", "**", "*E*.png")
        images = glob.glob(input_path, recursive=True)

        for image_path in images:
            file_name = os.path.basename(image_path)
            output_path = os.path.join(output_dir + str(score), file_name)
            shutil.move(image_path, output_path)
            print(f"Moved {image_path} to {output_path}")
        
    print("All file moves are complete.")


def main():
    base_directory = './strain_png'

    print("Select an option:")
    print("1. Organize files by first alphabetical character (G/S).")
    print("2. Move files containing 'E' in filename from SCORE 1 to SCORE 5.")
    
    choice = input("Enter the number of your choice (1 or 2): ")

    if choice == '1':
        organize_files_by_alpha(base_directory)
    elif choice == '2':
        move_files_containing_E(base_directory)
    else:
        print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()