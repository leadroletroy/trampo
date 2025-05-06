import os
import shutil
from pathlib import Path


def reassemble_folders(directory, cams, subdir='images'):
    for dir in os.listdir(directory):
        if str(dir).split('-') in cams:
            pass

    main_dir = Path(directory)
    output_dir = main_dir / 'images'

    output_dir.mkdir(exist_ok=True)

    for subdir in main_dir.iterdir():
        if subdir.is_dir():
            suffix = subdir.name.split('-')[-1]
            target_dir = output_dir / suffix
            target_dir.mkdir(exist_ok=True)

            for file in subdir.glob('*'):
                if file.is_file():
                    shutil.copy(file, target_dir / file.name)

    print('Files grouped succesfully!')