import argparse
from pathlib import Path
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset', type=str, required=True)
    args = parser.parse_args()
    train_set_path = Path(args.path_to_dataset)/'DATA/sets/TRAINSET'
    train_mos_list_path = Path(args.path_to_dataset)/'DATA/sets/train_mos_list.txt'
    shutil.copyfile( train_mos_list_path, train_mos_list_path.with_suffix('.txt.bak'))
    shutil.copyfile( train_set_path, train_set_path.with_suffix('.bak'))
    with open(train_mos_list_path) as f:
        lines = f.readlines()
    new_lines =[]
    for line in lines:
        if 'sys4bafa-uttc2e86f6.wav' in line:
            continue
        new_lines.append(line)
    with open(train_mos_list_path, 'w') as f:
        f.writelines(new_lines)
    with open(train_set_path) as f:
        lines = f.readlines()
    new_lines =[]
    for line in lines:
        if 'sys4bafa-uttc2e86f6.wav' in line:
            continue
        new_lines.append(line)
    with open(train_set_path, 'w') as f:
        f.writelines(new_lines)
    



