import argparse
import os
import csv
import shutil
 
def parse_arguments():
    parser = argparse.ArgumentParser(description='Moves data from a single folder to train test folder')
    parser.add_argument('--data-dir',
                        action='store',
                        type=str,
                        help='Path to directory where data files are stored')
 
    parser.add_argument('--output-dir',
                        action='store',
                        type=str,
                        help='Path to directory where output files will be stored')
 
    parser.add_argument('--csv',
                        action='store',
                        type=str,
                        help='CSV path')
 
    return parser.parse_args()
 
 
if __name__ == '__main__':
 
  cmd = 'cp '
#   cmd = 'mv '
 
  args = parse_arguments()
 
  output_dir = dict()
  output_dir = {'train': os.path.join(args.output_dir, 'train'),
                'test' : os.path.join(args.output_dir, 'test')}
 
  #create destination directories
  os.makedirs(args.output_dir, exist_ok=True)
 
  os.makedirs(output_dir['train'], exist_ok=True)
  os.makedirs(os.path.join(output_dir['train'], 'audio'), exist_ok=True)
  os.makedirs(os.path.join(output_dir['train'], 'video'), exist_ok=True)
 
  os.makedirs(output_dir['test'], exist_ok=True)
  os.makedirs(os.path.join(output_dir['test'], 'audio'), exist_ok=True)
  os.makedirs(os.path.join(output_dir['test'], 'video'), exist_ok=True)
 
  with open(args.csv, 'r') as f:
    data_reader = csv.reader(f)
    for row_idx, row in enumerate(data_reader):
        # Skip commented lines
 
        if row[0][0] == '#':
            continue
        # input(os.path.join(args.data_dir, 'audio', row[0]))
 
        os.system(cmd + os.path.join(args.data_dir, 'audio', row[0] + u'*') + ' ' + os.path.join(output_dir[row[3]], 'audio') )
        os.system(cmd + os.path.join(args.data_dir, 'video', row[0] + u'*') + ' ' + os.path.join(output_dir[row[3]], 'video') )