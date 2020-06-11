Commands for running on virgo:

# Step 1: train/test split
In the folder src/data/ run

pyl3 train_test_split.py --data-dir /nfssys/shine0/data/VGGSound/data/vggsound --output-dir /nfssys/shine0/data/VGGSound/interim --csv /nfssys/shine0/data/VGGSound/audioset_ds/VGGSound/vggsound.csv

# Step 2: data pre-processinig

pyl3 batch_extraction.py --data-dir /nfssys/shine0/data/VGGSound/interim --output-dir /nfssys/shine0/data/VGGSound/processed 
