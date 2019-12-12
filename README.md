### Command on virgo for 01_create_subsets.py

python3 01_create_subsets.py --ontology-path ontology/ontology.json --metadata-path metadata/balanced_train_segments.csv --data-dir /hardmnt/virgo1/data/fpaissan/data/data/balanced_train_segments --output-dir /hardmnt/virgo1/data/fpaissan/data/output --filename-prefix data


---------------------------------------------------------------

### Command on virgo for 02_generate_samples.py

python3 02_generate_samples.py --batch-size 64 --num-streamers 64 --num-workers 1 --output-dir /hardmnt/virgo1/data/fpaissan/data/02_out/ --subset-path /hardmnt/virgo1/data/fpaissan/data/output/data_train.csv --num-samples 1000

By running the code with that command, the code is not really optimised (--num-workers can be higher). Problems occur when --num-workers > 1 while using ffprobe, due to pescador streamer (it seems).


---------------------------------------------------------------


This repo is imported from https://github.com/marl/l3embedding. Refer to that repo for further understanding of work.