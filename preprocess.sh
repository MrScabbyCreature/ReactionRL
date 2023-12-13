# Download and extract 1-1 molecular transformations
python -m preprocessing.download_and_extract_transformations

# Extract centres and signatures as graph edits
python -m preprocessing.extract_centre_and_signatures

# generate an action dataset (filtering the useful ones from previous output based on various criteria)
python -m preprocessing.generate_action_dataset

# filter action dataset for problematic actions
python -m preprocessing.filter_action_dataset
python -m preprocessing.filter_action_dataset_2

# Dump start molecules 
python -m preprocessing.dump_start_mols

# Dump action embeddings - Repetitive (and very expensive) to compute during run-time 
python -m preprocessing.dump_action_embeddings