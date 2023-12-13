# Download and extract 1-1 molecular transformations
python -m preprocessing.download_and_extract_transformations.py

# Extract centres and signatures as graph edits
python -m preprocessing.extract_centre_and_signatures.py

# generate an action dataset (filtering the useful ones from previous output based on various criteria)
python -m preprocessing.generate_action_dataset.py

# filter action dataset for problematic actions
python -m preprocessing.filter_action_dataset.py
python -m preprocessing.filter_action_dataset_2.py

