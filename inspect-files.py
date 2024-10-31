import pickle

# Load input sequences
with open('/Users/cristovalneosasono/AI_FinalProject_V2/Subset_Dataset/input_sequences_durations.pkl', 'rb') as f:
    X = pickle.load(f)
print(f"Shape of X: {X.shape}")
print(f"First input sequence: {X[0]}")

# Load output labels
with open('Complete_Dataset/output_labels.pkl', 'rb') as f:
    y = pickle.load(f)
print(f"Shape of y: {y.shape}")
print(f"First output label: {y[0]}")

# Load the note-to-int mapping
with open('Complete_Dataset/note_to_int.pkl', 'rb') as f:
    note_to_int = pickle.load(f)
print(f"Note-to-int mapping: {note_to_int}")
