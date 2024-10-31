import os
from music21 import converter, note, chord
import numpy as np
from tensorflow.keras.utils import to_categorical
import pickle

# Directory containing the MIDI files
midi_directory = 'Classical_MIDI'

notes = []

# Traverse through all directories and subdirectories
for root, dirs, files in os.walk(midi_directory):
    for filename in files:
        if filename.endswith(".mid") or filename.endswith(".MIDI"):
            file_path = os.path.join(root, filename)  # Full file path
            print(f"Processing file: {file_path}")  # Debugging: print current file being processed
            try:
                midi = converter.parse(file_path)

                # Flatten the MIDI structure
                midi_flat = midi.flatten()  # Use .flatten() instead of .flat

                # Extract notes and chords
                file_notes = []
                for element in midi_flat.notes:
                    if isinstance(element, note.Note):
                        file_notes.append(str(element.pitch))  # Extract note
                    elif isinstance(element, chord.Chord):
                        file_notes.append('.'.join(str(n) for n in element.normalOrder))  # Extract chord

                notes.extend(file_notes)  # Add the file's notes to the main list
                print(f"Extracted {len(file_notes)} notes/chords from {filename}")  # Debugging: number of notes/chords extracted
                print(f"Sample notes from {filename}: {file_notes[:10]}")  # Debugging: print first 10 notes for validation

            except Exception as e:
                print(f"Error parsing {filename}: {e}")  # Error handling for faulty files

# Check if we have any notes before proceeding
if len(notes) == 0:
    print("No notes were extracted. Exiting.")
    exit()

# Create a mapping of notes/chords to integers
unique_notes = sorted(list(set(notes)))
note_to_int = {note: number for number, note in enumerate(unique_notes)}

# Convert notes to integers
sequence_data = [note_to_int[note] for note in notes]

# Create sequences for LSTM input
sequence_length = 100
input_sequences = []
output_notes = []

# Check if enough notes exist for sequence creation
if len(sequence_data) <= sequence_length:
    print(f"Not enough data for sequence creation. Total notes: {len(sequence_data)}")
    exit()

for i in range(0, len(sequence_data) - sequence_length):
    input_sequences.append(sequence_data[i:i + sequence_length])
    output_notes.append(sequence_data[i + sequence_length])

# Reshape the input data for the LSTM
X = np.reshape(input_sequences, (len(input_sequences), sequence_length, 1))
X = X / float(len(unique_notes))  # Normalize input

# Check if output_notes is empty before proceeding
if len(output_notes) == 0:
    print("No output notes were created. Exiting.")
    exit()

# One-hot encode the output
y = to_categorical(output_notes)

# Print some basic information about the processed data
print(f"Total notes extracted: {len(notes)}")
print(f"Unique notes: {len(unique_notes)}")
print(f"Input sequences: {len(input_sequences)}")
print(f"Output labels: {len(output_notes)}")

# Save input sequences (X), output labels (y), and the note-to-integer mapping
with open('Complete_Dataset/input_sequences.pkl', 'wb') as f:
    pickle.dump(X, f)

with open('Complete_Dataset/output_labels.pkl', 'wb') as f:
    pickle.dump(y, f)

with open('Complete_Dataset/note_to_int.pkl', 'wb') as f:
    pickle.dump(note_to_int, f)
