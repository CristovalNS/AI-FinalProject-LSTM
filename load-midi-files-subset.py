import os
import random
from music21 import converter, note, chord, meter, tempo, key
import numpy as np
from tensorflow.keras.utils import to_categorical
import pickle

# Directory containing the MIDI files
midi_directory = 'Classical_MIDI'

notes = []
durations = []  # List to store note/chord durations
tempos = []  # List to store tempos (BPM)
time_signatures = []  # List to store time signatures
key_signatures = []  # List to store key signatures
all_files = []

# Traverse through all directories and subdirectories to gather all file paths
for root, dirs, files in os.walk(midi_directory):
    for filename in files:
        if filename.endswith(".mid") or filename.endswith(".MIDI"):
            all_files.append(os.path.join(root, filename))  # Full file path

# Randomly select 10% of the files
subset_size = int(0.1 * len(all_files))  # Use 10% of total files
subset_files = random.sample(all_files, subset_size)

print(f"Processing {subset_size} MIDI files out of {len(all_files)} total files.")

# Process the selected subset of files
for file_path in subset_files:
    print(f"Processing file: {file_path}")  # Debugging: print current file being processed
    try:
        midi = converter.parse(file_path)

        # Flatten the MIDI structure
        midi_flat = midi.flatten()  # Use .flatten() instead of .flat

        # Extract tempo
        tempo_bpm = None
        for element in midi_flat:
            if isinstance(element, tempo.MetronomeMark):
                tempo_bpm = element.number  # Get the BPM of the piece
                break  # Stop after finding the first tempo marking

        if tempo_bpm is None:
            tempo_bpm = 120  # Default to 120 BPM if no tempo is found

        tempos.append(tempo_bpm)  # Store tempo (BPM)

        # Extract time signature
        time_sig = None
        for element in midi_flat:
            if isinstance(element, meter.TimeSignature):
                time_sig = element.ratioString  # Get the time signature in ratio form (e.g., '4/4')
                break  # Stop after finding the first time signature

        if time_sig is None:
            time_sig = "4/4"  # Default to 4/4 if no time signature is found

        time_signatures.append(time_sig)  # Store time signature

        # Extract key signature
        key_sig = None
        for element in midi_flat:
            if isinstance(element, key.KeySignature):
                key_sig = element.sharps  # Number of sharps or flats (negative for flats)
                break  # Stop after finding the first key signature

        if key_sig is None:
            key_sig = 0  # Default to C major / A minor if no key signature is found

        key_signatures.append(key_sig)  # Store key signature

        # Extract notes, chords, rests, and their durations
        file_notes = []
        file_durations = []
        for element in midi_flat.notesAndRests:
            if isinstance(element, note.Note):
                file_notes.append(str(element.pitch))  # Extract note
                file_durations.append(str(element.duration.quarterLength))  # Extract duration
            elif isinstance(element, chord.Chord):
                file_notes.append('.'.join(str(n) for n in element.normalOrder))  # Extract chord
                file_durations.append(str(element.duration.quarterLength))  # Extract duration
            elif isinstance(element, note.Rest):
                file_notes.append('rest')  # Handle rests
                file_durations.append(str(element.duration.quarterLength))  # Extract duration for rest

        notes.extend(file_notes)  # Add the file's notes/rests to the main list
        durations.extend(file_durations)  # Add the file's durations to the main list
        print(f"Extracted {len(file_notes)} notes/chords/rests from {file_path}")  # Debugging
        print(f"Sample notes from {file_path}: {file_notes[:10]}")  # Debugging
        print(f"Sample durations from {file_path}: {file_durations[:10]}")  # Debugging

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")  # Error handling for faulty files

# Ensure that tempos, time signatures, and key signatures match the length of notes
if len(tempos) < len(notes):
    tempos.extend([tempos[-1]] * (len(notes) - len(tempos)))  # Replicate last tempo
if len(time_signatures) < len(notes):
    time_signatures.extend([time_signatures[-1]] * (len(notes) - len(time_signatures)))  # Replicate last time signature
if len(key_signatures) < len(notes):
    key_signatures.extend([key_signatures[-1]] * (len(notes) - len(key_signatures)))  # Replicate last key signature


# Check if we have any notes before proceeding
if len(notes) == 0 or len(durations) == 0:
    print("No notes or durations were extracted. Exiting.")
    exit()

# Create mappings for notes/chords/rests, durations, tempos, time signatures, and key signatures to integers
unique_notes = sorted(list(set(notes)))
note_to_int = {note: number for number, note in enumerate(unique_notes)}

unique_durations = sorted(list(set(durations)))
duration_to_int = {duration: number for number, duration in enumerate(unique_durations)}

unique_tempos = sorted(list(set(tempos)))
tempo_to_int = {tempo: number for number, tempo in enumerate(unique_tempos)}

unique_time_signatures = sorted(list(set(time_signatures)))
time_signature_to_int = {time_sig: number for number, time_sig in enumerate(unique_time_signatures)}

unique_key_signatures = sorted(list(set(key_signatures)))
key_signature_to_int = {key_sig: number for number, key_sig in enumerate(unique_key_signatures)}

# Convert notes, durations, tempos, time signatures, and key signatures to integers
sequence_data_notes = [note_to_int[note] for note in notes]
sequence_data_durations = [duration_to_int[duration] for duration in durations]
sequence_data_tempos = [tempo_to_int[tempo] for tempo in tempos]
sequence_data_time_signatures = [time_signature_to_int[time_sig] for time_sig in time_signatures]
sequence_data_key_signatures = [key_signature_to_int[key_sig] for key_sig in key_signatures]

# Create sequences for LSTM input
sequence_length = 50
input_sequences_notes = []
output_notes = []

input_sequences_durations = []
output_durations = []

input_sequences_tempos = []
output_tempos = []

input_sequences_time_signatures = []
output_time_signatures = []

input_sequences_key_signatures = []
output_key_signatures = []

# Check if enough data exists for sequence creation
if len(sequence_data_notes) <= sequence_length:
    print(f"Not enough data for sequence creation. Total notes: {len(sequence_data_notes)}")
    exit()

# Ensure that all sequences (notes, durations, tempos, etc.) have the same length
min_length = min(len(sequence_data_notes), len(sequence_data_durations), len(sequence_data_tempos),
                 len(sequence_data_time_signatures), len(sequence_data_key_signatures))

# Create sequences for LSTM input
for i in range(0, min_length - sequence_length):
    input_sequences_notes.append(sequence_data_notes[i:i + sequence_length])
    output_notes.append(sequence_data_notes[i + sequence_length])

    input_sequences_durations.append(sequence_data_durations[i:i + sequence_length])
    output_durations.append(sequence_data_durations[i + sequence_length])

    input_sequences_tempos.append(sequence_data_tempos[i:i + sequence_length])
    output_tempos.append(sequence_data_tempos[i + sequence_length])

    input_sequences_time_signatures.append(sequence_data_time_signatures[i:i + sequence_length])
    output_time_signatures.append(sequence_data_time_signatures[i + sequence_length])

    input_sequences_key_signatures.append(sequence_data_key_signatures[i:i + sequence_length])
    output_key_signatures.append(sequence_data_key_signatures[i + sequence_length])

# Reshape the input data for the LSTM
X_notes = np.reshape(input_sequences_notes, (len(input_sequences_notes), sequence_length, 1))
X_notes = X_notes / float(len(unique_notes))  # Normalize input for notes

X_durations = np.reshape(input_sequences_durations, (len(input_sequences_durations), sequence_length, 1))
X_durations = X_durations / float(len(unique_durations))  # Normalize input for durations

X_tempos = np.reshape(input_sequences_tempos, (len(input_sequences_tempos), sequence_length, 1))
X_tempos = X_tempos / float(len(unique_tempos))  # Normalize input for tempos

X_time_signatures = np.reshape(input_sequences_time_signatures, (len(input_sequences_time_signatures), sequence_length, 1))
X_time_signatures = X_time_signatures / float(len(unique_time_signatures))  # Normalize input for time signatures

X_key_signatures = np.reshape(input_sequences_key_signatures, (len(input_sequences_key_signatures), sequence_length, 1))
X_key_signatures = X_key_signatures / float(len(unique_key_signatures))  # Normalize input for key signatures

# Check if output data is empty before proceeding
if len(output_notes) == 0 or len(output_durations) == 0:
    print("No output notes or durations were created. Exiting.")
    exit()

# One-hot encode the output (both notes, durations, tempos, time signatures, and key signatures)
y_notes = to_categorical(output_notes)
y_durations = to_categorical(output_durations)
y_tempos = to_categorical(output_tempos)
y_time_signatures = to_categorical(output_time_signatures)
y_key_signatures = to_categorical(output_key_signatures)

# Print some basic information about the processed data
print(f"Total notes extracted: {len(notes)}")
print(f"Unique notes: {len(unique_notes)}")
print(f"Input sequences (notes): {len(input_sequences_notes)}")
print(f"Output labels (notes): {len(output_notes)}")
print(f"Input sequences (durations): {len(input_sequences_durations)}")
print(f"Output labels (durations): {len(output_durations)}")
print(f"Input sequences (tempos): {len(input_sequences_tempos)}")
print(f"Output labels (tempos): {len(output_tempos)}")
print(f"Input sequences (time signatures): {len(input_sequences_time_signatures)}")
print(f"Output labels (time signatures): {len(output_time_signatures)}")
print(f"Input sequences (key signatures): {len(input_sequences_key_signatures)}")
print(f"Output labels (key signatures): {len(output_key_signatures)}")

print(f"Length of notes: {len(sequence_data_notes)}")
print(f"Length of durations: {len(sequence_data_durations)}")
print(f"Length of tempos: {len(sequence_data_tempos)}")
print(f"Length of time signatures: {len(sequence_data_time_signatures)}")
print(f"Length of key signatures: {len(sequence_data_key_signatures)}")

# Save input sequences and mappings
if not os.path.exists('Subset_Dataset'):
    os.makedirs('Subset_Dataset')

with open('Subset_Dataset/input_sequences_notes.pkl', 'wb') as f:
    pickle.dump(X_notes, f)

with open('Subset_Dataset/output_labels_notes.pkl', 'wb') as f:
    pickle.dump(y_notes, f)

with open('Subset_Dataset/input_sequences_durations.pkl', 'wb') as f:
    pickle.dump(X_durations, f)

with open('Subset_Dataset/output_labels_durations.pkl', 'wb') as f:
    pickle.dump(y_durations, f)

with open('Subset_Dataset/input_sequences_tempos.pkl', 'wb') as f:
    pickle.dump(X_tempos, f)

with open('Subset_Dataset/output_labels_tempos.pkl', 'wb') as f:
    pickle.dump(y_tempos, f)

with open('Subset_Dataset/input_sequences_time_signatures.pkl', 'wb') as f:
    pickle.dump(X_time_signatures, f)

with open('Subset_Dataset/output_labels_time_signatures.pkl', 'wb') as f:
    pickle.dump(y_time_signatures, f)

with open('Subset_Dataset/input_sequences_key_signatures.pkl', 'wb') as f:
    pickle.dump(X_key_signatures, f)

with open('Subset_Dataset/output_labels_key_signatures.pkl', 'wb') as f:
    pickle.dump(y_key_signatures, f)

with open('Subset_Dataset/note_to_int.pkl', 'wb') as f:
    pickle.dump(note_to_int, f)

with open('Subset_Dataset/duration_to_int.pkl', 'wb') as f:
    pickle.dump(duration_to_int, f)

with open('Subset_Dataset/tempo_to_int.pkl', 'wb') as f:
    pickle.dump(tempo_to_int, f)

with open('Subset_Dataset/time_signature_to_int.pkl', 'wb') as f:
    pickle.dump(time_signature_to_int, f)

with open('Subset_Dataset/key_signature_to_int.pkl', 'wb') as f:
    pickle.dump(key_signature_to_int, f)

print("Subset data (notes, durations, tempos, time signatures, and key signatures) saved successfully.")
