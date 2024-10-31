import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from music21 import stream, note, chord, instrument
import random

# Step 1: Define the Model Architecture (same as during training)
def build_model(sequence_length, n_unique_notes):
    model = Sequential()
    model.add(LSTM(512, input_shape=(sequence_length, 5), return_sequences=True))  # Adjusted to 5 input features
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_unique_notes, activation='softmax'))
    return model

# Step 2: Load the Trained Model Weights
sequence_length = 100  # The sequence length you used during training
n_unique_notes = 576  # Adjust this based on your unique notes

# Build the model
model = build_model(sequence_length, n_unique_notes)

# Load the weights from the latest checkpoint
latest_checkpoint = '/Users/cristovalneosasono/AI_FinalProject_V2/checkpoints/weights-epoch-05-loss-1.7863-acc-0.5304.weights.h5'
model.load_weights(latest_checkpoint)

# Step 4: Load the Mapping Data (int to note)
with open('Subset_Dataset/note_to_int.pkl', 'rb') as f:
    note_to_int = pickle.load(f)

int_to_note = {number: note for note, number in note_to_int.items()}

# Identify rest indices
rest_indices = [index for index, note in int_to_note.items() if note.lower() == 'rest' or note == 'R']

# Step 5: Load the Data and Concatenate Start Sequence
with open('Subset_Dataset/input_sequences_notes.pkl', 'rb') as f:
    X_notes = pickle.load(f)
with open('Subset_Dataset/input_sequences_durations.pkl', 'rb') as f:
    X_durations = pickle.load(f)
with open('Subset_Dataset/input_sequences_tempos.pkl', 'rb') as f:
    X_tempos = pickle.load(f)
with open('Subset_Dataset/input_sequences_time_signatures.pkl', 'rb') as f:
    X_time_signatures = pickle.load(f)
with open('Subset_Dataset/input_sequences_key_signatures.pkl', 'rb') as f:
    X_key_signatures = pickle.load(f)

# Concatenate all features to create the full start sequence
X = np.concatenate([X_notes, X_durations, X_tempos, X_time_signatures, X_key_signatures], axis=-1)

# Choose a random start sequence from the training data
start_sequence = X[np.random.randint(0, len(X))]

def transpose_note_down(note_name, semitones=1):
    """Transpose a note down by a specified number of semitones."""
    try:
        midi_note = note.Note(note_name).pitch.midi
        new_midi_note = midi_note - semitones
        return note.Note(new_midi_note).name
    except:
        return note_name  # Return the same note if transposition fails


def generate_music_with_patterns(model, start_sequence, int_to_note, rest_indices, n_generate=100, temperature=1.0, max_interval=5):
    generated_notes = []
    current_sequence = np.reshape(start_sequence, (1, len(start_sequence), 5))  # Reshape seed sequence to match model input

    # Step 1: Generate the first 10 notes (Pattern 1)
    for i in range(10):
        prediction = model.predict(current_sequence, verbose=0)[0]
        prediction[rest_indices] = 0
        prediction = prediction ** (1 / temperature)
        prediction = prediction / np.sum(prediction)
        index = np.random.choice(len(prediction), p=prediction)
        predicted_pattern = int_to_note[index]

        # Ensure that only single notes are selected
        if ('.' in predicted_pattern) or predicted_pattern.isdigit():
            continue  # Skip chord-like patterns

        # For single notes, limit the pitch to C3-C4
        note_midi = note.Note(predicted_pattern).pitch.midi
        if note_midi < 65 or note_midi > 80:
            predicted_pattern = note.Note(random.choice(range(65, 80))).name

        generated_notes.append(predicted_pattern)

        next_features = [index] * 5
        next_features_array = np.array([next_features]).reshape((1, 1, 5))
        current_sequence = np.append(current_sequence, next_features_array, axis=1)
        current_sequence = current_sequence[:, 1:]  # Keep sequence length constant

    # Step 2: Repeat the first 10 notes for notes 11-20
    generated_notes.extend(generated_notes[:10])

    # Step 3: Transpose Pattern 1 down by 1 semitone for notes 21-30, repeat for 31-40
    transposed_notes_1 = [transpose_note_down(note_name, semitones=1) for note_name in generated_notes[:10]]
    generated_notes.extend(transposed_notes_1)
    generated_notes.extend(transposed_notes_1)

    # Step 4: Transpose Pattern 2 down by 1 semitone for notes 41-50, repeat for 51-60
    transposed_notes_2 = [transpose_note_down(note_name, semitones=1) for note_name in transposed_notes_1]
    generated_notes.extend(transposed_notes_2)
    generated_notes.extend(transposed_notes_2)

    # Step 5: Repeat the pattern for the remaining notes (up to 100)
    while len(generated_notes) < n_generate:
        transposed_notes = [transpose_note_down(note_name, semitones=1) for note_name in transposed_notes_2]
        generated_notes.extend(transposed_notes)
        transposed_notes_2 = transposed_notes

    return generated_notes[:n_generate]  # Ensure only 100 notes are generated



def create_midi(generated_notes, output_file='generated_music_melody.mid'):
    output_stream = stream.Stream()
    # Example of flexible duration probabilities based on what you want
    durations = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]  # Adding longer note durations
    probabilities = [0.1, 0.2, 0.4, 0.1, 0.1, 0.05, 0.05]  # Adjusting to allow a variety of durations

    # durations = [0.5, 1.0, 1.5, 2.0]  # Adding longer note durations
    # probabilities = [0.1, 0.5, 0.2, 0.2]  # Adjusting to allow a variety of durations

    for note_name in generated_notes:
        # Randomly assign duration based on the probabilities
        duration = np.random.choice(durations, p=probabilities)

        # Create a new note object and assign its duration
        new_note = note.Note(note_name)
        new_note.duration.quarterLength = duration  # Set the note's duration
        new_note.storedInstrument = instrument.Piano()  # Ensure it's for piano

        output_stream.append(new_note)  # Add the note to the stream

    # Save the Stream to a MIDI file
    output_stream.write('midi', fp=output_file)

# Step 1: Generate 100 notes using the new pattern generation logic
generated_notes = generate_music_with_patterns(
    model,
    start_sequence,
    int_to_note,
    rest_indices,
    n_generate=50,  # Ensure we generate exactly 100 melody notes
    temperature=0.1
)

# Step 2: Create the MIDI file from the generated notes
create_midi(generated_notes, output_file='generated_music_result/generated_music_melody_with_patterns.mid')

print("Music generation complete. Saved to 'generated_music_melody_with_patterns.mid'.")

