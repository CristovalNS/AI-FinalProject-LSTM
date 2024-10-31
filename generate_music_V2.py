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
latest_checkpoint = '/Users/cristovalneosasono/AI_FinalProject_V2/checkpoints_E20/checkpointsweights-epoch-05-loss-2.3782-acc-0.4184.weights.h5'
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

def generate_music(model, start_sequence, int_to_note, rest_indices, n_generate=100, temperature=1.0, max_interval=5):
    generated_notes = []
    current_sequence = np.reshape(start_sequence, (1, len(start_sequence), 5))  # Reshape seed sequence to match model input

    previous_note_midi = None  # Keep track of the previous note's MIDI value

    while len(generated_notes) < n_generate:  # Keep generating until we have 100 valid melody notes
        # Predict the next note
        prediction = model.predict(current_sequence, verbose=0)[0]

        # Zero out the probabilities for rests
        prediction[rest_indices] = 0

        # Avoid division by zero
        total_probability = np.sum(prediction)
        if total_probability == 0:
            # Fallback: pick a random valid note if the prediction fails
            print(f"No valid notes to sample at {len(generated_notes)}/100. Picking a random valid note.")
            index = np.random.choice([i for i in range(len(int_to_note)) if i not in rest_indices])
        else:
            # Apply temperature to adjust randomness
            prediction = prediction ** (1 / temperature)
            prediction = prediction / np.sum(prediction)  # Renormalize

            # Choose the next note based on smoothed transitions
            if previous_note_midi is not None:
                for idx, note_str in int_to_note.items():
                    try:
                        current_note_midi = note.Note(note_str).pitch.midi
                        interval = abs(current_note_midi - previous_note_midi)
                        if interval > max_interval:  # Penalize large intervals
                            prediction[idx] *= 0.1
                    except:
                        pass

                prediction = prediction / np.sum(prediction)  # Renormalize after adjustments

            index = np.random.choice(len(prediction), p=prediction)

        predicted_pattern = int_to_note[index]

        # Ensure that only single notes are selected
        if ('.' in predicted_pattern) or predicted_pattern.isdigit():
            continue  # Skip chord-like patterns

        # For single notes, limit the pitch to C3-C4
        note_midi = note.Note(predicted_pattern).pitch.midi
        if note_midi < 60 or note_midi > 70:
            predicted_pattern = note.Note(random.choice(range(60, 70))).name

        previous_note_midi = note.Note(predicted_pattern).pitch.midi

        generated_notes.append(predicted_pattern)

        print(f"Generated {len(generated_notes)}/{n_generate} notes")

        next_features = [index] * 5
        next_features_array = np.array([next_features]).reshape((1, 1, 5))
        current_sequence = np.append(current_sequence, next_features_array, axis=1)
        current_sequence = current_sequence[:, 1:]  # Keep sequence length constant

    return generated_notes


def create_midi(generated_notes, output_file='generated_music_melody.mid'):
    output_stream = stream.Stream()
    # Example of flexible duration probabilities based on what you want
    durations = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]  # Adding longer note durations
    probabilities = [0.1, 0.2, 0.4, 0.1, 0.1, 0.05, 0.05]  # Adjusting to allow a variety of durations

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

# Step 1: Generate 100 notes without chords (melodies only)
generated_notes = generate_music(
    model,
    start_sequence,
    int_to_note,
    rest_indices,
    n_generate=50,  # Ensure we generate exactly 100 melody notes
    temperature=0.8
)

# Step 2: Create the MIDI file from the generated notes
create_midi(generated_notes, output_file='generated_music_result/generated_music_melody.mid')

print("Music generation complete. Saved to 'generated_music_melody.mid'.")

