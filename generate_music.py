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

# Step 3: Generate Music with Smoother Transitions
def generate_music(model, start_sequence, int_to_note, rest_indices, n_generate=100, temperature=1.0, max_interval=5):
    generated_notes = []
    current_sequence = np.reshape(start_sequence, (1, len(start_sequence), 5))  # Reshape seed sequence to match model input

    previous_note_midi = None  # Keep track of the previous note's MIDI value

    for _ in range(n_generate):
        # Predict the next note
        prediction = model.predict(current_sequence, verbose=0)[0]

        # Zero out the probabilities for rests
        prediction[rest_indices] = 0

        # Avoid division by zero
        total_probability = np.sum(prediction)
        if total_probability == 0:
            print("No valid notes to sample from. Exiting.")
            break

        # Apply temperature to adjust randomness
        prediction = prediction ** (1 / temperature)
        prediction = prediction / np.sum(prediction)  # Renormalize

        # Choose the next note based on smoothed transitions
        if previous_note_midi is not None:
            # Adjust probabilities to favor notes closer to the previous note
            for idx, note_str in int_to_note.items():
                try:
                    current_note_midi = note.Note(note_str).pitch.midi
                    interval = abs(current_note_midi - previous_note_midi)
                    if interval > max_interval:  # Penalize large intervals
                        prediction[idx] *= 0.1  # Reduce probability for larger intervals
                except:
                    pass  # Ignore non-note patterns (e.g., rests)

            # Renormalize the prediction after adjustments
            prediction = prediction / np.sum(prediction)

        # Choose the next note
        index = np.random.choice(len(prediction), p=prediction)
        predicted_pattern = int_to_note[index]

        # Check if the predicted pattern is a note or chord
        if ('.' in predicted_pattern) or predicted_pattern.isdigit():  # Chord
            notes_in_chord = [int(n) for n in predicted_pattern.split('.')]
            # Limit the pitch of the chord's notes to C3-C4 (MIDI numbers 48 to 60)
            notes_in_chord = [n if 48 <= n <= 60 else random.choice(range(48, 61)) for n in notes_in_chord]
            predicted_pattern = '.'.join(map(str, notes_in_chord))
            # Set the previous note to the first note of the chord for interval checking
            previous_note_midi = notes_in_chord[0]
        else:
            # For single notes, limit the pitch to C3-C4
            note_midi = note.Note(predicted_pattern).pitch.midi
            if note_midi < 60 or note_midi > 70:
                predicted_pattern = note.Note(random.choice(range(60, 70))).name
            # Update previous note MIDI for interval checking
            previous_note_midi = note.Note(predicted_pattern).pitch.midi

        generated_notes.append(predicted_pattern)

        # Create a new array for the next timestep with the predicted features
        next_features = [index] * 5  # Adjust this line if necessary
        next_features_array = np.array([next_features]).reshape((1, 1, 5))  # Shape (1, 1, 5)

        # Update the current sequence by appending the new timestep
        current_sequence = np.append(current_sequence, next_features_array, axis=1)
        current_sequence = current_sequence[:, 1:]  # Keep sequence length constant

    return generated_notes

# Generate music with smoother transitions
generated_notes_smooth = generate_music(
    model,
    start_sequence,
    int_to_note,
    rest_indices,
    n_generate=100,
    temperature=0.8,
    max_interval=5  # Maximum interval between consecutive notes
)

# Generate 100 notes without rests
generated_notes = generate_music(model, start_sequence, int_to_note, rest_indices, n_generate=100, temperature=0.8)

# Step 6: Convert Generated Notes to a MIDI File

def create_midi(generated_notes, output_file='generated_music.mid'):
    output_stream = stream.Stream()
    # Example duration probabilities based on your dataset
    durations = [0.25, 0.5, 1.0, 1.5, 2.0]
    probabilities = [0.1, 0.3, 0.4, 0.1, 0.1]

    for note_name in generated_notes:
        duration = np.random.choice(durations, p=probabilities)
        # If it's a chord (e.g., '7.11.2')
        if ('.' in note_name) or note_name.isdigit():
            notes_in_chord = note_name.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.duration.quarterLength = duration
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.duration.quarterLength = duration
            output_stream.append(new_chord)
        else:  # It's a note
            new_note = note.Note(note_name)
            new_note.duration.quarterLength = duration
            new_note.storedInstrument = instrument.Piano()
            output_stream.append(new_note)

    # Save the Stream to a MIDI file
    output_stream.write('midi', fp=output_file)

# Create the MIDI file from the generated notes
create_midi(generated_notes, output_file='generated_music_result/generated_music.mid')

print("Music generation complete. Saved to 'generated_music.mid'.")
