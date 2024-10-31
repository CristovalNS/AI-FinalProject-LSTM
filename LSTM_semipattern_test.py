import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from music21 import stream, note, chord, instrument


# Step 1: Define the Model Architecture (same as during training)
def build_model(sequence_length, n_unique_notes):
    model = Sequential()
    model.add(LSTM(512, input_shape=(sequence_length, 1), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_unique_notes, activation='softmax'))
    return model


# Step 2: Load the model weights
def load_trained_model(weights_path, sequence_length, n_unique_notes):
    model = build_model(sequence_length, n_unique_notes)
    model.load_weights(weights_path)
    return model


# Step 3: Generate Music with Semi-Repeating Pattern
def generate_music_with_semi_repeating_pattern(model, start_sequence, int_to_note, total_notes=100, block_size=10,
                                               temperature=0.5, min_note='C4', max_note='C5'):
    generated_notes = []
    current_sequence = np.reshape(start_sequence, (1, len(start_sequence), 1))  # Reshape seed sequence

    # Convert the range of note names to MIDI numbers
    min_midi = note.Note(min_note).pitch.midi
    max_midi = note.Note(max_note).pitch.midi

    # Generate the total number of notes in blocks of 10
    for i in range(0, total_notes, block_size):
        # Generate the first block (new random generation)
        block = []
        for _ in range(block_size):
            prediction = model.predict(current_sequence, verbose=0)[0]

            # Apply temperature to adjust randomness
            prediction = np.log(prediction) / temperature
            prediction = np.exp(prediction) / np.sum(np.exp(prediction))

            # Choose the next note within the specified range
            valid_note = False
            while not valid_note:
                index = np.random.choice(len(prediction), p=prediction)
                predicted_note = int_to_note[index]

                # Check if the predicted note (or chord) is within the specified range
                if ('.' in predicted_note) or predicted_note.isdigit():  # Chord case
                    chord_midi = [note.Note(int(n)).pitch.midi for n in predicted_note.split('.')]
                    if all(min_midi <= midi_val <= max_midi for midi_val in chord_midi):
                        valid_note = True
                else:  # Single note case
                    note_midi = note.Note(predicted_note).pitch.midi
                    if min_midi <= note_midi <= max_midi:
                        valid_note = True

            block.append(predicted_note)

            # Reshape the index to match the 3D sequence format (batch_size=1, sequence_length=1, input_dim=1)
            index_array = np.array([[[index]]])  # Shape (1, 1, 1)

            # Update the current sequence with the new prediction
            current_sequence = np.append(current_sequence, index_array, axis=1)
            current_sequence = current_sequence[:, 1:]  # Drop the first note to keep sequence length constant

        # Add the block to the generated notes
        generated_notes.extend(block)

        # Copy the block exactly (second block)
        generated_notes.extend(block)

        # Add a modified block (third block with variations)
        modified_block = []
        for note_str in block:
            if ('.' in note_str) or note_str.isdigit():  # Chord case
                chord_midi = [note.Note(int(n)).pitch.midi for n in note_str.split('.')]
                modified_chord = [(midi_val + np.random.choice([-1, 0, 1])) for midi_val in chord_midi]
                modified_block.append('.'.join(str(m) for m in modified_chord))
            else:  # Single note case
                note_midi = note.Note(note_str).pitch.midi
                modified_midi = note_midi + np.random.choice([-1, 0, 1])  # Slight change
                modified_block.append(note.Note(modified_midi).name)

        generated_notes.extend(modified_block)

    return generated_notes


# Step 4: Convert generated notes to a MIDI file
def create_midi(generated_notes, output_file='generated_music_semipattern.mid'):
    output_stream = stream.Stream()

    for pattern in generated_notes:
        # If it's a chord (e.g., '7.11.2')
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            output_stream.append(new_chord)
        else:  # It's a note
            new_note = note.Note(pattern)
            new_note.storedInstrument = instrument.Piano()
            output_stream.append(new_note)

    # Save the Stream to a MIDI file
    output_stream.write('midi', fp=output_file)


# Step 5: Load the required data (mapping, seed, etc.)
def load_data():
    # Load the int_to_note mapping (inverse of note_to_int)
    with open('Subset_Dataset/note_to_int.pkl', 'rb') as f:
        note_to_int = pickle.load(f)
    int_to_note = {number: note for note, number in note_to_int.items()}

    # Load your input sequences (or generate a random seed)
    with open('Subset_Dataset/input_sequences.pkl', 'rb') as f:
        X = pickle.load(f)

    return X, int_to_note


# Main execution
if __name__ == "__main__":
    # Step 6: Parameters
    weights_path = 'checkpoints_E5_OLD/weights-epoch-05-loss-4.6063-acc-0.0296.weights.h5'
    output_midi_file = 'generated_music_result/generated_music_semipattern.mid'
    total_notes = 100  # Total number of notes to generate
    block_size = 10  # Size of the block to repeat and alter
    temperature = 0.5  # Temperature for randomness

    # Step 7: Load the data and build the model
    X, int_to_note = load_data()
    sequence_length = X.shape[1]  # Length of input sequences
    n_unique_notes = len(int_to_note)  # Number of unique notes/chords

    # Step 8: Load the trained model with weights
    model = load_trained_model(weights_path, sequence_length, n_unique_notes)

    # Step 9: Generate music with a semi-repeating pattern
    start_sequence = X[np.random.randint(0, len(X))]  # Random seed from training data
    generated_notes = generate_music_with_semi_repeating_pattern(model, start_sequence, int_to_note, total_notes,
                                                                 block_size, temperature)

    # Step 10: Save the generated music to a MIDI file
    create_midi(generated_notes, output_file=output_midi_file)

    print(f"Music generation complete. Saved to {output_midi_file}")
