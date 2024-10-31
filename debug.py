from music21 import converter

midi = converter.parse('/Users/cristovalneosasono/AI_FinalProject_V2/Classical_MIDI/Baroque/Johann Caspar Ferdinand Fischer/Erato Allemande.mid')

# Print all elements in the MIDI file
for element in midi.flat:
    print(element)
