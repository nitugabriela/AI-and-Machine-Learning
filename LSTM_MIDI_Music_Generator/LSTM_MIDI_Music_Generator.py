import os
import glob
import numpy as np
import pretty_midi
import random
from tensorflow.keras import Sequential, layers
from scipy.ndimage import median_filter

# --- Parameters ---
sequence_len = 500
window_size = 64
epochs = 100
batch_size = 64
silence_thresh = 0.05
midi_folder = "hw-midi-songs"

# 1. Data Processing
def midi_to_sequence(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    instrument = None
    for inst in midi.instruments:
        if not inst.is_drum and len(inst.notes) > 10:
            instrument = inst
            break
    if instrument is None and len(midi.instruments) > 0:
        instrument = max(midi.instruments, key=lambda x: len(x.notes))
    if instrument is None:
        return None

    sequence = np.zeros(sequence_len)
    for note in instrument.notes:
        start = int(note.start * 20)
        end = int(note.end * 20)
        if start < sequence_len:
            sequence[start:min(end, sequence_len)] = note.pitch

    normalized = np.zeros_like(sequence)
    mask = sequence > 0
    normalized[mask] = (sequence[mask] - 21) / 87
    return normalized if normalized.sum() > 0 else None

def load_midi_files(folder):
    files = glob.glob(os.path.join(folder, "*.mid"))
    songs = []
    for path in files:
        seq = midi_to_sequence(path)
        if seq is not None:
            songs.append(seq)
    return np.array(songs)

def create_training_data(songs, window_size):
    X, y = [], []
    for song in songs:
        for i in range(len(song) - window_size - 1):
            X.append(song[i:i + window_size])
            y.append(song[i + window_size])
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y

# 2. Model Definition
def build_model(window_size):
    model = Sequential([
        layers.LSTM(256, input_shape=(window_size, 1), return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(256),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

# 3. Generative Inference (with Auto-Tune)
def generate_melody(model, seed_song, window_size, length=300, temperature=0.05):
    generated = list(seed_song[:window_size])
    scale_indices = [0, 2, 4, 5, 7, 9, 11] # C Major Scale

    for _ in range(length):
        window = np.array(generated[-window_size:]).reshape(1, window_size, 1)
        prediction = model.predict(window, verbose=0)[0, 0]

        noise = np.random.normal(0, temperature)
        raw_note = np.clip(prediction + noise, 0, 1)

        pitch = int(21 + raw_note * 87)
        note_in_octave = pitch % 12
        
        # Snap to scale
        if note_in_octave not in scale_indices:
            closest = min(scale_indices, key=lambda x: abs(x - note_in_octave))
            pitch = pitch - note_in_octave + closest

        corrected_note = (pitch - 21) / 87
        generated.append(corrected_note)

    smoothed = median_filter(generated, size=5)
    return smoothed

# 4. Custom Audio Engineering (Post-Processing) 
def save_as_pro_band(sequence, filename="mario_band.mid"):
    midi = pretty_midi.PrettyMIDI()
    lead_instrument = pretty_midi.Instrument(program=80) # Square wave
    bass_instrument = pretty_midi.Instrument(program=33) # Electric Bass

    current_pitch = None
    start_time = 0
    step_duration = 0.05

    for i, value in enumerate(sequence):
        pitch = int(21 + value * 87) if value > 0.05 else 0

        if pitch != current_pitch:
            if current_pitch is not None and current_pitch > 0:
                end_time = i * step_duration
                human_velocity = random.randint(85, 115)
                note = pretty_midi.Note(velocity=human_velocity, pitch=current_pitch, start=start_time, end=end_time)

                # Band Split Logic
                if current_pitch < 55:
                    bass_instrument.notes.append(note)
                else:
                    lead_instrument.notes.append(note)

            current_pitch = pitch
            start_time = i * step_duration

    midi.instruments.append(lead_instrument)
    midi.instruments.append(bass_instrument)
    midi.write(filename)
    print(f"Saved Pro Band Version: {filename}")

def save_smooth_legato(sequence, filename="mario_pure.mid"):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=80)
    
    current_pitch = None
    start_time = 0
    step_duration = 0.05
    min_gap_to_break_note = 2
    silence_counter = 0

    for i, value in enumerate(sequence):
        pitch = int(21 + value * 87) if value > 0.05 else 0
        is_silence = pitch == 0

        if pitch != current_pitch:
            if is_silence and silence_counter < min_gap_to_break_note:
                silence_counter += 1
                continue
            else:
                silence_counter = 0

            if current_pitch is not None and current_pitch > 0:
                end_time = i * step_duration
                note = pretty_midi.Note(velocity=100, pitch=current_pitch, start=start_time, end=end_time)
                instrument.notes.append(note)

            current_pitch = pitch
            start_time = i * step_duration

    midi.instruments.append(instrument)
    midi.write(filename)
    print(f"✨ Saved Pure Legato Version: {filename}")

# Execution Block 
if __name__ == "__main__":
    print("Loading data...")
    songs = load_midi_files(midi_folder)
    
    if len(songs) > 0:
        X_train, y_train = create_training_data(songs, window_size)
        
        print("Building and training model...")
        model = build_model(window_size)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        
        print("Generating new melody...")
        seed_index = random.randint(0, len(songs)-1)
        new_melody = generate_melody(model, songs[seed_index], window_size, length=400, temperature=0.08)
        
        save_as_pro_band(new_melody, "AI_Generated_Band.mid")
        save_smooth_legato(new_melody, "AI_Generated_Legato.mid")
    else:
        print(f"No MIDI files found in {midi_folder}. Please add some data to train the model.")
