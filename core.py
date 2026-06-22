import os
import hashlib
import sqlite3
import numpy as np
import matplotlib.mlab as mlab
from scipy.ndimage import maximum_filter, generate_binary_structure, iterate_structure, binary_erosion
from pydub import AudioSegment

# import static_ffmpeg

# Initialize ffmpeg binaries on startup
# static_ffmpeg.add_paths()

# Algorithm Parameters (from the repository)
DEFAULT_FS = 44100
DEFAULT_WINDOW_SIZE = 4096
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_FAN_VALUE = 15
DEFAULT_AMP_MIN = 10
PEAK_NEIGHBORHOOD_SIZE = 20
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200
FINGERPRINT_REDUCTION = 20


def load_audio(filepath, limit=None):
    """
    Loads an audio file using pydub, decodes it into raw PCM, and returns channel data.
    """
    songname, extension = os.path.splitext(os.path.basename(filepath))
    audiofile = AudioSegment.from_file(filepath)

    if limit:
        audiofile = audiofile[:limit * 1000]

    # Get raw PCM data as numpy array
    data = np.frombuffer(audiofile._data, dtype=np.int16)

    channels = []
    for chn in range(audiofile.channels):
        channels.append(data[chn::audiofile.channels])

    return {
        "songname": songname,
        "extension": extension,
        "channels": channels,
        "Fs": audiofile.frame_rate,
        "file_hash": get_file_hash(filepath)
    }


def get_file_hash(filepath, blocksize=2**20):
    """
    Computes SHA1 hash of a file for identification.
    """
    s = hashlib.sha1()
    with open(filepath, "rb") as f:
        while True:
            buf = f.read(blocksize)
            if not buf:
                break
            s.update(buf)
    return s.hexdigest().upper()


def get_2D_peaks(arr2D, amp_min=DEFAULT_AMP_MIN):
    """
    Extracts peak frequencies from 2D spectrogram using morphological filter.
    """
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # find local maxima using our filter shape
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
    background = (arr2D == 0)
    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1)

    # Boolean mask of arr2D with True at peaks
    detected_peaks = local_max ^ eroded_background

    # extract peaks
    amps = arr2D[detected_peaks]
    j, i = np.where(detected_peaks)  # j is frequency (rows), i is time (cols)

    # filter peaks
    amps = amps.flatten()
    peaks = list(zip(i, j, amps))
    peaks_filtered = [x for x in peaks if x[2] > amp_min]  # time, freq, amp

    # get indices for frequency and time
    frequency_idx = [x[1] for x in peaks_filtered]
    time_idx = [x[0] for x in peaks_filtered]

    return list(zip(frequency_idx, time_idx))


def generate_hashes(peaks, fan_value=DEFAULT_FAN_VALUE):
    """
    Generates combinatorial hashes pairing nearby peaks in time.
    """
    # Sort peaks by time index (which is at index 1: (frequency, time))
    peaks.sort(key=lambda x: x[1])

    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if (i + j) < len(peaks):
                freq1 = peaks[i][0]
                freq2 = peaks[i + j][0]
                t1 = peaks[i][1]
                t2 = peaks[i + j][1]
                t_delta = t2 - t1

                if MIN_HASH_TIME_DELTA <= t_delta <= MAX_HASH_TIME_DELTA:
                    h_str = f"{freq1}|{freq2}|{t_delta}"
                    h = hashlib.sha1(h_str.encode('utf-8'))
                    yield (h.hexdigest()[0:FINGERPRINT_REDUCTION].upper(), int(t1))


def get_spectrogram(channel_samples, Fs=DEFAULT_FS, wsize=DEFAULT_WINDOW_SIZE, wratio=DEFAULT_OVERLAP_RATIO):
    """
    Computes the log spectrogram of the audio channel samples.
    """
    arr2D = mlab.specgram(
        channel_samples,
        NFFT=wsize,
        Fs=Fs,
        window=mlab.window_hanning,
        noverlap=int(wsize * wratio)
    )[0]

    # Apply log transform
    arr2D = 10 * np.log10(arr2D)
    arr2D[arr2D == -np.inf] = 0
    return arr2D


class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self.check_and_unzip()
        self.init_db()

    def check_and_unzip(self):
        if not os.path.exists(self.db_path):
            zip_path = self.db_path + ".zip"
            if os.path.exists(zip_path):
                import zipfile
                print(f"Unzipping pre-indexed database: {zip_path}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(self.db_path))
                print("Database extraction complete.")

    def get_connection(self):
        # 1. Get the folder path from the full database path
        db_dir = os.path.dirname(self.db_path)

        # 2. If there is a folder path, create it (and ignore if it already exists)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        # 3. Now it is safe to connect!
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        return conn

    def init_db(self):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS songs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    filehash TEXT UNIQUE
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fingerprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    song_fk INTEGER,
                    hash TEXT,
                    offset INTEGER,
                    FOREIGN KEY(song_fk) REFERENCES songs(id)
                );
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_fingerprints_hash ON fingerprints(hash);")
            conn.commit()

    def get_song_by_filehash(self, filehash):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, name, filehash FROM songs WHERE filehash = ?", (filehash,))
            return cur.fetchone()

    def get_song_by_id(self, song_id):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, name, filehash FROM songs WHERE id = ?", (song_id,))
            return cur.fetchone()

    def get_all_songs(self):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, name, filehash,
                (SELECT count(*) FROM fingerprints WHERE song_fk = songs.id) as hash_count
                FROM songs ORDER BY name ASC;
            """)
            return cur.fetchall()

    def add_song(self, name, filehash):
        with self.get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "INSERT INTO songs (name, filehash) VALUES (?, ?)", (name, filehash))
                conn.commit()
                return cur.lastrowid
            except sqlite3.IntegrityError:
                # Song already exists, retrieve its id
                cur.execute(
                    "SELECT id FROM songs WHERE filehash = ?", (filehash,))
                row = cur.fetchone()
                return row[0] if row else None

    def store_fingerprints(self, values):
        """
        Inserts multiple fingerprints in batches.
        """
        with self.get_connection() as conn:
            cur = conn.cursor()
            # Batch size of 1000
            for i in range(0, len(values), 1000):
                batch = values[i:i+1000]
                cur.executemany(
                    "INSERT OR IGNORE INTO fingerprints (song_fk, hash, offset) VALUES (?, ?, ?)", batch)
            conn.commit()

    def get_song_hashes_count(self, song_id):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT count(*) FROM fingerprints WHERE song_fk = ?", (song_id,))
            row = cur.fetchone()
            return row[0] if row else 0

    def delete_song(self, song_id):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM fingerprints WHERE song_fk = ?", (song_id,))
            cur.execute("DELETE FROM songs WHERE id = ?", (song_id,))
            conn.commit()

    def reset_db(self):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DROP TABLE IF EXISTS fingerprints;")
            cur.execute("DROP TABLE IF EXISTS songs;")
            conn.commit()
        self.init_db()

    def find_matches(self, hashes):
        """
        Searches database for matching hashes and computes offsets.
        Returns a list of tuples: (song_fk, db_offset - query_offset)
        """
        # Create a map of hash -> query_offset
        mapper = {}
        for h, offset in hashes:
            mapper[h.upper()] = offset

        values = list(mapper.keys())
        matches = []

        if not values:
            return matches

        with self.get_connection() as conn:
            cur = conn.cursor()
            # Query in chunks of 999 (SQLite limit for parameters)
            chunk_size = 900
            for i in range(0, len(values), chunk_size):
                chunk = values[i:i+chunk_size]
                placeholders = ",".join("?" * len(chunk))
                query = f"""
                    SELECT upper(hash), song_fk, offset
                    FROM fingerprints
                    WHERE hash IN ({placeholders})
                """
                cur.execute(query, chunk)
                rows = cur.fetchall()
                for h_db, song_fk, db_offset in rows:
                    h_upper = h_db.upper()
                    if h_upper in mapper:
                        matches.append(
                            (song_fk, int(db_offset) - int(mapper[h_upper])))

        return matches


def align_matches(matches, db):
    """
    Analyzes list of matches to find the most probable song using alignment.
    Returns details of the matching song, all scores, and raw offset counts for the winner.
    """
    if not matches:
        return None

    # count structure: {song_fk: {offset_diff: count}}
    counts = {}
    for song_id, diff in matches:
        if song_id not in counts:
            counts[song_id] = {}
        counts[song_id][diff] = counts[song_id].get(diff, 0) + 1

    # Find the maximum count across all songs and offset diffs
    winner_song_id = -1
    winner_diff = 0
    winner_count = 0

    for song_id, diff_map in counts.items():
        for diff, cnt in diff_map.items():
            if cnt > winner_count:
                winner_count = cnt
                winner_diff = diff
                winner_song_id = song_id

    # Compute a confidence score for other candidate matches as well
    song_scores = []
    for song_id, diff_map in counts.items():
        song_info = db.get_song_by_id(song_id)
        if not song_info:
            continue
        max_spike = max(diff_map.values())
        total_matches = sum(diff_map.values())
        song_scores.append({
            "id": song_id,
            "name": song_info[1],
            "max_spike": max_spike,
            "total_matches": total_matches
        })

    # Sort candidates by max_spike (descending)
    song_scores.sort(key=lambda x: x["max_spike"], reverse=True)

    winner_info = db.get_song_by_id(winner_song_id)
    if not winner_info:
        return None

    # Calculate time offset in seconds
    # nseconds = (largest_offset * overlap_ratio * window_size) / sampling_rate
    nseconds = round(
        float(winner_diff) / DEFAULT_FS *
        DEFAULT_WINDOW_SIZE * DEFAULT_OVERLAP_RATIO, 5
    )

    # Get raw offsets list for the winner song to draw the histogram
    winner_offsets = [diff for song_id,
                      diff in matches if song_id == winner_song_id]

    return {
        "song_id": winner_song_id,
        "song_name": winner_info[1],
        "confidence": winner_count,
        "offset": winner_diff,
        "offset_secs": nseconds,
        "candidates": song_scores,
        "winner_offsets": winner_offsets
    }
