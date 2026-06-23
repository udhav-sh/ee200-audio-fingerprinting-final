import core
import streamlit as st
import os
import io
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Add workspace path to system path to import core
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# App configuration
st.set_page_config(
    page_title="AcousticID | Audio Fingerprinting",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
# Automatically find the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the path relative to the app's location
DB_DIR = os.path.join(BASE_DIR, "audio-fingerprint-identifying-python", "db")
DB_PATH = os.path.join(DB_DIR, "fingerprints.db")

# Initialize database securely


@st.cache_resource
def get_database():
    return core.Database(DB_PATH)


db = get_database()

# Styling CSS for rich premium look
st.markdown("""
<style>
    /* Gradient App Title and Subheader */
    .title-text {
        background: linear-gradient(90deg, #FF4B4B, #8A2BE2, #00C9FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        font-family: 'Inter', sans-serif;
    }
    .subtitle-text {
        color: #A0AEC0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Styled Metric Cards */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        flex: 1;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFF;
    }
    .metric-val-winner {
        color: #00FFCC;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.3);
    }
    
    /* Custom Info banner */
    .info-banner {
        background: rgba(138, 43, 226, 0.15);
        border: 1px solid rgba(138, 43, 226, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# Custom header
st.markdown("<div class='title-text'>🎵 AcousticID</div>",
            unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Audio Fingerprinting & Identification System using Constellation Peak Hashing</div>", unsafe_allow_html=True)

# Helper function to load audio from uploaded file


def load_audio_from_uploaded(file_buffer, limit=None):
    audiofile = core.AudioSegment.from_file(file_buffer)
    if limit:
        audiofile = audiofile[:limit * 1000]
    data = np.frombuffer(audiofile._data, dtype=np.int16)
    channels = []
    for chn in range(audiofile.channels):
        channels.append(data[chn::audiofile.channels])
    return {
        "channels": channels,
        "Fs": audiofile.frame_rate
    }


# Create Tabs
tab_identify, tab_db, tab_batch = st.tabs([
    "🔍 Song Identification",
    "📁 Database Explorer & Manager",
    "📊 Batch Identification & Export"
])

# TAB 1: Song Identification
with tab_identify:
    st.header("Identify Audio")

    # Check if DB is empty
    all_songs = db.get_all_songs()
    if not all_songs:
        st.markdown("""
        <div class="info-banner">
            ⚠️ <b>Database is currently empty.</b><br>
            Please go to the <b>Database Explorer & Manager</b> tab and click <b>"Scan & Index DB Folder"</b> to index the songs before performing identification.
        </div>
        """, unsafe_allow_html=True)

    # Input selections
    col_input, col_settings = st.columns([2, 1])

    with col_input:
        input_type = st.radio("Choose Input Method", [
                              "Upload Audio File", "Select Sample Song from DB Folder (Simulated Query)"])

        uploaded_file = None
        sample_filename = None

        if input_type == "Upload Audio File":
            uploaded_file = st.file_uploader(
                "Upload query song (.mp3 or .wav)", type=["mp3", "wav"])
        else:
            # List actual .mp3 files in the DB directory
            if os.path.exists(DB_DIR):
                mp3_files = [f for f in os.listdir(
                    DB_DIR) if f.endswith(".mp3")]
                if mp3_files:
                    sample_filename = st.selectbox(
                        "Select song to query:", sorted(mp3_files))
                else:
                    st.warning(f"No .mp3 files found in DB folder: `{DB_DIR}`")
            else:
                st.error(f"DB Directory `{DB_DIR}` does not exist!")

    with col_settings:
        st.subheader("Query Constraints")
        limit_seconds = st.slider("Query Duration Limit (seconds)", min_value=3, max_value=60, value=15, step=1,
                                  help="Limits the processed sample size from the start of the audio.")

        offset_seconds = 0
        if input_type != "Upload Audio File" and sample_filename:
            # Allow user to crop snippet at a specific start offset to test robustness to shifts
            offset_seconds = st.slider("Query Start Offset (seconds)", min_value=0, max_value=120, value=0, step=1,
                                       help="Crop a segment starting at this offset to simulate middle-of-song query matching.")

    # Execute Identification
    if st.button("Identify Song", type="primary", use_container_width=True):
        audio_data = None
        query_display_name = ""

        if input_type == "Upload Audio File" and uploaded_file is not None:
            with st.spinner("Decoding uploaded audio..."):
                try:
                    # Load and crop
                    audio_data = load_audio_from_uploaded(
                        uploaded_file, limit=limit_seconds)
                    query_display_name = uploaded_file.name
                except Exception as e:
                    st.error(f"Error loading audio file: {e}")
        elif input_type != "Upload Audio File" and sample_filename:
            with st.spinner(f"Loading {sample_filename}..."):
                try:
                    filepath = os.path.join(DB_DIR, sample_filename)
                    # Use pydub to crop starting from offset_seconds
                    audiofile = core.AudioSegment.from_file(filepath)
                    cropped_audio = audiofile[offset_seconds *
                                              1000: (offset_seconds + limit_seconds)*1000]

                    data = np.frombuffer(cropped_audio._data, dtype=np.int16)
                    channels = []
                    for chn in range(cropped_audio.channels):
                        channels.append(data[chn::cropped_audio.channels])

                    audio_data = {
                        "channels": channels,
                        "Fs": cropped_audio.frame_rate
                    }
                    query_display_name = f"{sample_filename} (Crop: {offset_seconds}s - {offset_seconds + limit_seconds}s)"
                except Exception as e:
                    st.error(f"Error reading sample song: {e}")
        else:
            st.info("Please provide an audio file to identify.")

        if audio_data:
            st.divider()

            import time
            start_time = time.time()

            # Step 1: Extract hashes
            with st.spinner("Extracting spectrogram & generating peaks..."):
                # Use channel 0 for analysis
                channel_samples = audio_data['channels'][0]
                Fs = audio_data['Fs']

                # Spectrogram
                arr2D = core.get_spectrogram(channel_samples, Fs=Fs)
                # Peaks
                peaks = core.get_2D_peaks(arr2D)
                # Hashes
                hashes = list(core.generate_hashes(peaks))

            step1_time = time.time()
            extraction_ms = int((step1_time - start_time) * 1000)

            # Step 2: Database matching
            with st.spinner("Searching SQLite fingerprints database..."):
                matches = db.find_matches(hashes)
                alignment = core.align_matches(matches, db)

            step2_time = time.time()
            search_ms = int((step2_time - step1_time) * 1000)
            total_ms = extraction_ms + search_ms

            if alignment:
                # Display Results metrics
                st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-card'>
                        <div class='metric-title'>Most Likely Song Match</div>
                        <div class='metric-value metric-val-winner'>{alignment['song_name']}</div>
                    </div>
                    <div class='metric-card'>
                        <div class='metric-title'>Alignment Spike Confidence</div>
                        <div class='metric-value'>{alignment['confidence']} matches</div>
                    </div>
                    <div class='metric-card'>
                        <div class='metric-title'>Time Shift Offset</div>
                        <div class='metric-value'>{alignment['offset_secs']} seconds</div>
                    </div>
                </div>
                <div class='metric-container' style='margin-top: -1rem;'>
                    <div class='metric-card' style='padding: 1rem;'>
                        <div class='metric-title'>Spectrogram & Hash Time</div>
                        <div class='metric-value' style='font-size: 1.2rem;'>{extraction_ms} ms</div>
                    </div>
                    <div class='metric-card' style='padding: 1rem;'>
                        <div class='metric-title'>DB Search & Align Time</div>
                        <div class='metric-value' style='font-size: 1.2rem;'>{search_ms} ms</div>
                    </div>
                    <div class='metric-card' style='padding: 1rem;'>
                        <div class='metric-title'>Total Processing Time</div>
                        <div class='metric-value' style='font-size: 1.2rem; color: #00C9FF;'>{total_ms} ms</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Visual plots columns
                col_spec, col_hist = st.columns(2)

                with col_spec:
                    st.subheader("Spectrogram & Constellation Peak Map")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    # Show log-spectrogram
                    im = ax.imshow(arr2D, aspect='auto',
                                   origin='lower', cmap='viridis')
                    # Overlay peaks
                    freqs = [p[0] for p in peaks]
                    times = [p[1] for p in peaks]
                    ax.scatter(times, freqs, color='red', s=4,
                               label='Constellation Peaks', alpha=0.8)
                    ax.set_xlabel("Time Bins")
                    ax.set_ylabel("Frequency Bins")

                    # Dark mode styling for figure
                    fig.patch.set_facecolor('#0E1117')
                    ax.set_facecolor('#0E1117')
                    ax.tick_params(colors='white')
                    ax.yaxis.label.set_color('white')
                    ax.xaxis.label.set_color('white')
                    ax.title.set_color('white')
                    ax.legend(facecolor='#0E1117', labelcolor='white')
                    st.pyplot(fig)

                with col_hist:
                    st.subheader("Cross-Correlation Alignment Spike")
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    winner_offsets = alignment['winner_offsets']
                    ax2.hist(winner_offsets, bins=100, color='#8A2BE2',
                             edgecolor='black', alpha=0.8)
                    ax2.set_xlabel("Time Offset Difference (bins)")
                    ax2.set_ylabel("Match Count")

                    # Highlight peak
                    mode_offset = Counter(winner_offsets).most_common(1)[0][0]
                    ax2.axvline(x=mode_offset, color='#FF4B4B', linestyle='--', linewidth=2,
                                label=f'Spike at bin {mode_offset} ({alignment["offset_secs"]}s)')

                    fig2.patch.set_facecolor('#0E1117')
                    ax2.set_facecolor('#0E1117')
                    ax2.tick_params(colors='white')
                    ax2.yaxis.label.set_color('white')
                    ax2.xaxis.label.set_color('white')
                    ax2.title.set_color('white')
                    ax2.legend(facecolor='#0E1117', labelcolor='white')
                    st.pyplot(fig2)

                # Other Candidates Table
                st.subheader("Candidate Match Leaderboard")
                cand_df = pd.DataFrame(alignment['candidates'])
                if not cand_df.empty:
                    # Rename columns for presentation
                    cand_df = cand_df.rename(columns={
                        "name": "Song Name",
                        "max_spike": "Spike Confidence",
                        "total_matches": "Total Matches"
                    })
                    st.dataframe(
                        cand_df[["Song Name", "Spike Confidence", "Total Matches"]], use_container_width=True)
            else:
                st.warning(
                    "❌ No matching song found. Try increasing the duration or querying a different part of the audio.")

# TAB 2: Database Explorer & Manager
with tab_db:
    st.header("Database Manager")

    # Quick Statistics
    all_songs = db.get_all_songs()
    total_songs = len(all_songs)

    # Calculate total fingerprints in database
    with db.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM fingerprints")
        total_fingerprints = cur.fetchone()[0]

    avg_fingerprints = int(total_fingerprints /
                           total_songs) if total_songs > 0 else 0

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Total Indexed Songs", f"{total_songs}")
    with col_stat2:
        st.metric("Total Fingerprints", f"{total_fingerprints:,}")
    with col_stat3:
        st.metric("Avg Fingerprints per Song", f"{avg_fingerprints:,}")

    st.divider()

    # Action buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📁 Scan & Index DB Folder", type="primary", use_container_width=True):
            if not os.path.exists(DB_DIR):
                st.error(f"DB directory does not exist at `{DB_DIR}`!")
            else:
                mp3_files = [f for f in os.listdir(
                    DB_DIR) if f.endswith(".mp3")]
                if not mp3_files:
                    st.warning(f"No .mp3 files found in `{DB_DIR}` to index!")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    added_count = 0
                    skipped_count = 0

                    for idx, filename in enumerate(mp3_files):
                        status_text.text(
                            f"Processing ({idx+1}/{len(mp3_files)}): {filename}")
                        filepath = os.path.join(DB_DIR, filename)

                        try:
                            # 1. Check file hash to avoid duplicate parsing
                            filehash = core.get_file_hash(filepath)
                            existing_song = db.get_song_by_filehash(filehash)

                            if existing_song:
                                hash_count = db.get_song_hashes_count(
                                    existing_song[0])
                                if hash_count > 0:
                                    skipped_count += 1
                                    progress_bar.progress(
                                        (idx + 1) / len(mp3_files))
                                    continue

                            # 2. Parse and generate hashes
                            audio = core.load_audio(filepath)
                            song_id = db.add_song(filename, audio['file_hash'])

                            hashes = set()
                            for channeln, channel in enumerate(audio['channels']):
                                arr2D = core.get_spectrogram(
                                    channel, Fs=audio['Fs'])
                                peaks = core.get_2D_peaks(arr2D)
                                hashes |= set(core.generate_hashes(peaks))

                            # 3. Store in DB
                            fingerprints = [(song_id, h, offset)
                                            for h, offset in hashes]
                            db.store_fingerprints(fingerprints)
                            added_count += 1
                        except Exception as e:
                            st.error(f"Failed to process {filename}: {e}")

                        progress_bar.progress((idx + 1) / len(mp3_files))

                    status_text.text(
                        f"Indexing Complete! Stored {added_count} new songs, skipped {skipped_count} existing songs.")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()

    with col_btn2:
        if st.button("🗑️ Clear Database Index", type="secondary", use_container_width=True):
            db.reset_db()
            st.success("Database cleared successfully!")
            time.sleep(1)
            st.rerun()

    # Song List Table
    st.subheader("Indexed Songs in Database")
    if all_songs:
        song_df = pd.DataFrame(all_songs, columns=[
                               "ID", "Filename", "File Hash (SHA1)", "Fingerprint Count"])
        st.dataframe(song_df, use_container_width=True, hide_index=True)

        # Individual song deletion
        st.subheader("Delete Song from Index")
        delete_song_name = st.selectbox("Select song to delete:", [
                                        s[1] for s in all_songs])
        if st.button("Delete Selected Song"):
            selected_song = next(
                s for s in all_songs if s[1] == delete_song_name)
            db.delete_song(selected_song[0])
            st.success(f"Deleted '{delete_song_name}' from the index.")
            time.sleep(1)
            st.rerun()
    else:
        st.info(
            "Database is empty. Click 'Scan & Index DB Folder' above to search for songs.")
    st.divider()
    st.subheader("🌌 Database Constellation Gallery")
    st.write("Visual representation of stored hashes for each song.")

    if all_songs:
        # We put this behind a button because loading 50 graphs at once can slow down the browser
        if st.button("Load Constellation Visuals (May take a moment)"):
            with st.spinner("Generating constellation plots..."):
                # Create a grid layout with 4 columns
                cols = st.columns(4)

                for idx, song in enumerate(all_songs):
                    song_id = song[0]
                    song_name = song[1].replace('.mp3', '').replace('.wav', '')
                    hash_count = song[3]

                    with db.get_connection() as conn:
                        cur = conn.cursor()
                        # Limit to 1500 points per song so the app stays fast
                        cur.execute(
                            "SELECT hash, offset FROM fingerprints WHERE song_fk = ? LIMIT 1500", (song_id,))
                        fingerprints = cur.fetchall()

                    if fingerprints:
                        # Extract x (time offset)
                        x_vals = [fp[1] for fp in fingerprints]
                        # Convert first 4 characters of the hex hash to an integer for the Y-coordinate
                        y_vals = [int(fp[0][:4], 16) for fp in fingerprints]

                        fig, ax = plt.subplots(figsize=(3, 2))
                        fig.patch.set_facecolor('#0E1117')
                        ax.set_facecolor('#0E1117')

                        # Plot styling
                        ax.scatter(x_vals, y_vals, s=1,
                                   color='#00FFCC', alpha=0.5)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.axis('off')

                        # Place the graph in the correct column grid
                        col = cols[idx % 4]
                        with col:
                            st.pyplot(fig)
                            st.markdown(
                                f"<p style='text-align: left; font-size: 0.8rem; color: #E2E8F0; font-weight: 600; margin-bottom: 0;'>{song_name}</p>", unsafe_allow_html=True)
                            st.markdown(
                                f"<p style='text-align: left; font-size: 0.75rem; color: #A0AEC0;'>{hash_count:,} hashes</p>", unsafe_allow_html=True)

                        # Clean up memory so the app doesn't crash
                        plt.close(fig)

# TAB 3: Batch Identification & Export
with tab_batch:
    st.header("Batch Song Identification")
    st.write("Upload multiple songs, run them through the identification pipeline, and export the results to CSV.")

    uploaded_files = st.file_uploader("Upload query files (.mp3 or .wav)", type=[
                                      "mp3", "wav"], accept_multiple_files=True)

    if uploaded_files:
        st.write(
            f"Loaded {len(uploaded_files)} files. Press run below to process them.")

        if st.button("Run Batch Identification", type="primary"):
            results = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, file_obj in enumerate(uploaded_files):
                status_text.text(
                    f"Identifying ({idx+1}/{len(uploaded_files)}): {file_obj.name}")

                try:
                    # Load audio
                    audio_data = load_audio_from_uploaded(
                        file_obj, limit=limit_seconds)
                    channel_samples = audio_data['channels'][0]
                    Fs = audio_data['Fs']

                    # Generate fingerprints
                    arr2D = core.get_spectrogram(channel_samples, Fs=Fs)
                    peaks = core.get_2D_peaks(arr2D)
                    hashes = list(core.generate_hashes(peaks))

                    # Search
                    matches = db.find_matches(hashes)
                    alignment = core.align_matches(matches, db)

                    if alignment:
                        # Clean the extension off the predicted song name
                        clean_song_name = alignment['song_name'].replace(
                            '.mp3', '').replace('.wav', '')

                        results.append({
                            "filename": file_obj.name,
                            "prediction": clean_song_name
                        })
                    else:
                        results.append({
                            "filename": file_obj.name,
                            "prediction": "none"
                        })
                except Exception as e:
                    results.append({
                        "filename": file_obj.name,
                        "prediction": "none"
                    })

                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.text("Batch identification complete!")

            # Show DataFrame
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)

            # Download CSV
            csv_buffer = io.StringIO()
            res_df.to_csv(csv_buffer, index=False)
            csv_str = csv_buffer.getvalue()

            st.download_button(
                label="📥 Export Results to CSV",
                data=csv_str,
                file_name="fingerprinting_batch_results.csv",
                mime="text/csv"
            )
