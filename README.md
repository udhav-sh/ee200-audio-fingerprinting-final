# Audio Fingerprinting System

## Experiments
The requiered documentation and code is present in the `report_experiments.ipynb` python notebook. 

## Live Demo
You can interact with the live application here:
[https://ee200-audio-fingerprinting-final-7svvdcxdcszmoqbdsncqr5.streamlit.app/]

## Important Note for Reviewers
The application utilizes a pre-indexed database. Upon the **first boot** of the application, the system may take a moment to initialize and extract the fingerprint database. Please allow the loading spinner to complete before interacting with the tool.

## Setup Instructions
If you wish to run this project locally:
1. Clone the repository: `git clone https://github.com/udhav-sh/ee200-audio-fingerprinting.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Install FFmpeg (required for audio processing).
4. Run the application: `streamlit run app.py`