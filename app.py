import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from deepface import DeepFace
import cv2
import numpy as np
import pandas as pd
from collections import deque, Counter
from datetime import datetime

st.set_page_config(page_title="Live Face Analyzer", layout="centered")
st.title("🧠 Real-Time Emotion, Age, Gender & Race Detector")
st.caption("Uses DeepFace + Webcam to analyze your face in real time.")

# History & logs
emotion_history = deque(maxlen=50)
age_history = deque(maxlen=50)
gender_history = deque(maxlen=50)
race_history = deque(maxlen=50)
session_log = []

# Session timer variables (initialized once)
start_time = datetime.now()
stop_time = None

class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.result = {}
        self.frame_count = 0
        self.last_result = None
        self.paused = False

    def transform(self, frame):
        if self.paused:
            img = frame.to_ndarray(format="bgr24")
            if self.last_result:
                self._draw_labels(img, self.last_result)
            return img

        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Resize smaller for speed
        small_img = cv2.resize(img, (320, int(img.shape[0] * 320 / img.shape[1])))

        if self.frame_count % 10 == 0:
            try:
                analysis = DeepFace.analyze(
                    small_img,
                    actions=["emotion", "age", "gender", "race"],
                    enforce_detection=False
                )[0]

                age = analysis["age"]
                emotion_scores = analysis["emotion"]
                dominant_emotion = analysis["dominant_emotion"]
                gender_scores = analysis["gender"]
                gender = max(gender_scores, key=gender_scores.get)
                race = analysis["dominant_race"]

                emotion_history.append(dominant_emotion)
                age_history.append(age)
                gender_history.append(gender)
                race_history.append(race)

                timestamp = datetime.now().strftime("%H:%M:%S")
                session_log.append({
                    "Time": timestamp,
                    "Age": age,
                    "Gender": gender,
                    "Emotion": dominant_emotion,
                    "Race": race
                })

                self.last_result = {
                    "age": age,
                    "gender": gender,
                    "race": race,
                    "emotion": dominant_emotion,
                    "emotion_scores": emotion_scores,
                    "gender_scores": gender_scores,
                    "race_scores": analysis.get("race", {}),
                }

            except Exception as e:
                print("Error:", e)

        if self.last_result:
            self._draw_labels(img, self.last_result)
            self.result = self.last_result

        return img

    def _draw_labels(self, img, res):
        # Approximate face box in center
        h, w, _ = img.shape
        box_w, box_h = int(w * 0.4), int(h * 0.5)
        x1, y1 = int(w * 0.3), int(h * 0.25)
        x2, y2 = x1 + box_w, y1 + box_h
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 100), 2)

        emotion = res["emotion"]
        emo_conf = res["emotion_scores"][emotion]
        age = res["age"]
        gender = res["gender"]
        race = res["race"]

        gender_conf = res["gender_scores"].get(gender, 0)
        race_conf = res["race_scores"].get(race, 0)

        label = (f"Emotion: {emotion.upper()} ({emo_conf:.1f}%) | "
                 f"Age: {age} | "
                 f"Gender: {gender} ({gender_conf:.1f}%) | "
                 f"Race: {race} ({race_conf:.1f}%)")

        cv2.putText(img, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 100), 2, cv2.LINE_AA)


# Start webcam with webrtc streamer
ctx = webrtc_streamer(
    key="emotion-analyzer",
    video_transformer_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

detector = ctx.video_transformer

# Control buttons: Pause / Resume
col1, col2 = st.columns(2)
with col1:
    if st.button("⏸ Pause Analysis"):
        if detector:
            detector.paused = True
            if stop_time is None:
                stop_time = datetime.now()

with col2:
    if st.button("▶ Resume Analysis"):
        if detector:
            detector.paused = False
            if start_time is None:
                start_time = datetime.now()
            if stop_time is not None:
                stop_time = None

# Show current prediction info
if detector and detector.result:
    res = detector.result
    st.subheader("🔍 Current Prediction")
    st.markdown(f"**Emotion**: {res['emotion']} ({res['emotion_scores'][res['emotion']]:.2f}%)")
    st.markdown(f"**Age**: {res['age']}")
    st.markdown(f"**Gender**: {res['gender']} ({res['gender_scores'][res['gender']]:.2f}%)")
    st.markdown(f"**Race**: {res['race']} ({res['race_scores'].get(res['race'], 0):.2f}%)")

    st.subheader("📊 Emotion Confidence")
    emotion_df = pd.DataFrame(res['emotion_scores'].items(), columns=["Emotion", "Confidence"])
    st.bar_chart(emotion_df.set_index("Emotion"))

# Emotion history dashboard
if len(emotion_history) > 0:
    st.subheader("📈 Recent Emotion Distribution")
    emotion_counts = pd.Series(list(emotion_history)).value_counts()
    st.bar_chart(emotion_counts)

    st.subheader("🎂 Gender Distribution")
    gender_counts = pd.Series(list(gender_history)).value_counts()
    st.bar_chart(gender_counts)

    st.subheader("🌍 Race Distribution")
    race_counts = pd.Series(list(race_history)).value_counts()
    st.bar_chart(race_counts)

    st.subheader("🔢 Age Distribution")
    age_series = pd.Series(list(age_history))
    st.bar_chart(age_series.value_counts(bins=10).sort_index())

# Show session summary if paused
if stop_time:
    duration = stop_time - start_time
    st.subheader("📝 Session Summary")
    st.markdown(f"**Total Duration:** {str(duration).split('.')[0]} (hh:mm:ss)")

    if len(emotion_history) > 0:
        most_common_emo = Counter(emotion_history).most_common(1)[0]
        st.markdown(f"**Most Frequent Emotion:** {most_common_emo[0]} ({most_common_emo[1]} frames)")

    if len(age_history) > 0:
        avg_age = sum(age_history) / len(age_history)
        st.markdown(f"**Average Age Estimate:** {avg_age:.1f}")

    if len(gender_history) > 0:
        gender_dist = pd.Series(list(gender_history)).value_counts(normalize=True) * 100
        st.markdown("**Gender Distribution:**")
        st.table(gender_dist.apply(lambda x: f"{x:.1f}%"))

    if len(race_history) > 0:
        race_dist = pd.Series(list(race_history)).value_counts(normalize=True) * 100
        st.markdown("**Race Distribution:**")
        st.table(race_dist.apply(lambda x: f"{x:.1f}%"))

# Export session log as CSV
if st.button("💾 Export Session Log as CSV"):
    if session_log:
        df_log = pd.DataFrame(session_log)
        csv = df_log.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, file_name="emotion_session_log.csv", mime="text/csv")
    else:
        st.warning("No session data to export yet.")
