import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif,SelectKBest, f_classif
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from tensorflow.keras.models import load_model
import streamlit as st


model = load_model('sleep_Ann_v2.h5')
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

with st.form("my_form"):
    st.number_input("duration_minutes",key="duration_minutes",step=0.001,format="%0.3f")
    st.number_input("sleep_latency_minutes",key="sleep_latency_minutes",step=0.001,format="%0.3f")
    st.number_input("wake_after_sleep_onset_minutes",key="wake_after_sleep_onset_minutes",step=0.001,format="%0.3f")
    st.number_input("sleep_efficiency_pct",key="sleep_efficiency_pct",step=0.001,format="%0.3f")
    st.number_input("sleep_score",key="sleep_score",step=0.001,format="%0.3f")
    st.number_input("sleep_stage_deep_pct",key="sleep_stage_deep_pct",step=0.001,format="%0.3f")
    st.number_input("sleep_stage_light_pct",key="sleep_stage_light_pct",step=0.001,format="%0.3f")
    st.number_input("sleep_stage_rem_pct",key="sleep_stage_rem_pct",step=0.001,format="%0.3f")
    st.number_input("sleep_stage_awake_pct",key="sleep_stage_awake_pct",step=0.001,format="%0.3f")
    st.number_input("heart_rate_mean_bpm",key="heart_rate_mean_bpm",step=0.001,format="%0.3f")
    st.number_input("heart_rate_min_bpm",key="heart_rate_min_bpm",step=0.001,format="%0.3f")
    st.number_input("heart_rate_max_bpm",key="heart_rate_max_bpm",step=0.001,format="%0.3f")
    st.number_input("hrv_rmssd_ms",key="hrv_rmssd_ms",step=0.001,format="%0.3f")
    st.number_input("respiration_rate_bpm",key="respiration_rate_bpm",step=0.001,format="%0.3f")
    st.number_input("spo2_mean_pct",key="spo2_mean_pct",step=0.001,format="%0.3f")
    st.number_input("spo2_min_pct",key="spo2_min_pct",step=0.001,format="%0.3f")
    st.number_input("movement_count",key="movement_count",step=0.001,format="%0.3f")
    st.number_input("snore_events",key="snore_events",step=0.001,format="%0.3f")
    st.number_input("ambient_noise_db",key="ambient_noise_db",step=0.001,format="%0.3f")
    st.number_input("room_temperature_c",key="room_temperature_c",step=0.001,format="%0.3f")
    st.number_input("room_humidity_pct",key="room_humidity_pct",step=0.001,format="%0.3f")
    st.number_input("step_count_day",key="step_count_day",step=0.001,format="%0.3f")
    st.number_input("caffeine_mg",key="caffeine_mg",step=0.001,format="%0.3f")
    st.number_input("alcohol_units",key="alcohol_units",step=0.001,format="%0.3f")
    st.number_input("medication_flag",key="medication_flag",step=0.001,format="%0.3f")
    st.number_input("jetlag_hours",key="jetlag_hours",step=0.001,format="%0.3f")
    st.number_input("age",key="age",step=0.001,format="%0.3f")
    st.number_input("weight_kg",key="weight_kg",step=0.001,format="%0.3f")
    st.number_input("height_cm",key="height_cm",step=0.001,format="%0.3f")
    st.number_input("bedtime_consistency_std_min",key="bedtime_consistency_std_min",step=0.001,format="%0.3f")
    st.number_input("stress_score",key="stress_score",step=0.001,format="%0.3f")
    st.number_input("activity_before_bed_min",key="activity_before_bed_min",step=0.001,format="%0.3f")
    st.number_input("screen_time_before_bed_min",key="screen_time_before_bed_min",step=0.001,format="%0.3f")
    st.number_input("apnea_risk_score",key="apnea_risk_score",step=0.001,format="%0.3f")
    st.number_input("nap_duration_minutes",key="nap_duration_minutes",step=0.001,format="%0.3f")
    st.number_input("step_count_day_KM",key="step_count_day_KM",step=0.001,format="%0.3f")


    submitted = st.form_submit_button('Predict')

if submitted:
    #baseline_value = st.write(f"baseline value: {st.session_state.baseline_value}")
    #accelerations = st.write(f"accelerations: {st.session_state.accelerations}")
    x_test=[st.session_state.duration_minutes,
            st.session_state.sleep_latency_minutes,
            st.session_state.wake_after_sleep_onset_minutes,
            st.session_state.sleep_efficiency_pct,
            st.session_state.sleep_score,
            st.session_state.sleep_stage_deep_pct,
            st.session_state.sleep_stage_light_pct,
            st.session_state.sleep_stage_rem_pct,
            st.session_state.sleep_stage_awake_pct,
            st.session_state.heart_rate_mean_bpm,
            st.session_state.heart_rate_min_bpm,
            st.session_state.heart_rate_max_bpm,
            st.session_state.hrv_rmssd_ms,
            st.session_state.respiration_rate_bpm,
            st.session_state.spo2_mean_pct,
            st.session_state.spo2_min_pct,
            st.session_state.movement_count,
            st.session_state.snore_events,
            st.session_state.ambient_noise_db,
            st.session_state.room_temperature_c,
            st.session_state.room_humidity_pct,
            st.session_state.step_count_day,
            st.session_state.caffeine_mg,
            st.session_state.alcohol_units,
            st.session_state.medication_flag,
            st.session_state.jetlag_hours,
            st.session_state.age,
            st.session_state.weight_kg,
            st.session_state.height_cm,
            st.session_state.bedtime_consistency_std_min,
            st.session_state.stress_score,
            st.session_state.activity_before_bed_min,
            st.session_state.screen_time_before_bed_min,
            st.session_state.apnea_risk_score,
            st.session_state.nap_duration_minutes,
            st.session_state.step_count_day_KM
            ]
    y_pred = model.predict(np.array(x_test).reshape(1,-1))
    val_predict=np.argmax(y_pred)
    st.write(y_pred)
    st.write(val_predict)
    if val_predict==0:
        st.success("la classe est Normale")
    elif val_predict==1:
        st.error("la classe est Suspect")

    
    






