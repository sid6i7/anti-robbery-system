import streamlit as st
from PIL import Image
import torch
import cv2
import os
import time

st.set_page_config(layout="wide")

model_folder = 'models'
model = None
confidence = 0.25

video_folder = 'data/video_source'

alert_threshold = 1

alerts = []

detected_objects_set = set()

def video_input():
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('mp4', 'mpv', 'avi'))]
    selected_video = st.sidebar.selectbox("Select a video file", video_files)
    vid_file = os.path.join(video_folder, selected_video)
    
    cap = cv2.VideoCapture(vid_file)
    custom_size = st.sidebar.checkbox("Custom frame size")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if custom_size:
        width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
        height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

    fps = 0
    st1, st2, st3 = st.columns(3)
    with st1:
        st1_text = st.markdown(f"{height}")
    with st2:
        st2_text = st.markdown(f"{width}")
    with st3:
        st3_text = st.markdown(f"{fps:.2f}")

    st.markdown("---")
    output = st.empty()
    prev_time = 0
    curr_time = 0

    detected_objects_set = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Can't read frame, stream ended? Exiting ....")
            break
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result, output_img = infer_image(frame)

        detected_objects = [result.names[pred[-1].item()] for pred in result.pred[0]]
        new_objects = [obj for obj in detected_objects if obj not in detected_objects_set]

        if new_objects:
            alert_message = f"{', '.join(new_objects)} found at {selected_video}."
            alerts.append(alert_message)
            st.warning(alert_message, icon="⚠️")
            detected_objects_set.update(new_objects)

        output.image(output_img)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        st1_text.markdown(f"**{height}**")
        st2_text.markdown(f"**{width}**")
        st3_text.markdown(f"**{fps:.2f}**")

    cap.release()


def image_input():
    img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
    if img_bytes:
        img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
        Image.open(img_bytes).save(img_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            result, img = infer_image(img_file)
            st.image(img, caption="Model prediction")

            if len(result.pred[0]) > 0:
                detected_objects = [result.names[pred[-1].item()] for pred in result.pred[0]]
                alert_message = f"{', '.join(detected_objects)} found in the uploaded image."
                alerts.append(alert_message)
                st.warning(alert_message, icon="⚠️")

def infer_image(img, size=None):
    model.conf = confidence
    result = model(img, size=size) if size else model(img)
    result.render()
    if(len(result.pred[0]))>0:
        print(result.names[result.pred[0][0][-1].item()])
    image = Image.fromarray(result.ims[0])
    return result, image

def load_model(model_path, device):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    model.to(device)
    print("model to ", device)
    return model

def main():
    global model, confidence, cfg_model_path

    st.title("Robbery Detection Dashboard")

    st.sidebar.title("Settings")

    available_models = [f for f in os.listdir(model_folder) if f.endswith('.pt')]
    selected_model = st.sidebar.selectbox("Select a model", available_models)

    model_path = os.path.join(model_folder, selected_model)

    if not os.path.isfile(model_path):
        st.warning("Model file not available!!!, please add it to the models folder.", icon="⚠️")
    else:
        if torch.cuda.is_available():
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
        else:
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

        model = load_model(model_path, device_option)

        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=0.45)

        if st.sidebar.checkbox("Custom Classes"):
            model_names = list(model.names.values())
            assigned_class = st.sidebar.multiselect("Select Classes", model_names, default=[model_names[0]])
            classes = [model_names.index(name) for name in assigned_class]
            model.classes = classes
        else:
            model.classes = list(model.names.keys())

        st.sidebar.markdown("---")

        input_option = st.sidebar.radio("Select input type: ", ['image', 'video'])

        if input_option == 'image':
            image_input()
        else:
            video_input()

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass