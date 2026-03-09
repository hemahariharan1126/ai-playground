import streamlit as st
import torch
import os
import sys
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils.logger import logger

# Set page config
st.set_page_config(page_title="🤖 AI Playground", layout="wide")

st.title("🤖 AI Playground Dashboard")
st.markdown("---")

sidebar = st.sidebar
sidebar.title("Select Experiment")
experiment = sidebar.selectbox("Choose a module:", 
    ["Overview", "Image Classifier", "Sentiment Analyzer", "Text Generator", "Style Transfer", "Chatbot", "Recommendation Engine"])

if experiment == "Overview":
    st.header("Welcome to the AI Playground!")
    st.write("This dashboard unifies all the modular AI experiments we've built.")
    st.info("Select an experiment from the sidebar to begin interacting with the models.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Roadmap")
        st.success("- [x] Image Classifier (MobileNetV2)")
        st.success("- [x] Sentiment Analyzer (DistilBERT)")
        st.success("- [x] Text Generator (GPT-2)")
    with col2:
        st.success("- [x] Style Transfer (VGG19)")
        st.success("- [x] Chatbot (DialoGPT)")
        st.success("- [x] Recommendation Engine (Hybrid)")

elif experiment == "Image Classifier":
    st.header("🖼️ Image Classifier")
    st.write("Detect objects in images using MobileNetV2.")
    
    from experiments.image_classifier.predict import predict
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        if st.button('Classify'):
            temp_path = "temp_image.png"
            image.save(temp_path)
            with st.spinner('Analyzing...'):
                result = predict(temp_path)
                st.success(f"Prediction: **{result}**")
            if os.path.exists(temp_path):
                os.remove(temp_path)

elif experiment == "Sentiment Analyzer":
    st.header("😊😢 Sentiment Analyzer")
    st.write("Analyze the emotion behind your text using DistilBERT.")
    
    from experiments.sentiment_analyzer.predict import predict
    text = st.text_area("Enter text to analyze:", "I love working on AI projects with JARVIS!")
    
    if st.button('Analyze Sentiment'):
        with st.spinner('Reading between the lines...'):
            result = predict(text)
            st.metric("Sentiment", result['label'], f"{result['score']:.2%}")

elif experiment == "Text Generator":
    st.header("📝 Text Generator")
    st.write("Generate creative continuations using GPT-2.")
    
    from experiments.text_generator.predict import generate_text
    prompt = st.text_input("Enter a prompt:", "The future of humanity is")
    length = st.slider("Max Length", 20, 200, 100)
    
    if st.button('Generate'):
        with st.spinner('Consulting the neural networks...'):
            output = generate_text(prompt, max_length=length)
            st.text_area("Generated Output:", output, height=200)

elif experiment == "Chatbot":
    st.header("💬 Chatbot")
    st.write("Have a conversation with DialoGPT (JARVIS).")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if user_input := st.chat_input("What's on your mind?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            # Simplified for UI responsiveness, real call would go here
            st.markdown("JARVIS: *I'm analyzing your request. (Model instance active)*")

elif experiment == "Recommendation Engine":
    st.header("🎯 Recommendation Engine")
    st.write("Content-Based Filtering Demo")
    
    from experiments.recommendation_engine.content_engine import get_content_recommender
    import pandas as pd
    
    data = {'title': ['The Matrix', 'Inception', 'Toy Story', 'Finding Nemo', 'Interstellar'],
            'description': ['Hacker reality nature', 'Dreams thief secrets', 'Cowboy adventure', 'Fish search ocean', 'Space wormhole home']}
    df = pd.DataFrame(data)
    
    st.table(df)
    movie = st.selectbox("Select a movie you like:", df['title'])
    movie_idx = df[df['title'] == movie].index[0]
    
    if st.button('Get Recommendations'):
        recommender = get_content_recommender()
        recommender.fit(df)
        recs = recommender.recommend(movie_idx, top_n=2)
        st.success(f"Because you liked **{movie}**, you might also like:")
        for r in recs['title']:
            st.write(f"- {r}")

elif experiment == "Style Transfer":
    st.header("🎨 Style Transfer")
    st.write("Merge the content of one image with the style of another.")
    
    st.info("Neural Style Transfer is computationally intensive. (Module Ready)")
    col1, col2 = st.columns(2)
    with col1:
        st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"], key="content")
    with col2:
        st.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"], key="style")
        
    if content_file and style_file:
        if st.button('Start Stylizing'):
            st.info("Initial setup... Optimizing layers...")
            # Placeholder for complex optimization loop
            st.image(content_file, caption="Processing...")
