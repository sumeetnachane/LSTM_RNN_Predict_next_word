import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="✨",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #0f172a, #1e293b);
    color: white;
}

.stApp {
    background: linear-gradient(to right, #0f172a, #1e293b);
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #38bdf8;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 40px;
}

.prediction-box {
    background-color: #111827;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #334155;
    text-align: center;
    margin-top: 25px;
    box-shadow: 0px 0px 15px rgba(56,189,248,0.3);
}

.prediction-text {
    font-size: 32px;
    font-weight: bold;
    color: #22c55e;
}

.footer {
    text-align: center;
    margin-top: 60px;
    color: gray;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = load_model('next_word_lstm.h5')

# ---------------- LOAD TOKENIZER ----------------
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# ---------------- PREDICTION FUNCTION ----------------
def predict_next_word(model, tokenizer, text, max_sequence_len):

    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len-1,
        padding='pre'
    )

    predicted = model.predict(token_list, verbose=0)

    predicted_word_index = np.argmax(predicted, axis=1)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return None

# ---------------- UI ----------------

st.markdown(
    '<div class="title">✨ Next Word Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">LSTM + Early Stopping + Shakespeare Hamlet Dataset</div>',
    unsafe_allow_html=True
)

# Input Box
input_text = st.text_input(
    "Enter your sentence:",
    placeholder="Example: To be or not to"
)

# Button
predict_btn = st.button("🚀 Predict Next Word")

# Prediction
if predict_btn:

    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:

        with st.spinner("Predicting next word..."):

            max_sequence_len = model.input_shape[1] + 1 ## Next text +1 se mila 

            next_word = predict_next_word(
                model,
                tokenizer,
                input_text,
                max_sequence_len
            )

        st.markdown(f"""
        <div class="prediction-box">
            <p style="font-size:20px;color:#94a3b8;">
                Predicted Next Word
            </p>
            <div class="prediction-text">
                {next_word}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown(
    '<div class="footer">Built with TensorFlow, LSTM & Streamlit 🚀</div>',
    unsafe_allow_html=True
)