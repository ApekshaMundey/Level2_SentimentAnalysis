import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    tokenizer=AutoTokenizer.from_pretrained("Apss30/sentiment-model")
    model=AutoModelForSequenceClassification.from_pretrained("Apss30/sentiment-model")
    
    model.to("cpu")   # IMPORTANT for LIME
    model.eval()
    
    return tokenizer, model

tokenizer, model = load_model()

# ------------------------------
# Prediction Function (for app)
# ------------------------------
def predict_sentiment(text):
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    confidence, prediction = torch.max(probs, dim=1)

    label = "Positive 😊" if prediction.item() == 1 else "Negative 😡"

    return label, confidence.item()

# ------------------------------
# LIME Prediction Function
# ------------------------------
def predict_proba(texts):

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to("cpu") for k, v in inputs.items()}  # ✅ ensure CPU

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    return probs.detach().cpu().numpy()   # ✅ FIXED

# ------------------------------
# LIME Explainer
# ------------------------------
@st.cache_resource
def load_explainer():
    return LimeTextExplainer(class_names=["Negative", "Positive"])

explainer = load_explainer()

def highlight_text(text, exp_list):
    word_weights = {w.lower(): v for w, v in exp_list}

    words = text.split()
    highlighted_text = ""

    for word in words:
        clean_word = word.strip(".,!?").lower()

        if clean_word in word_weights:
            weight = word_weights[clean_word]
            intensity = min(abs(weight) * 3, 1)

            if weight > 0:
                color = f"rgba(0,255,0,{intensity})"   # green
            else:
                color = f"rgba(255,0,0,{intensity})"   # red

            highlighted_text += f"<span style='background-color:{color}; padding:2px; margin:1px; border-radius:4px'>{word}</span> "
        else:
            highlighted_text += word + " "

    return highlighted_text

positive_words = {"amazing", "great", "awesome", "good", "love", "excellent", "fantastic", "enjoy"}
negative_words = {"bad", "boring", "waste", "wasting", "terrible", "worst", "poor", "disappointing"}

def detect_sentiment_shift(text):
    words = [w.strip(".,!?").lower() for w in text.split()]

    pos_seen = False

    for w in words:
        if w in positive_words:
            pos_seen = True

        elif pos_seen and w in negative_words:
            return True

    return False

# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬")

st.title("🎬 Sentiment Analyzer with Explainability")

review = st.text_area("Enter your movie review:")

if st.button("Predict & Explain"):
    st.session_state["run"] = True
    st.session_state["last_review"] = review

if "run" in st.session_state and st.session_state["run"] and review == st.session_state.get("last_review", ""):
    if review.strip() == "":
        st.warning("Please enter a review!")
    
    else:
        # Prediction
        label, confidence = predict_sentiment(review)

        st.subheader(f"Prediction: {label}")
        if detect_sentiment_shift(review):
            confidence *= 0.85
            st.warning("⚠️ Possible sarcasm or mixed sentiment detected")
        st.write(f"Confidence: {confidence:.2f}")

        show_explain = st.checkbox("Show Explanation (LIME)")

        if show_explain:
            with st.spinner("Generating explanation... ⏳"): 
                exp = explainer.explain_instance(
                    review,
                    predict_proba,
                    num_features=6,
                    num_samples=500
                )

            if not exp.as_list():
                st.warning("LIME couldn't generate a strong explanation.")
            else:
                st.subheader("✨ Highlighted Explanation")
                highlighted = highlight_text(review, exp.as_list())
                st.markdown(f"""
                <div style="
                    background-color:#1c1f26;
                    padding:15px;
                    border-radius:10px;
                    font-size:16px;
                    line-height:1.6;
                ">
                {highlighted}
                </div>
                """, unsafe_allow_html=True)   