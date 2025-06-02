import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Personality Predictor", page_icon="🧠", layout="centered")
st.title("🧠 Personality Predictor")

st.sidebar.header("Upload Training History")
uploaded_hist = st.sidebar.file_uploader("training_history.npz", type="npz")

mode = st.sidebar.radio("Mode", ["Predict Personality", "Visualize Training"])

weights = np.array([  
    1.36953457,  
    0.29358210,  
   -0.47103967,  
    0.17539252,  
    0.20089438,  
   -1.07238178,  
    0.20322112  
])
bias = 0.20541888

means = np.array([3.9175, 0.486034483, 4.8, 3.62155172, 0.486034483, 7.27413793, 4.97844827])
stds  = np.array([2.77603885, 0.49992397, 2.76505043, 2.13312453, 0.49992397, 4.05783294, 2.95865705])
num_idx = [0, 2, 3, 5, 6]  # numerical feature indices

def preprocess(x_raw):
    x = x_raw.astype(float).copy()
    x[num_idx] = (x[num_idx] - means[num_idx]) / stds[num_idx]
    return x

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def predict(x_scaled):
    z = np.dot(x_scaled, weights) + bias
    p = sigmoid(z)
    return ("Introvert" if p >= 0.5 else "Extrovert"), p

# — Prediction Mode —
if mode == "Predict Personality":
    st.subheader("Enter your details:")
    with st.form("predict_form"):
        t_alone = st.slider("Hours spent alone daily (0–11)", 0, 11, 5, 1)
        fear    = st.radio("Stage fright?", ("Yes","No"))
        soc     = st.slider("Social event attendance (0–10)", 0, 10, 5, 1)
        out     = st.slider("Going outside frequency (0–7)", 0, 7, 3, 1)
        drained = st.radio("Drained after socializing?", ("Yes","No"))
        friends = st.slider("Number of close friends (0–15)", 0, 15, 7, 1)
        posts   = st.slider("Social media post frequency (0–10)", 0, 10, 3, 1)
        go = st.form_submit_button("Predict")

    if go:
        raw = np.array([
            t_alone,
            1 if fear=="Yes" else 0,
            soc,
            out,
            1 if drained=="Yes" else 0,
            friends,
            posts
        ], dtype=float)
        scaled = preprocess(raw)
        cls, prob = predict(scaled)
        st.success(f"### Prediction: **{cls}**")
        st.write(f"Probability Introvert: {prob*100:.2f}%  |  Extrovert: {(1-prob)*100:.2f}%")

# — Visualization Mode —
else:
    st.subheader("Upload training_history.npz to see your gradient-descent curves")
    if uploaded_hist is None:
        st.error("Please upload **training_history.npz** in the sidebar.")
        st.stop()

    data = np.load(io.BytesIO(uploaded_hist.read()))
    J_history = data["J"]
    W_history = data["W"]
    b_history = data["b"]
    iters = np.arange(len(J_history))

    # Cost vs Iteration
    fig1, ax1 = plt.subplots()
    ax1.plot(iters, J_history, 'b-')
    ax1.set(xlabel="Iteration", ylabel="Cost", title="Cost vs. Iteration")
    ax1.grid(True)
    st.pyplot(fig1)

    # Weights vs Iteration
    fig2, ax2 = plt.subplots()
    for idx, name in enumerate([
        "Time_alone","Stage_fear","Social_event_attendance",
        "Going_outside","Drained_after_socializing","Friends_circle_size",
        "Post_frequency"
    ]):
        ax2.plot(iters, W_history[:, idx], label=name)
    ax2.set(xlabel="Iteration", ylabel="Weight Value", title="Weights vs. Iteration")
    ax2.legend(bbox_to_anchor=(1.05,1), loc='upper left'); ax2.grid(True)
    st.pyplot(fig2)

    # Bias vs Iteration
    fig3, ax3 = plt.subplots()
    ax3.plot(iters, b_history, 'r-')
    ax3.set(xlabel="Iteration", ylabel="Bias Value", title="Bias vs. Iteration")
    ax3.grid(True)
    st.pyplot(fig3)
