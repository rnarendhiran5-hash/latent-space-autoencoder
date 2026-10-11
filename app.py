import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, Model

st.set_page_config(page_title="Latent Space Explorer", page_icon="🧠", layout="wide")
st.title("🧠 Latent Space Visualization Using Autoencoders")
st.write("Train a small autoencoder on MNIST, compress handwritten digits into a 2D latent space, and explore how similar images group together.")

@st.cache_data
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return x_train[..., None], y_train, x_test[..., None], y_test

@st.cache_resource
def build_model():
    inp = layers.Input(shape=(28, 28, 1), name="image")
    x = layers.Flatten()(inp)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    latent = layers.Dense(2, name="latent_space")(x)
    x = layers.Dense(64, activation="relu")(latent)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(784, activation="sigmoid")(x)
    out = layers.Reshape((28, 28, 1))(x)
    autoencoder = Model(inp, out, name="autoencoder")
    encoder = Model(inp, latent, name="encoder")
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    return autoencoder, encoder

x_train, y_train, x_test, y_test = load_data()
autoencoder, encoder = build_model()

with st.sidebar:
    st.header("Experiment settings")
    epochs = st.slider("Training epochs", 1, 15, 5)
    train_size = st.select_slider("Training samples", options=[5000, 10000, 20000, 40000, 60000], value=20000)
    sample_count = st.slider("Points to visualize", 500, 5000, 2000, step=500)
    projection = st.radio("Latent visualization", ["Learned 2D latent space", "PCA comparison", "t-SNE comparison"])
    train_button = st.button("🚀 Train autoencoder", type="primary", use_container_width=True)
    st.caption("The MNIST dataset downloads automatically the first time the app runs.")

if "trained" not in st.session_state:
    st.session_state.trained = False

if train_button:
    progress = st.progress(0)
    status = st.empty()
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress.progress((epoch + 1) / epochs)
            status.write(f"Epoch {epoch + 1}/{epochs} — loss: {logs.get('loss', 0):.4f}")
    autoencoder.fit(
        x_train[:train_size], x_train[:train_size],
        epochs=epochs, batch_size=256, shuffle=True,
        validation_split=0.1, verbose=0, callbacks=[ProgressCallback()]
    )
    st.session_state.trained = True
    st.success("Training complete! Explore the learned representation below.")

if not st.session_state.trained:
    st.info("Click **Train autoencoder** in the sidebar to train the model and unlock the visualizations.")
    st.subheader("How it works")
    c1, c2, c3 = st.columns(3)
    c1.metric("Input image", "28 × 28")
    c2.metric("Latent space", "2 values")
    c3.metric("Output image", "28 × 28")
    st.markdown("""
    1. **Encoder:** compresses each image into two numbers.
    2. **Latent space:** similar digit images may appear near one another.
    3. **Decoder:** reconstructs the image from the compressed representation.
    """)
    st.stop()

n = min(sample_count, len(x_test))
images = x_test[:n]
labels = y_test[:n]
latent = encoder.predict(images, batch_size=256, verbose=0)

if projection == "PCA comparison":
    flat = images.reshape(n, -1)
    coords = PCA(n_components=2, random_state=42).fit_transform(flat)
    plot_title = "PCA projection of image pixels"
elif projection == "t-SNE comparison":
    flat = images.reshape(n, -1)
    # t-SNE is intentionally limited for interactive performance.
    n_tsne = min(n, 2000)
    coords = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto",
                  perplexity=min(30, max(5, (n_tsne - 1) // 3))).fit_transform(flat[:n_tsne])
    labels = labels[:n_tsne]
    images = images[:n_tsne]
    plot_title = "t-SNE projection of image pixels"
else:
    coords = latent
    plot_title = "Learned 2D latent space"

left, right = st.columns([1.4, 1])
with left:
    st.subheader("Latent-space map")
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=9, alpha=0.7)
    ax.set_title(plot_title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    fig.colorbar(scatter, ax=ax, ticks=range(10), label="Digit label")
    ax.grid(alpha=0.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Each dot is one image. Color indicates its true digit label; labels are used only for coloring, not training.")

with right:
    st.subheader("Original vs reconstructed")
    reconstructed = autoencoder.predict(images[:10], verbose=0)
    fig2, axes = plt.subplots(2, 5, figsize=(9, 4))
    for i, ax in enumerate(axes[0]):
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.set_title(f"Original: {labels[i]}")
        ax.axis("off")
    for i, ax in enumerate(axes[1]):
        ax.imshow(reconstructed[i].squeeze(), cmap="gray")
        ax.set_title("Rebuilt")
        ax.axis("off")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
    mse = mean_squared_error(images.reshape(-1), reconstructed.reshape(-1))
    st.metric("Reconstruction MSE (first 10 images)", f"{mse:.4f}")

st.subheader("Interpretation")
st.write("""
- Groups of the same color suggest that the representation captures visual similarities.
- Overlap between colors is normal: some handwritten digits look alike.
- The learned latent map is trained specifically to reconstruct images, not to classify them, so clusters may not be perfectly separated.
- PCA and t-SNE are comparison techniques applied to pixels; they are not the autoencoder's learned latent space.
""")
st.caption("Built with TensorFlow/Keras, Streamlit, Matplotlib, and scikit-learn.")
