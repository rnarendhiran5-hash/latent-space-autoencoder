# Latent Space Visualization Using Autoencoders

An interactive mini-project that trains a neural-network autoencoder on MNIST handwritten digits and visualizes the compressed representations.

## Features
- Train an encoder-decoder autoencoder from the web app.
- Compress each 28×28 digit image into a **2-dimensional latent vector**.
- Plot latent points colored by their digit labels to inspect grouping.
- Compare learned latent space with PCA and t-SNE projections of raw pixels.
- View original and reconstructed digit images.
- Display reconstruction mean squared error (MSE).

## Tech stack
Python · TensorFlow/Keras · Streamlit · NumPy · Matplotlib · scikit-learn

## Run locally
Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed in the terminal. The MNIST dataset downloads automatically on first run. Training speed depends on your computer.

## How it works
1. **Encoder:** Flatten → Dense(128) → Dense(64) → Dense(2).
2. **Latent representation:** two numbers per image, shown on a scatter plot.
3. **Decoder:** Dense(64) → Dense(128) → Dense(784) → reshape to 28×28.
4. **Training objective:** binary cross-entropy reconstruction loss.
5. **Evaluation:** compare original and reconstructed images and inspect MSE.

## Project structure
```text
latent-space-autoencoder/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Notes
- Digit labels are used to color the plot, not as model targets.
- The learned latent space optimizes reconstruction; perfect class separation is not guaranteed.
- t-SNE can be slower, so the app caps that comparison at 2,000 samples.

## Push to GitHub
Create a new empty repository on GitHub named `latent-space-autoencoder`, then run:

```bash
git init
git add .
git commit -m "Build latent space visualization using autoencoders"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/latent-space-autoencoder.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.
