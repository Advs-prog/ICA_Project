## Step 1: Mixing

Two source signals (a sine wave and a square wave) are mixed with a matrix A,
so each "microphone" recording contains a bit of both sources.
The goal of ICA is to recover the original sources from the mixtures alone.

X = A @ S

![Toy mixing example](images/toy_mixing.png)

## Step 2: Whitening

Whitening makes the mixed signals uncorrelated with equal variance.
After whitening, the covariance matrix is the identity matrix.

![Whitening on the toy example](images/toy_whitening.png)

## Step 3: FastICA

After whitening, only a rotation is left to find. FastICA searches for the
rotation that makes the outputs as non-Gaussian as possible.

Correlation between true sources (rows) and recovered signals (columns):

|                    | Recovered 1 | Recovered 2 |
|--------------------|-------------|-------------|
| True source 1 (sine)   | 0.086   | 0.996       |
| True source 2 (square) | 1.000   | ~0          |

Each true source matches exactly one recovered signal. The order is swapped,
which is the permutation ambiguity of ICA.

![FastICA on the toy example](images/toy_ica.png)


## Final step (complete after deployment)

### Publish a hosted demo

The repository now includes a static browser demo in `index.html`. It displays `images/results.png` and lets visitors play the generated WAV files in `audio/`. The Python script remains a separate command-line demo.

After pushing the repository to GitHub, create a **Static Site** on Render connected to that repository. The page is at the repository root, so use:

- **Build command:** leave blank
- **Publish directory:** `.`

Deploy the site, open its public URL, and verify the plot and audio links work. Then replace the placeholder below with the Render URL:

**Hosted demo:** `ADD_RENDER_URL_HERE`

The hosted page displays the saved results; it does not run ICA in the browser. Keep private or unlicensed source audio out of the public repository, and do not commit the local `venv/` folder.


