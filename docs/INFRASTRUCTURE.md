# Infrastructure Setup — For Your First Training Run

You need three things before running the Colab notebook:

1. A **Cloudflare R2 bucket** to store datasets and trained models.
2. **API credentials** for that bucket.
3. This repo pushed to **GitHub** so Colab can `git clone` it.

Follow the steps in order. You'll do each once.

---

## 1. Cloudflare R2 bucket

R2 is Cloudflare's S3-compatible object store. 10 GB storage + 1M operations/month are free — more than enough for v0.

### 1.1 Create an account and enable R2

1. Go to [dash.cloudflare.com](https://dash.cloudflare.com) and sign up (free).
2. In the sidebar, click **R2 Object Storage** → **Get Started**.
3. If it asks for payment info, add a card. You won't be charged unless you exceed the free tier — but Cloudflare requires a card on file.

### 1.2 Create the bucket

1. **Create bucket** → Name: `nutrilens-ml` → Location: **Automatic** → **Create**.
2. Note your **Account ID** (top-right of the R2 page, also in the URL). You'll need it.

### 1.3 Create API credentials

1. Back on the R2 page, click **Manage R2 API Tokens** → **Create API Token**.
2. Name: `nutrilens-ml-colab`.
3. Permissions: **Object Read & Write**.
4. Specify bucket: **Apply to specific buckets only** → select `nutrilens-ml`.
5. TTL: leave at **Forever** (we'll rotate later).
6. **Create API Token**.
7. **Copy all four values immediately** (Cloudflare will not show the secret again):
   - Access Key ID
   - Secret Access Key
   - Endpoint URL (should look like `https://<account-id>.r2.cloudflarestorage.com`)
   - Bucket name — `nutrilens-ml`

Paste them into a password manager. You'll paste them into the Colab notebook in a minute.

### 1.4 Update the Rust backend to know about the bucket

Edit `nutrilens-prod/.env` so future upload flows target the real bucket:

```
S3_BUCKET=nutrilens-ml
S3_REGION=auto
S3_ENDPOINT=https://<your-account-id>.r2.cloudflarestorage.com
S3_ACCESS_KEY_ID=<access key>
S3_SECRET_ACCESS_KEY=<secret key>
S3_PUBLIC_URL_BASE=https://<your-account-id>.r2.cloudflarestorage.com/nutrilens-ml
```

Restart the backend (`cargo run`) after saving.

---

## 2. Push `nutrilens-ml` to GitHub

Colab can't see your laptop's filesystem. It needs to pull the repo from GitHub.

### 2.1 Create the GitHub repo

1. [github.com/new](https://github.com/new).
2. Name: `nutrilens-ml`. Private. **No** README / gitignore / license — the repo already has them.
3. **Create repository**.

### 2.2 Push from your laptop

From inside `nutrilens-ml/` on your Mac:

```bash
git remote add origin git@github.com:<your-username>/nutrilens-ml.git
git push -u origin main
```

If you don't have SSH keys set up, use the HTTPS URL GitHub shows you and authenticate with a personal access token, or run `gh repo create` if you have the `gh` CLI.

### 2.3 Make Colab able to clone a private repo

Two options:

- **Easiest**: make the repo public (you can flip it back to private any time). No code in here is sensitive.
- **Private stays private**: generate a [fine-grained personal access token](https://github.com/settings/tokens?type=beta) with read-only access to this one repo, and you'll paste it into Colab at clone time.

I'd pick public for now. Flip to private later.

---

## 3. Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com).
2. **File → Upload notebook** → upload `notebooks/train_plate_classifier_colab.ipynb` from this repo.
3. **Runtime → Change runtime type → Hardware accelerator: GPU** (T4 is fine, free tier).
4. **Runtime → Run all**.

The notebook will prompt you for the R2 credentials when it needs them. Training on Food-101 takes roughly **45–90 minutes** on a free T4. When it finishes, the trained `.onnx` will be uploaded to `r2://nutrilens-ml/models/plate-classifier/0.1.0/`.

---

## 4. After training: convert to CoreML

CoreML conversion happens on your Mac (not Colab).

```bash
cd nutrilens-ml
pip install -e ".[export]"  # installs coremltools
nutrilens-ml export plate --checkpoint <path-to-downloaded-onnx>
```

This produces a `.mlpackage`. Drag it into Xcode → `NutriLens` target → **Copy items if needed**. Your iOS app can then load it via CoreML.

That final Xcode step is a v1 task — for v0, just getting an ONNX uploaded to R2 is a legitimate end-of-first-training milestone.
