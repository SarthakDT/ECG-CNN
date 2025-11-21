# GitHub Setup Instructions

Follow these steps to create a GitHub repository and push the ECG classifier code:

## Prerequisites
- GitHub account at https://github.com
- Git installed on your system

## Step 1: Install Git (if not already installed)

**Windows:**
Download and install from: https://git-scm.com/download/win

Verify installation:
```powershell
git --version
```

## Step 2: Configure Git (First Time Only)

```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 3: Create a New Repository on GitHub

1. Go to https://github.com/new
2. Enter repository name: `ecg-classifier` (or your preferred name)
3. Description: "Binary ECG Signal Classifier (Calm vs Excited) using Conv1D + Engineered Features"
4. Choose **Public** or **Private**
5. **Do NOT initialize** with README, .gitignore, or license (we already have these)
6. Click "Create repository"
7. Copy the repository URL (HTTPS or SSH)

## Step 4: Initialize Git and Push

From the project directory:

```powershell
cd "C:\Users\sarth\Desktop\Stuff\Interesting shit\ECG"

# Initialize git repo
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: ECG binary classifier with Conv1D model and inference plots"

# Add remote repository (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/ecg-classifier.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 5: Verify on GitHub

1. Go to your repository URL: `https://github.com/YOUR_USERNAME/ecg-classifier`
2. Confirm all files appear correctly
3. README should render automatically

## Pushing Future Changes

After making changes:

```powershell
git add .
git commit -m "Your commit message"
git push origin main
```

## Files Included in Repository

- `train.py` — Main training and inference pipeline
- `generate_inference_plots.py` — Plot generation script
- `sample_inference.py` — Example inference output
- `ecg_pf_norm_30.keras` — Trained model (Keras format)
- `csvs/` — Sample ECG data files
- `plots/` — Generated inference visualizations
- `README.md` — Comprehensive documentation
- `requirements.txt` — Python dependencies
- `LICENSE` — MIT License
- `.gitignore` — Git ignore rules

## Troubleshooting

**"fatal: not a git repository"**
- Run `git init` first

**"fatal: 'origin' does not appear to be a 'git' repository"**
- Verify your remote URL: `git remote -v`
- Re-add if needed: `git remote add origin https://...`

**Authentication issues (HTTPS)**
- Use GitHub Personal Access Token instead of password
- Generate at: https://github.com/settings/tokens

**SSH Alternative**
- Set up SSH key: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
- Use SSH URL instead: `git@github.com:YOUR_USERNAME/ecg-classifier.git`
