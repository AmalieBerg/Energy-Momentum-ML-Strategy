# 📁 File Organization Guide

## How to Organize Your Downloaded Files

All files are in one folder, but they need to go into the correct directory structure. Here's where each file goes:

### 📋 Root Directory Files (put in main folder)

```
energy-momentum-ml-strategy/
├── START_HERE.md          ← Read this first!
├── README.md              ← Your new professional README
├── MIGRATION_GUIDE.md     ← How to use new structure
├── CHANGELOG.md           ← Complete list of changes
├── LICENSE                ← MIT License
├── requirements.txt       ← Updated dependencies
├── .gitignore            ← Fixed .gitignore
└── setup.py              ← Setup verification script
```

### 📁 Create This Directory Structure

Before placing files, create these folders:

```bash
mkdir -p src/data
mkdir -p src/models
mkdir -p src/backtest
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/saved
mkdir -p results
mkdir -p scripts
mkdir -p tests
mkdir -p notebooks
```

### 🐍 Python Files Go In src/

```
src/
├── config.py                     ← Goes in src/
├── data/
│   └── data_pipeline.py         ← Goes in src/data/
├── models/
│   ├── garch_model.py           ← Goes in src/models/
│   ├── feature_engineering.py   ← Goes in src/models/
│   ├── regime_detector.py       ← Goes in src/models/
│   └── ml_training.py           ← Goes in src/models/
└── backtest/
    └── backtest_engine.py       ← Goes in src/backtest/
```

### ✅ Step-by-Step Setup

**Option 1: Manual (Recommended if you're keeping existing data)**

1. Download all files from this conversation
2. In your existing project, create the new folders:
   ```bash
   cd path/to/energy-momentum-ml-strategy
   mkdir -p src/data src/models src/backtest
   ```
3. Move files to their locations:
   - Root files → project root
   - `config.py` → `src/`
   - `data_pipeline.py` → `src/data/`
   - `garch_model.py`, `feature_engineering.py`, `regime_detector.py`, `ml_training.py` → `src/models/`
   - `backtest_engine.py` → `src/backtest/`
4. Create empty `__init__.py` files:
   ```bash
   touch src/__init__.py
   touch src/data/__init__.py
   touch src/models/__init__.py
   touch src/backtest/__init__.py
   ```
5. Keep your existing `data/` and `models/saved/` folders!

**Option 2: Fresh Start**

1. Create a new folder: `energy-momentum-ml-strategy-new`
2. Create the directory structure (see above)
3. Place all downloaded files in their correct locations
4. Copy your `data/` and `models/saved/` from old project if you have them

### 🔍 Quick Verification

After organizing, your structure should look like:

```
energy-momentum-ml-strategy/
├── START_HERE.md
├── README.md
├── MIGRATION_GUIDE.md
├── CHANGELOG.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── setup.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_pipeline.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── garch_model.py
│   │   ├── feature_engineering.py
│   │   ├── regime_detector.py
│   │   └── ml_training.py
│   └── backtest/
│       ├── __init__.py
│       └── backtest_engine.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── saved/
└── results/
```

### ✅ Verify Setup Works

```bash
# Test it works
python setup.py

# Should see all green checkmarks ✅
```

### 📦 What About My Old Files?

**Keep these from your old project:**
- `data/` folder (your downloaded data)
- `models/saved/` folder (your trained models)
- `results/` folder (your backtest results)

**Can delete these (now in src/):**
- Old `config.py` (replaced by `src/config.py`)
- Old `data/data_pipeline.py`
- Old `models/garch_model.py`
- Old `models/feature_engineering.py`
- Old `models/regime_detector.py`
- Old `models/ml_training.py`
- Old `models/backtest_engine.py`
- `models/ml_tranining.py` (typo file)
- `models/check_data.py` (debugging script)
- `test_setup.py` (replaced by `setup.py`)

### 🚨 Important Notes

1. **DO NOT delete your `data/` folder** - it has your downloaded market data
2. **DO NOT delete `models/saved/`** - it has your trained ML models
3. **DO keep `results/`** - it has your backtest results
4. **Replace** the root-level documentation files (README, etc.)

### 🎯 Next Steps

1. Organize files as shown above
2. Run `python setup.py` to verify
3. Read `START_HERE.md` for usage guide
4. Test: `python src/data/data_pipeline.py`

**Need help?** Check MIGRATION_GUIDE.md for detailed instructions.
