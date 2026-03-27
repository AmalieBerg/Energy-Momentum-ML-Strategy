"""
Setup Script for Energy Momentum ML Strategy
Helps verify installation and setup
"""
import sys
from pathlib import Path

print("\n" + "="*70)
print("Energy Momentum ML Strategy - Setup Verification")
print("="*70 + "\n")

# Check Python version
print("1. Checking Python version...")
if sys.version_info >= (3, 8):
    print(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
else:
    print(f"   ❌ Python {sys.version_info.major}.{sys.version_info.minor} (need 3.8+)")
    sys.exit(1)

# Check directories
print("\n2. Checking directory structure...")
required_dirs = [
    'src', 'src/data', 'src/models', 'src/backtest',
    'data/raw', 'data/processed', 'models/saved', 'results'
]
for dir_path in required_dirs:
    if Path(dir_path).exists():
        print(f"   ✅ {dir_path}/")
    else:
        print(f"   ❌ {dir_path}/ (missing)")

# Check key files
print("\n3. Checking key files...")
required_files = [
    'src/config.py',
    'src/data/data_pipeline.py',
    'src/models/garch_model.py',
    'requirements.txt',
    'README.md'
]
for file_path in required_files:
    if Path(file_path).exists():
        print(f"   ✅ {file_path}")
    else:
        print(f"   ❌ {file_path} (missing)")

# Check imports
print("\n4. Checking dependencies...")
dependencies = [
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('yfinance', 'yfinance'),
    ('arch', 'arch'),
    ('sklearn', 'scikit-learn'),
    ('matplotlib', 'matplotlib'),
    ('seaborn', 'seaborn')
]

missing = []
for module_name, pip_name in dependencies:
    try:
        __import__(module_name)
        print(f"   ✅ {pip_name}")
    except ImportError:
        print(f"   ❌ {pip_name} (not installed)")
        missing.append(pip_name)

if missing:
    print(f"\n   To install missing packages:")
    print(f"   pip install {' '.join(missing)}")
    print(f"\n   Or install all at once:")
    print(f"   pip install -r requirements.txt")

# Test config import
print("\n5. Testing configuration...")
try:
    sys.path.insert(0, str(Path.cwd()))
    from src.config import DATA_PATH, PROCESSED_PATH, MODEL_PATH, RESULTS_PATH
    print(f"   ✅ Config imported successfully")
    print(f"   📁 DATA_PATH: {DATA_PATH}")
    print(f"   📁 PROCESSED_PATH: {PROCESSED_PATH}")
    print(f"   📁 MODEL_PATH: {MODEL_PATH}")
    print(f"   📁 RESULTS_PATH: {RESULTS_PATH}")
except Exception as e:
    print(f"   ❌ Config import failed: {e}")

print("\n" + "="*70)
print("Setup Verification Complete!")
print("="*70 + "\n")

print("Next steps:")
print("1. If any checks failed, install missing dependencies")
print("2. Run: python src/data/data_pipeline.py")
print("3. Follow the README for the complete pipeline")
print()
