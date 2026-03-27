"""
Git Cleanup Helper
Identifies files that should not be pushed to GitHub
"""

from pathlib import Path
import sys

project_root = Path.cwd()

print("="*70)
print("🗂️  GIT CLEANUP - IDENTIFYING UNWANTED FILES")
print("="*70)

# Files to exclude (temporary/diagnostic files from our troubleshooting)
unwanted_patterns = [
    'diagnose_*.py',
    'fix_*.py', 
    'patch_*.py',
    'quick_fix_*.py',
    'clear_cache.py',
    'build_cache.py',
    'compare_*.py',
    'evaluate_*.py',
    'manual_fix_*.py',
    '*_FIX_*.md',
    'diagnostics_*.png',
    '*.backup',
    '*_backup_*',
    'cached_*.csv',
    'performance_summary_post_fix.png',
    'portfolio_summary_post_fix.csv',
]

print("\n📋 Scanning for temporary/diagnostic files...\n")

files_to_remove = []

for pattern in unwanted_patterns:
    matches = list(project_root.glob(pattern))
    if matches:
        print(f"Pattern: {pattern}")
        for file in matches:
            print(f"  • {file.name}")
            files_to_remove.append(file)

# Also check in subdirectories
for pattern in unwanted_patterns:
    matches = list(project_root.glob(f"**/{pattern}"))
    for file in matches:
        if file not in files_to_remove and file.is_file():
            rel_path = file.relative_to(project_root)
            print(f"  • {rel_path}")
            files_to_remove.append(file)

if not files_to_remove:
    print("✅ No unwanted files found!")
else:
    print(f"\n📊 Found {len(files_to_remove)} file(s) to clean up")
    
    print("\n" + "="*70)
    print("OPTIONS")
    print("="*70)
    print("""
Option 1: DELETE these files (recommended for fresh git push)
Option 2: KEEP these files (they'll be ignored by .gitignore)
Option 3: CANCEL (do nothing)

If you choose Option 2, make sure .gitignore is properly configured.
    """)
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == '1':
        print("\n🗑️  Deleting unwanted files...")
        deleted_count = 0
        for file in files_to_remove:
            try:
                file.unlink()
                print(f"  ✓ Deleted: {file.name}")
                deleted_count += 1
            except Exception as e:
                print(f"  ✗ Error deleting {file.name}: {e}")
        
        print(f"\n✅ Deleted {deleted_count} file(s)")
        
    elif choice == '2':
        print("\n✅ Files will be kept but ignored by git")
        print("   Make sure .gitignore is up to date!")
        
    else:
        print("\n⏭️  No changes made")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)

print("""
1. Update .gitignore:
   - Replace your current .gitignore with the new one provided
   - Or manually add the unwanted patterns

2. Check what git will track:
   git status

3. Add files to git:
   git add .

4. Commit:
   git commit -m "Complete ML energy momentum trading strategy"

5. Push to GitHub:
   git push origin main
""")

print("="*70)
