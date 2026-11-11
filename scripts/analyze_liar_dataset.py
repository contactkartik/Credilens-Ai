"""
Quick analysis script for LIAR dataset to get statistics without running Jupyter.
Uses only standard library.
"""
from collections import Counter

# Define column names based on README
columns = [
    'id', 'label', 'statement', 'subjects', 'speaker',
    'speaker_job', 'state_info', 'party_affiliation',
    'barely_true_counts', 'false_counts', 'half_true_counts',
    'mostly_true_counts', 'pants_on_fire_counts', 'context'
]

labels = ['barely-true', 'false', 'half-true', 'mostly-true', 'pants-fire', 'true']

def read_tsv(filepath):
    """Read TSV file and return list of rows."""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            rows.append(line.strip().split('\t'))
    return rows

def analyze_file(filepath, name):
    """Analyze a single TSV file."""
    rows = read_tsv(filepath)
    data = []
    for row in rows:
        if len(row) == len(columns):
            data.append(dict(zip(columns, row)))

    print(f"\n{name}:")
    print(f"  Total rows: {len(rows)}")
    print(f"  Valid rows: {len(data)}")

    # Label distribution
    label_counts = Counter([row['label'] for row in data])
    print(f"\n  Label Distribution:")
    for label in labels:
        count = label_counts.get(label, 0)
        pct = (count / len(data)) * 100 if len(data) > 0 else 0
        print(f"    {label:15s}: {count:4d} ({pct:5.2f}%)")

    # Missing values
    missing_by_col = {col: 0 for col in columns}
    for row in data:
        for col in columns:
            if row[col] == '' or row[col] is None:
                missing_by_col[col] += 1

    total_missing = sum(missing_by_col.values())
    print(f"\n  Missing Values: {total_missing}")
    for col, count in missing_by_col.items():
        if count > 0:
            print(f"    {col}: {count}")

    # Statement length
    lengths = [len(row['statement']) for row in data if row['statement']]
    if lengths:
        print(f"\n  Statement Length:")
        print(f"    Mean: {sum(lengths)/len(lengths):.2f}")
        print(f"    Median: {sorted(lengths)[len(lengths)//2]:.2f}")
        print(f"    Min: {min(lengths)}")
        print(f"    Max: {max(lengths)}")

    # Sample statements
    print(f"\n  Sample Statements:")
    for label in labels:
        for row in data:
            if row['label'] == label:
                print(f"    {label}: {row['statement'][:80]}...")
                break

    return data, label_counts, missing_by_col

print("="*60)
print("LIAR DATASET ANALYSIS")
print("="*60)

print(f"\nColumns ({len(columns)}):")
for i, col in enumerate(columns, 1):
    print(f"  {i}. {col}")

print(f"\nLabels ({len(labels)}):")
for i, label in enumerate(labels, 1):
    print(f"  {i}. {label}")

# Analyze all files
train_data, train_labels, train_missing = analyze_file('../data/raw/train.tsv', 'TRAIN SET')
valid_data, valid_labels, valid_missing = analyze_file('../data/raw/valid.tsv', 'VALIDATION SET')
test_data, test_labels, test_missing = analyze_file('../data/raw/test.tsv', 'TEST SET')

# Combined statistics
total_samples = len(train_data) + len(valid_data) + len(test_data)
combined_labels = Counter()
combined_labels.update(train_labels)
combined_labels.update(valid_labels)
combined_labels.update(test_labels)

print("\n" + "="*60)
print("COMBINED STATISTICS")
print("="*60)
print(f"\nTotal Samples: {total_samples}")
print(f"  Train: {len(train_data)}")
print(f"  Validation: {len(valid_data)}")
print(f"  Test: {len(test_data)}")

print(f"\nCombined Label Distribution:")
for label in labels:
    count = combined_labels.get(label, 0)
    pct = (count / total_samples) * 100 if total_samples > 0 else 0
    print(f"  {label:15s}: {count:4d} ({pct:5.2f}%)")

total_missing = sum(train_missing.values()) + sum(valid_missing.values()) + sum(test_missing.values())
print(f"\nTotal Missing Values: {total_missing}")

print("\n" + "="*60)
