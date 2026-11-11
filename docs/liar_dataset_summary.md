# LIAR Dataset Summary

## Dataset Statistics

### Total Size
- **Total Samples**: 12,707 (after removing invalid rows)
- **Train Set**: 10,169 samples (80.0%)
- **Validation Set**: 1,272 samples (10.0%)
- **Test Set**: 1,266 samples (10.0%)

### Column Information (14 columns)
1. `id` - Statement ID
2. `label` - Truthfulness label
3. `statement` - The statement text
4. `subjects` - Subject categories
5. `speaker` - Speaker name
6. `speaker_job` - Speaker's job title
7. `state_info` - State information
8. `party_affiliation` - Political party
9. `barely_true_counts` - Historical barely-true count
10. `false_counts` - Historical false count
11. `half_true_counts` - Historical half-true count
12. `mostly_true_counts` - Historical mostly-true count
13. `pants_on_fire_counts` - Historical pants-on-fire count
14. `context` - Context/venue of statement

### Six Original Labels
1. `barely-true`
2. `false`
3. `half-true`
4. `mostly-true`
5. `pants-fire`
6. `true`

## Label Distribution

### Combined Dataset
- `barely-true`: 2,091 (16.46%)
- `false`: 2,486 (19.56%)
- `half-true`: 2,615 (20.58%)
- `mostly-true`: 2,444 (19.23%)
- `pants-fire`: 1,035 (8.15%)
- `true`: 2,036 (16.02%)

### Train Set Distribution
- `barely-true`: 1,643 (16.16%)
- `false`: 1,977 (19.44%)
- `half-true`: 2,106 (20.71%)
- `mostly-true`: 1,949 (19.17%)
- `pants-fire`: 831 (8.17%)
- `true`: 1,663 (16.35%)

### Validation Set Distribution
- `barely-true`: 234 (18.40%)
- `false`: 262 (20.60%)
- `half-true`: 247 (19.42%)
- `mostly-true`: 249 (19.58%)
- `pants-fire`: 113 (8.88%)
- `true`: 167 (13.13%)

### Test Set Distribution
- `barely-true`: 214 (16.90%)
- `false`: 247 (19.51%)
- `half-true`: 262 (20.70%)
- `mostly-true`: 246 (19.43%)
- `pants-fire`: 91 (7.19%)
- `true`: 206 (16.27%)

## Data Quality

### Missing Values
- **Total Missing Values**: 6,255
- **Missing in `speaker_job`**: 3,536 (27.8% of samples)
- **Missing in `state_info`**: 2,719 (21.4% of samples)
- **Other columns**: No missing values

### Duplicate Records
- **Train Set**: 0 duplicates
- **Validation Set**: 0 duplicates
- **Test Set**: 0 duplicates

### Invalid Rows
- **Train Set**: 100 invalid rows (malformed TSV)
- **Validation Set**: 12 invalid rows
- **Test Set**: 17 invalid rows
- **Total Invalid**: 129 rows

## Statement Length Statistics

### Train Set
- **Mean**: 106.40 characters
- **Median**: 99.00 characters
- **Min**: 11 characters
- **Max**: 395 characters

### Validation Set
- **Mean**: 106.78 characters
- **Median**: 99.00 characters
- **Min**: 17 characters
- **Max**: 327 characters

### Test Set
- **Mean**: 106.86 characters
- **Median**: 99.00 characters
- **Min**: 12 characters
- **Max**: 289 characters

## Sample Statements by Label

### barely-true
"Jim Dunnam has not lived in the district he represents for years now."

### false
"Says the Annies List political group supports third-trimester abortions on demand."

### half-true
"When did the decline of coal start? It started when natural gas took off that started to begin in (President George W.) Bushs administration."

### mostly-true
"Hillary Clinton agrees with John McCain 'by voting to give George Bush the benefit of the doubt on Iran.'"

### pants-fire
"In the case of a catastrophic event, the Atlanta-area offices of the Centers for Disease Control and Prevention will be unable to communicate with state and local agencies."

### true
"The Chicago Bears have had more starting quarterbacks in the last 10 years than the Green Bay Packers."

## Key Observations

1. **Balanced Dataset**: The label distribution is relatively balanced, with each label ranging from 8-21% of the dataset.

2. **Smallest Class**: `pants-fire` is the smallest class (8.15%), which may require special handling during training.

3. **Missing Data**: Significant missing values in `speaker_job` (27.8%) and `state_info` (21.4%) columns.

4. **Statement Length**: Statements are relatively short (mean ~106 characters), suitable for NLP analysis.

5. **Original Splits**: The dataset comes with pre-defined train/validation/test splits that should be preserved.

6. **Multimodal Features**: The dataset includes metadata (speaker, party, context) that can be used for multimodal analysis.

## Files Created/Modified

### Created
- `data/raw/train.tsv` - Training dataset
- `data/raw/valid.tsv` - Validation dataset
- `data/raw/test.tsv` - Test dataset
- `data/raw/README` - Dataset documentation
- `notebooks/01_liar_eda.ipynb` - Jupyter notebook for EDA
- `scripts/analyze_liar_dataset.py` - Analysis script
- `docs/liar_dataset_summary.md` - This summary document

### No Errors Encountered
The dataset download and analysis completed successfully without errors.
