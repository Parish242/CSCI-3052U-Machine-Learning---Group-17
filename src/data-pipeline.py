from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from collections import Counter


def get_grouped_splits():
    DATASET_ID = "akhilayerukola/NormAd"

    raw = load_dataset(DATASET_ID, split="train").to_pandas()

    # Cleaning
    missing = [c for c in ["Country", "Rule-of-Thumb", "Story", "Gold Label"] if c not in raw.columns]
    assert not missing, f"Missing columns {missing}"
    data = pd.DataFrame({
        "ID": raw["ID"],
        "Country": raw["Country"],
        "Background": raw["Background"],
        "Rule-of-Thumb": raw["Rule-of-Thumb"],
        "Story": raw["Story"],
        "Gold Label": raw["Gold Label"],
    })
    assert data["ID"].is_unique

    # Map neutral -> neither
    LABELS = ["yes", "no", "neither"]
    LABEL_MAP = {"yes": "yes", "no": "no", "neither": "neither", "neutral": "neither"}
    gold_lower = data["Gold Label"].astype(str).str.strip().str.lower()
    unknown = sorted(set(gold_lower) - set(LABEL_MAP))
    assert not unknown, f"Unexpected gold labels {unknown}"
    data["gold_label"] = gold_lower.map(LABEL_MAP)
    raw_label_counts = Counter(gold_lower)

    # Country capitalized mapping
    COUNTRY_NAMES = {
        'malaysia': 'Malaysia', 'lebanon': 'Lebanon', 'colombia': 'Colombia',
        'romania': 'Romania', 'south_africa': 'South Africa', 'myanmar': 'Myanmar',
        'saudi_arabia': 'Saudi Arabia', 'samoa': 'Samoa', 'sudan': 'Sudan',
        'united_kingdom': 'United Kingdom', 'australia': 'Australia', 'taiwan': 'Taiwan',
        'thailand': 'Thailand', 'pakistan': 'Pakistan', 'philippines': 'Philippines',
        'sri_lanka': 'Sri Lanka', 'north_macedonia': 'North Macedonia', 'germany': 'Germany',
        'cambodia': 'Cambodia', 'türkiye': 'Türkiye', 'zimbabwe': 'Zimbabwe',
        'fiji': 'Fiji', 'hungary': 'Hungary', 'mexico': 'Mexico',
        'brazil': 'Brazil', 'palestinian_territories': 'Palestinian Territories', 'iraq': 'Iraq',
        'poland': 'Poland', 'sweden': 'Sweden', 'ukraine': 'Ukraine',
        'bangladesh': 'Bangladesh', 'new_zealand': 'New Zealand', 'france': 'France',
        'ethiopia': 'Ethiopia', 'somalia': 'Somalia', 'portugal': 'Portugal',
        'croatia': 'Croatia', 'united_states_of_america': 'United States of America', 'egypt': 'Egypt',
        'china': 'China', 'mauritius': 'Mauritius', 'cyprus': 'Cyprus',
        'greece': 'Greece', 'italy': 'Italy', 'south_sudan': 'South Sudan',
        'venezuela': 'Venezuela', 'iran': 'Iran', 'hong_kong': 'Hong Kong',
        'timor-leste': 'Timor-Leste', 'austria': 'Austria', 'india': 'India',
        'israel': 'Israel', 'nepal': 'Nepal', 'bosnia_and_herzegovina': 'Bosnia and Herzegovina',
        'malta': 'Malta', 'ireland': 'Ireland', 'peru': 'Peru',
        'vietnam': 'Vietnam', 'singapore': 'Singapore', 'papua_new_guinea': 'Papua New Guinea',
        'japan': 'Japan', 'indonesia': 'Indonesia', 'netherlands': 'Netherlands',
        'kenya': 'Kenya', 'argentina': 'Argentina', 'chile': 'Chile',
        'serbia': 'Serbia', 'afghanistan': 'Afghanistan', 'canada': 'Canada',
        'spain': 'Spain', 'syria': 'Syria', 'laos': 'Laos',
        'tonga': 'Tonga', 'russia': 'Russia', 'south_korea': 'South Korea',
    }
    unmapped = sorted(set(data["Country"]) - set(COUNTRY_NAMES))
    for c in unmapped:
        COUNTRY_NAMES[c] = c.replace("_", " ").title()
    data["country_name"] = data["Country"].map(COUNTRY_NAMES)

    # Hold out 20% for the test set grouped by Background
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_val_idx, test_idx = next(gss.split(data, groups=data['Background']))

    train_val_data = data.iloc[train_val_idx].reset_index(drop=True)
    test_data = data.iloc[test_idx].reset_index(drop=True)

    # 5-Fold StratifiedGroupKFold on the train/val set for hyperparameter tuning
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(sgkf.split(train_val_data, y=train_val_data['gold_label'], groups=train_val_data['Background']))

    return train_val_data, test_data, folds

if __name__ == "__main__":
    train_val, test, cv_folds = get_grouped_splits()
    print(f"Train/val size: {len(train_val)} | Test size: {len(test)}")
    print(f"Number of CV folds: {len(cv_folds)}")