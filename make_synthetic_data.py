"""Generates a small synthetic dataset with the exact schema described in the
problem statement, purely to smoke-test that the pipeline runs end-to-end
without crashing. NOT representative of real data difficulty or scale."""
import os
import random
import pandas as pd

random.seed(0)
os.makedirs("dataset/train", exist_ok=True)
os.makedirs("dataset/test", exist_ok=True)

NAME_POOL = [
    "Sri Lakshmi Enterprises", "Acme", "Bharat Textiles", "Global Traders",
    "Sunrise Foods", "Metro Hardware", "Prime Logistics", "Coastal Exports",
    "Silver Star Industries", "Golden Gate Retail", "National Widgets",
    "Blue Ocean Shipping", "Green Valley Farms", "Royal Textiles",
    "United Motors", "Apex Solutions", "Delta Engineering", "Horizon Traders",
]
SUFFIXES = ["Pvt Ltd", "Ltd", "Corp", "Inc", "LLC", "Co"]
CITIES_INDIA = [("Coimbatore", "641001"), ("Chennai", "600002"), ("Madurai", "625001")]
CITIES_US = [("Springfield", "62701"), ("Austin", "73301"), ("Denver", "80014")]


def make_record(eid_prefix, i, name, country, matched_variant=False):
    if country == "India":
        city, pin = random.choice(CITIES_INDIA)
        addr = f"{random.randint(1,99)} Gandhi Road, {city}, {pin}"
    else:
        city, pin = random.choice(CITIES_US)
        addr = f"{random.randint(1,999)} Main St, {city}, {pin}"
    full_name = f"{name} {random.choice(SUFFIXES)}"
    if matched_variant:
        # introduce light noise: abbreviation swap, minor typo-ish change
        full_name = full_name.replace("Pvt Ltd", "Private Limited").replace("Ltd", "Limited").replace("Corp", "Corporation")
        addr = addr.replace("Road", "Rd").replace("St", "Street")
    return {"entity_id": f"{eid_prefix}-{i:05d}", "business_name": full_name,
            "business_address": addr, "country": country}

# --- TRAIN --- (40 S1 entities: ~70% have matches, ~30% singletons)
source1_rows, source2_rows, source3_rows, gt_rows = [], [], [], []
s2_counter, s3_counter = 1, 1

for i in range(1, 41):
    name = random.choice(NAME_POOL)
    country = random.choice(["India", "US"])
    s1_rec = make_record("S1", i, name, country)
    source1_rows.append(s1_rec)

    is_singleton = random.random() < 0.3
    matched_ids = []
    if not is_singleton:
        # 1-2 true matches, split across source2/source3
        n_matches = random.choice([1, 1, 2])
        for _ in range(n_matches):
            target_source = random.choice(["S2", "S3"])
            if target_source == "S2":
                rec = make_record("S2", s2_counter, name, country, matched_variant=True)
                source2_rows.append(rec)
                matched_ids.append(rec["entity_id"])
                s2_counter += 1
            else:
                rec = make_record("S3", s3_counter, name, country, matched_variant=True)
                source3_rows.append(rec)
                matched_ids.append(rec["entity_id"])
                s3_counter += 1
    gt_rows.append({"source1_entity_id": s1_rec["entity_id"], "matched_entity_ids": ",".join(matched_ids)})

# Add some unrelated noise records to source2/source3 (no match to any S1)
for _ in range(15):
    name = random.choice(NAME_POOL)
    country = random.choice(["India", "US"])
    source2_rows.append(make_record("S2", s2_counter, name, country))
    s2_counter += 1
    source3_rows.append(make_record("S3", s3_counter, name, country))
    s3_counter += 1

source1 = pd.DataFrame(source1_rows)
source2 = pd.DataFrame(source2_rows)
source3 = pd.DataFrame(source3_rows)
ground_truth = pd.DataFrame(gt_rows)

source1.to_csv("dataset/train/train_source1.tsv", sep="\t", index=False)
source2.to_csv("dataset/train/train_source2.tsv", sep="\t", index=False)
source3.to_csv("dataset/train/train_source3.tsv", sep="\t", index=False)
ground_truth.to_csv("dataset/train/train_ground_truth.tsv", sep="\t", index=False)

# --- TEST (reuse similar records with new IDs, incl. a France record) ---
test_source1 = pd.DataFrame([
    {"entity_id": "S1-10001", "business_name": "Sri Lakshmi Enterprises Pvt Ltd", "business_address": "12 Gandhi Road, Coimbatore, 641001", "country": "India"},
    {"entity_id": "S1-10002", "business_name": "Le Petit Boulanger SARL", "business_address": "10 Rue de Paris, 75001 Paris", "country": "France"},
])
test_source2 = pd.DataFrame([
    {"entity_id": "S2-10001", "business_name": "Sri Lakshmi Ent. Private Limited", "business_address": "12 Gandhi Rd, Coimbatore, 641001", "country": "India"},
    {"entity_id": "S2-10002", "business_name": "Le Petit Boulanger SARL", "business_address": "10 Rue de Paris, 75001 Paris", "country": "France"},
])
test_source3 = pd.DataFrame([
    {"entity_id": "S3-10001", "business_name": "Completely Unrelated Ltd", "business_address": "1 Nowhere St, Nowhere", "country": "US"},
])
test_source1.to_csv("dataset/test/test_source1.tsv", sep="\t", index=False)
test_source2.to_csv("dataset/test/test_source2.tsv", sep="\t", index=False)
test_source3.to_csv("dataset/test/test_source3.tsv", sep="\t", index=False)

print("Synthetic dataset written.")
