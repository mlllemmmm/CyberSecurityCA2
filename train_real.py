import pandas as pd
import numpy as np
import json, time

df = pd.read_csv('/mnt/user-data/uploads/cicddos2019_sample.csv')
df.columns = [c.strip() for c in df.columns]

# --- Map the real CICDDoS2019 labels onto the case study's 4-class scheme ---
def map_label(l):
    if l == 'BENIGN': return 'Benign'
    if l == 'Syn': return 'SYN_Flood'
    if l == 'DrDoS_UDP': return 'UDP_Flood'
    return 'DrDoS'  # DNS, LDAP, MSSQL, NetBIOS, NTP, SNMP, SSDP, TFTP reflection/amplification family

df['class'] = df['Label'].apply(map_label)
print(df['class'].value_counts())
print("\nOriginal label breakdown within 'DrDoS' bucket:")
print(df[df['class']=='DrDoS']['Label'].value_counts())

# --- Balance classes: the merged DrDoS bucket (8 reflection sub-types) naturally
# has ~8x more rows than the other three classes. Rather than let that swamp
# training, downsample it to match -- the same class-imbalance handling step
# described in the case study's preprocessing section (3.2). ---
min_count = df['class'].value_counts().min()
balanced_parts = [g.sample(n=min_count, random_state=42) for _, g in df.groupby('class')]
df = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)
print("\nAfter balancing:\n", df['class'].value_counts())

# --- Feature engineering: 6 features, matching what the live browser demo uses ---
total_pkts = (df['Total Fwd Packets'] + df['Total Backward Packets']).replace(0, np.nan)
feat = pd.DataFrame({
    'packet_rate': df['Flow Packets/s'],
    'byte_rate_kbs': df['Flow Bytes/s'] / 1024.0,
    'flow_duration_s': df['Flow Duration'] / 1e6,
    'avg_packet_size': df['Average Packet Size'],
    'syn_flag_ratio': (df['SYN Flag Count'] / total_pkts).fillna(0),
    'down_up_ratio': df['Down/Up Ratio'],
})

feat = feat.replace([np.inf, -np.inf], np.nan)
# Cap extreme outliers (Flow Bytes/Packets per second can spike to billions on 0-duration flows)
for c in feat.columns:
    cap = feat[c].quantile(0.995)
    feat[c] = feat[c].clip(upper=cap)
feat = feat.fillna(0)
feat = feat.clip(lower=0)

y_labels = df['class'].values
classes = ["Benign", "SYN_Flood", "UDP_Flood", "DrDoS"]
y = np.array([classes.index(c) for c in y_labels])
X = feat.values

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(y)), test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Decision Tree: the one embedded in the live browser demo. A tree's splits
# are just threshold checks, so it runs natively in JS without a softmax/matrix
# step, and at depth 8 it's one of the strongest real performers here too. ---
from sklearn.tree import DecisionTreeClassifier as _DTC
tree_clf = _DTC(max_depth=8, random_state=42)
tree_clf.fit(X_train, y_train)  # trees don't need scaled input
pred = tree_clf.predict(X_test)
acc = accuracy_score(y_test, pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, labels=[0,1,2,3])
cm = confusion_matrix(y_test, pred, labels=[0,1,2,3])

t = tree_clf.tree_
class_id = [int(np.argmax(v[0])) for v in t.value]
model_out = {
    "classes": classes,
    "feature_names": ["packet_rate", "byte_rate_kbs", "flow_duration_s", "avg_packet_size", "syn_flag_ratio", "down_up_ratio"],
    "type": "decision_tree",
    "feature": t.feature.tolist(),
    "threshold": t.threshold.tolist(),
    "children_left": t.children_left.tolist(),
    "children_right": t.children_right.tolist(),
    "class_id": class_id,
    "node_confidence": [round(float(np.max(v[0])/np.sum(v[0])), 4) for v in t.value],
    "accuracy": round(float(acc), 4),
    "precision": [round(float(x), 4) for x in prec],
    "recall": [round(float(x), 4) for x in rec],
    "f1": [round(float(x), 4) for x in f1],
    "confusion_matrix": cm.tolist(),
    "test_size": int(len(y_test)),
    "train_size": int(len(y_train)),
    "node_count": int(t.node_count),
    "source": "CICDDoS2019 sample (44,000 real flows, user-provided; class-balanced to 16,000)",
}
with open('/home/claude/model_real.json', 'w') as f:
    json.dump(model_out, f)
print("\nDecision Tree (live demo model):", model_out['accuracy'], "nodes:", model_out['node_count'])

# --- Full comparison across all 6 classifiers, same split ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, C=1.0),
    "Decision Tree": tree_clf,
    "Naive Bayes": GaussianNB(),
    "SVM (RBF)": SVC(kernel="rbf", C=2.0, gamma="scale"),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.2, eval_metric="mlogloss", random_state=42),
}
comparison = []
for name, m in models.items():
    if name != "Decision Tree":
        t0 = time.time()
        m.fit(X_train_s, y_train)
        train_time = time.time()-t0
    else:
        train_time = 0.0
    t0 = time.time()
    p = m.predict(X_test if name == "Decision Tree" else X_test_s)
    infer_ms = (time.time()-t0)/len(X_test)*1000
    a = accuracy_score(y_test, p)
    pr, rc, f, _ = precision_recall_fscore_support(y_test, p, labels=[0,1,2,3], average="macro")
    comparison.append({"name": name, "accuracy": round(float(a),4), "precision": round(float(pr),4),
                        "recall": round(float(rc),4), "f1": round(float(f),4),
                        "train_time_s": round(train_time,3), "infer_time_ms": round(infer_ms,4)})
comparison.sort(key=lambda d: -d["accuracy"])
with open('/home/claude/comparison_real.json', 'w') as f:
    json.dump(comparison, f, indent=2)
print("\n--- Comparison (real CICDDoS2019 sample) ---")
for c in comparison: print(c)

# --- Sample real flows per class for the live browser demo to replay ---
# (unscaled, human-readable feature values straight from the dataset)
pool = {}
for i, cname in enumerate(classes):
    rows = feat.iloc[idx_test][y[idx_test]==i]
    n = min(60, len(rows))
    sample = rows.sample(n=n, random_state=1).round(4)
    pool[cname] = sample.values.tolist()
with open('/home/claude/flow_pool.json', 'w') as f:
    json.dump(pool, f)
print("\nPool sizes:", {k: len(v) for k,v in pool.items()})
