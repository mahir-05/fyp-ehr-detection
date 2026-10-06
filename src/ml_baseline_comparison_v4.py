from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import warnings
warnings.filterwarnings('ignore')

# Database connection
engine = create_engine(
    "mysql+mysqlconnector://openemr:openemrpass@localhost:3306/openemr"
)

def load_labeled_data():
    """Pulls only the entries I generated with the simulation script.
    They all have 'SIMULATED|' in the comments field so it's easy to filter out
    the real OpenEMR records. Also parses out the ground truth labels I embedded
    as JSON so I can evaluate the detection later."""
    print("=" * 80)
    print("ML Version 4")
    print("=" * 80)
    print("\n[1/9] Loading LABELED audit data...")

    query = """
        SELECT id, date, event, user, patient_id, success, comments
        FROM log WHERE comments LIKE 'SIMULATED|%%' ORDER BY date
    """

    df = pd.read_sql(query, engine)
    print(f"    Loaded {len(df)} labeled entries")

    def parse_ground_truth(comment):
        try:
            if comment and 'SIMULATED|' in comment:
                return json.loads(comment.replace('SIMULATED|', ''))
        except:
            pass
        return {'is_malicious': False, 'threat_pattern': None}

    df['ground_truth'] = df['comments'].apply(parse_ground_truth)
    df['is_malicious'] = df['ground_truth'].apply(lambda x: x.get('is_malicious', False))
    df['date'] = pd.to_datetime(df['date'])

    malicious = df['is_malicious'].sum()
    print(f"    Malicious: {malicious} | Legitimate: {len(df)-malicious} | Ratio: {malicious/len(df)*100:.1f}%")

    return df

def load_patients():
    return pd.read_sql("SELECT pid, fname, lname, postal_code, street FROM patient_data WHERE pid > 0", engine)

def load_users():
    return pd.read_sql("SELECT id, username, fname, lname, title FROM users WHERE username != ''", engine)

def extract_features(df, patients_df, users_df):
    """Build the feature set for the ML models. I tried to mirror the same signals
    that the detection rules use things like time of access, who the user is,
    how many patients they accessed per hour, surname/postcode clustering etc.
    The idea is that if the ML models pick up on the same features, it validates
    that those patterns are genuinely predictive."""
    print("\n[2/9] Extracting features...")

    features = pd.DataFrame(index=df.index)

    surname_lookup = patients_df.set_index('pid')['lname'].to_dict()
    postal_lookup = patients_df.set_index('pid')['postal_code'].to_dict()
    role_lookup = users_df.set_index('username')['title'].to_dict()

    # time-based features - the after-hours rule relies on this heavily
    features['hour'] = df['date'].dt.hour
    features['is_after_hours'] = ((df['date'].dt.hour < 6) | (df['date'].dt.hour >= 22)).astype(int)
    features['is_night'] = ((df['date'].dt.hour >= 22) | (df['date'].dt.hour <= 5)).astype(int)
    features['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    features['day_of_week'] = df['date'].dt.dayofweek

    # who the user is and what their role is - important for the cross-department rule
    features['user_encoded'] = LabelEncoder().fit_transform(df['user'].fillna('unknown'))
    user_role = df['user'].map(role_lookup).fillna('unknown')
    features['is_receptionist'] = user_role.str.lower().str.contains('receptionist', na=False).astype(int)

    # what type of event it was - whether it's something clinical that admin staff shouldn't need
    features['event_encoded'] = LabelEncoder().fit_transform(df['event'].fillna('unknown'))
    clinical_kw = ['medication', 'diagnosis', 'lab', 'clinical', 'notes']
    features['is_clinical_event'] = df['event'].fillna('').str.lower().apply(
        lambda x: any(kw in x for kw in clinical_kw)).astype(int)

    # how many patients the user accessed - feeds into the high-volume detection
    user_counts = df.groupby('user').size()
    features['user_total_accesses'] = df['user'].map(user_counts)

    df_temp = df.copy()
    df_temp['hour_bucket'] = df_temp['date'].dt.floor('h')
    hourly = df_temp.groupby(['user', 'hour_bucket']).size().reset_index(name='hourly_count')
    df_temp = df_temp.merge(hourly, on=['user', 'hour_bucket'], how='left')
    features['hourly_access_count'] = df_temp['hourly_count'].values

    # surname clustering - how many same-surname patients did this user access
    df_temp['surname'] = df['patient_id'].map(surname_lookup)
    surname_counts = df_temp.groupby(['user', 'surname']).size().reset_index(name='surname_count')
    df_temp = df_temp.merge(surname_counts, on=['user', 'surname'], how='left')
    features['same_surname_count'] = df_temp['surname_count'].fillna(1).values

    # postcode clustering - same idea as surname but based on where patients live
    df_temp['postal'] = df['patient_id'].map(postal_lookup)
    postal_counts = df_temp.groupby(['user', 'postal']).size().reset_index(name='postal_count')
    df_temp = df_temp.merge(postal_counts, on=['user', 'postal'], how='left')
    features['same_postal_count'] = df_temp['postal_count'].fillna(1).values

    # whether the access was denied - repeated failures can indicate probing
    features['is_denied_event'] = df['event'].fillna('').str.lower().str.contains('denied|failed', na=False).astype(int)

    feature_names = list(features.columns)
    print(f"    Extracted {len(feature_names)} features")

    return features, feature_names

def run_cross_validation(X, y, feature_names):
    """5-fold stratified CV to get reliable results rather than relying on a single
    train/test split. Stratified means each fold has the same class ratio which matters
    here since malicious entries are a minority. SVM and LR need scaled data, the
    tree-based ones don't."""
    print("\n[3/9] Running 5-Fold Cross-Validation...")
    print("-" * 80)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # tried n_estimators=200 for RF but runtime was noticeably slower with minimal gain
    # max_depth=10 for RF and 5 for GB came from a quick grid search - deeper trees overfit on this dataset
    # SVM is slow but included as a standard baseline for the comparison table
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
    }

    cv_results = {}

    for name, model in models.items():
        print(f"    {name}...", end=" ")

        if name in ['SVM (RBF)', 'Logistic Regression']:
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
        else:
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1')

        cv_results[name] = {
            'mean': scores.mean() * 100,
            'std': scores.std() * 100,
            'scores': scores * 100
        }
        print(f"F1 = {scores.mean()*100:.2f}% (+/-{scores.std()*100:.2f}%)")

    return cv_results, scaler

def train_final_models(X, y, scaler, feature_names):
    """Trains each model on the 80/20 split so I can get the full confusion matrix,
    AUC scores, and ROC curve data for the dissertation charts. I kept the same
    random_state=42 throughout so the results are reproducible."""
    print("\n[4/9] Training final models (80/20 split)...")
    print("-" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Random Forest': (RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42), False),
        'Gradient Boosting': (GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42), False),
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
        'SVM (RBF)': (SVC(kernel='rbf', random_state=42, probability=True), True),
    }

    results = []
    trained_models = {}
    roc_data = {}

    for name, (model, use_scaled) in models.items():
        if use_scaled:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        trained_models[name] = model

        # calculate all the metrics I need for the comparison table
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, zero_division=0) * 100
        rec = recall_score(y_test, y_pred, zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, zero_division=0) * 100
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # save the ROC curve data so I can plot everything on one chart later
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

        results.append({
            'Model': name, 'Type': 'Supervised',
            'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1,
            'AUC': roc_auc * 100, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
        })

        print(f"    {name}: F1={f1:.2f}%, AUC={roc_auc*100:.2f}%")

    return results, trained_models, roc_data, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

def train_isolation_forest(X_train_scaled, X_test_scaled, y_train, y_test):
    """This is the Tabassum et al. (2019) comparison model - they used Isolation Forest
    for insider threat detection. I set the contamination rate to match the actual
    proportion of malicious entries in my dataset. Predictions of -1 mean anomaly."""
    print("\n[5/9] Training Isolation Forest (unsupervised)...")

    # setting contamination to the actual malicious rate rather than the default 0.1
    # the default kept giving weird results because my dataset has a different ratio
    contamination = y_train.mean()
    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    iso.fit(X_train_scaled)

    y_pred = (iso.predict(X_test_scaled) == -1).astype(int)

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec = recall_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"    Isolation Forest: F1={f1:.2f}% (unsupervised)")

    return {
        'Model': 'Isolation Forest', 'Type': 'Unsupervised',
        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1,
        'AUC': 0, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
    }

def analyze_feature_importance(trained_models, feature_names):
    """Check which features each model actually ended up relying on. This is one of
    the more interesting parts - if the ML models are learning the same signals as
    my detection rules, that's evidence the rules are capturing real patterns rather
    than just overfitting to my simulation setup. For LR I use absolute coefficient
    values since direction doesn't tell us importance on its own."""
    print("\n[6/9] Analyzing Feature Importance...")
    print("-" * 80)

    importance_data = {}

    # random forest gives you feature importances directly from the trees
    rf = trained_models['Random Forest']
    rf_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    importance_data['Random Forest'] = rf_importance

    print("\n    RANDOM FOREST - Top Features:")
    for i, row in rf_importance.head(7).iterrows():
        bar = "#" * int(row['Importance'] * 50)
        print(f"    {row['Feature']:<25} {row['Importance']:.3f} {bar}")

    # gradient boosting also has built-in feature importances
    gb = trained_models['Gradient Boosting']
    gb_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': gb.feature_importances_
    }).sort_values('Importance', ascending=False)
    importance_data['Gradient Boosting'] = gb_importance

    print("\n    GRADIENT BOOSTING - Top Features:")
    for i, row in gb_importance.head(7).iterrows():
        bar = "#" * int(row['Importance'] * 50)
        print(f"    {row['Feature']:<25} {row['Importance']:.3f} {bar}")

    # for logistic regression I take the absolute value of coefficients as a proxy for importance
    lr = trained_models['Logistic Regression']
    lr_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(lr.coef_[0])
    }).sort_values('Importance', ascending=False)
    importance_data['Logistic Regression'] = lr_importance

    print("\n    LOGISTIC REGRESSION - Top Features (by coefficient magnitude):")
    for i, row in lr_importance.head(7).iterrows():
        bar = "#" * int(row['Importance'] * 5)
        print(f"    {row['Feature']:<25} {row['Importance']:.3f} {bar}")

    return importance_data

def create_roc_curves(roc_data):
    """Plots all the ROC curves on one figure for easy comparison. I also added my
    rule-based V12 result as a single point (since it doesn't produce a full curve)
    so I can show where it sits relative to the ML models."""
    print("\n[7/9] Generating ROC Curves...")

    plt.figure(figsize=(10, 8))

    colors = {'Random Forest': '#2ecc71', 'Gradient Boosting': '#3498db',
              'Logistic Regression': '#e74c3c', 'SVM (RBF)': '#9b59b6'}

    for name, data in roc_data.items():
        plt.plot(data['fpr'], data['tpr'],
                label=f"{name} (AUC = {data['auc']:.3f})",
                color=colors.get(name, 'gray'), linewidth=2)

    # plot V12 as a single star marker - it has one fixed operating point, not a curve
    # coordinates taken from the actual V12 evaluation results (FPR=0.012, TPR=0.901)
    plt.scatter([0.012], [0.901], marker='*', s=400, c='gold',
                edgecolors='black', linewidths=1.5, zorder=5,
                label='Rule-Based V12 (F1=92.22%)')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curve Comparison: ML Models vs Rule-Based Detection', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/roc_curves_comparison.png', dpi=150, bbox_inches='tight')
    print("    Saved: outputs/roc_curves_comparison.png")

    return 'outputs/roc_curves_comparison.png'

def create_feature_importance_chart(importance_data):
    """Side-by-side bar charts for RF and GB feature importances. Kept it to top 10
    so the chart doesn't get too cramped."""
    print("\n[8/9] Generating Feature Importance Chart...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # left panel: random forest
    rf_data = importance_data['Random Forest'].head(10)
    axes[0].barh(rf_data['Feature'], rf_data['Importance'], color='#2ecc71')
    axes[0].set_xlabel('Importance Score')
    axes[0].set_title('Random Forest - Feature Importance')
    axes[0].invert_yaxis()

    # right panel: gradient boosting
    gb_data = importance_data['Gradient Boosting'].head(10)
    axes[1].barh(gb_data['Feature'], gb_data['Importance'], color='#3498db')
    axes[1].set_xlabel('Importance Score')
    axes[1].set_title('Gradient Boosting - Feature Importance')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('outputs/feature_importance_comparison.png', dpi=150, bbox_inches='tight')
    print("    Saved: outputs/feature_importance_comparison.png")

    return 'outputs/feature_importance_comparison.png'

def load_rule_based_results(filepath="outputs/rule_based_results.json"):
    """Loads the V12 rule-based metrics saved by detection_engine_v12.py.
    Run the detection engine first to generate this file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"{filepath} not found - run detection_engine_v12.py first to generate it."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    required_keys = {"Model", "Type", "Accuracy", "Precision", "Recall", "F1-Score", "AUC", "TP", "FP", "TN", "FN"}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"Missing keys in {filepath}: {sorted(missing)}")
    return data


def print_final_results(results, cv_results, total_entries):
    """Prints the comparison tables only, without the dissertation commentary block."""

    print("\n" + "=" * 80)
    print("[9/9] FINAL RESULTS")
    print("=" * 80)

    # load V12 results from the detection engine output rather than hardcoding them
    v12 = load_rule_based_results()

    all_results = [v12] + results

    print(f"\nDataset: {total_entries} labeled synthetic entries")
    print("Methodology: 5-Fold Stratified Cross-Validation + 80/20 Test Split")

    # cross-validation results
    print("\n" + "-" * 80)
    print("CROSS-VALIDATION RESULTS (5-Fold, F1-Score)")
    print("-" * 80)
    print(f"{'Model':<25} {'Mean F1':<12} {'Std Dev':<12} {'95% CI':<20}")
    print("-" * 80)
    for name, data in cv_results.items():
        ci_low = data['mean'] - 1.96 * data['std']
        ci_high = data['mean'] + 1.96 * data['std']
        print(f"{name:<25} {data['mean']:.2f}%{'':<5} +/-{data['std']:.2f}%{'':<4} [{ci_low:.2f}% - {ci_high:.2f}%]")
    print("-" * 80)

    # holdout test set results
    print("\n" + "-" * 80)
    print("TEST SET RESULTS (20% holdout)")
    print("-" * 80)
    print(f"{'Model':<22} {'Type':<12} {'Acc':<9} {'Prec':<9} {'Rec':<9} {'F1':<9} {'AUC':<9}")
    print("-" * 80)

    for r in all_results:
        marker = " [YOURS]" if r['Model'] == 'Rule-Based (V12)' else ""
        auc_str = f"{r['AUC']:.1f}%" if r['AUC'] > 0 else "N/A"
        print(f"{r['Model']:<22} {r['Type']:<12} {r['Accuracy']:.1f}%{'':<2} {r['Precision']:.1f}%{'':<2} {r['Recall']:.1f}%{'':<2} {r['F1-Score']:.1f}%{'':<2} {auc_str:<8}{marker}")

    print("-" * 80)
    print("[YOURS] = Rule-based evaluated on full dataset")

    # confusion matrices
    print("\n" + "-" * 80)
    print("CONFUSION MATRICES")
    print("-" * 80)
    print(f"{'Model':<25} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['Model']:<25} {r['TP']:<8} {r['FP']:<8} {r['TN']:<8} {r['FN']:<8}")

    return all_results

def main():
    """Runs everything in order - load data, extract features, cross-validate,
    train models, generate charts, print results. Wrapped in try/except so if
    something goes wrong with the DB connection it fails gracefully."""
    try:
        # step 1: pull the labeled simulation data
        df = load_labeled_data()
        if len(df) == 0:
            print("ERROR: No SIMULATED entries found!")
            return

        patients_df = load_patients()
        users_df = load_users()

        # build features from the audit log + patient/user tables
        X, feature_names = extract_features(df, patients_df, users_df)
        y = df['is_malicious'].astype(int)
        X = X.fillna(0)

        # cross-validate first to get reliable F1 estimates
        cv_results, scaler = run_cross_validation(X, y, feature_names)

        # then train on the 80/20 split to get full metrics + ROC data
        results, trained_models, roc_data, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = \
            train_final_models(X, y, scaler, feature_names)

        # add isolation forest as the unsupervised comparison
        iso_result = train_isolation_forest(X_train_scaled, X_test_scaled, y_train, y_test)
        results.append(iso_result)

        # check which features each model relied on most
        importance_data = analyze_feature_importance(trained_models, feature_names)

        # generate the dissertation figures
        roc_file = create_roc_curves(roc_data)
        importance_file = create_feature_importance_chart(importance_data)

        # print everything and return the results
        all_results = print_final_results(results, cv_results, len(df))

        # dump results to CSV so I can reference the numbers later
        pd.DataFrame(all_results).to_csv('outputs/ml_comparison_results_v4.csv', index=False)
        print(f"\nResults saved to: outputs/ml_comparison_results_v4.csv")

        # also save the per-fold CV breakdown
        cv_df = pd.DataFrame([{
            'Model': name,
            'Mean_F1': data['mean'],
            'Std_F1': data['std'],
            'Fold_1': data['scores'][0],
            'Fold_2': data['scores'][1],
            'Fold_3': data['scores'][2],
            'Fold_4': data['scores'][3],
            'Fold_5': data['scores'][4],
        } for name, data in cv_results.items()])
        cv_df.to_csv('outputs/cross_validation_results.csv', index=False)
        print(f"Cross-validation saved to: outputs/cross_validation_results.csv")

        print("\nFiles generated:")
        print(" - outputs/ml_comparison_results_v4.csv")
        print(" - outputs/cross_validation_results.csv")
        print(" - outputs/roc_curves_comparison.png")
        print(" - outputs/feature_importance_comparison.png")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
