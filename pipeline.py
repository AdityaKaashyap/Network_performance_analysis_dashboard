import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# ---------------- 1. LOAD & CLEAN ----------------
def load_and_clean(json_file):
    df = pd.read_json(json_file)
    df = pd.json_normalize(df.to_dict(orient="records"), sep="_")
    
    # Standardize network type
    df["network_type"] = df["network_type"].str.upper().str.strip()
    df["network_type"] = df["network_type"].replace({
        "LTe": "LTE", "FIVEG": "5G", "4G+": "4G"
    })
    
    # Handle missing values
    df.fillna({
        "latency_sec": df["latency_sec"].median(),
        "download_speed_mbps": df["download_speed_mbps"].median(),
        "tower_load_percent": df["tower_load_percent"].median(),
        "dropped_calls": 0,
        "packet_loss_percent": 0
    }, inplace=True)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# ---------------- 2. FEATURE SCALING ----------------
def scale_features(df):
    features = ["latency_sec","download_speed_mbps","tower_load_percent",
                "dropped_calls","packet_loss_percent"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return df, X_scaled, features

# ---------------- 3. CLUSTERING ----------------
def cluster_towers(df, X_scaled, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    return df

# ---------------- 4. ANOMALY DETECTION ----------------
def detect_anomalies(df, X_scaled):
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = iso.fit_predict(X_scaled)
    return df

# ---------------- 5. PCA FOR VISUALIZATION ----------------
def add_pca(df, X_scaled):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df["pca1"] = X_pca[:,0]
    df["pca2"] = X_pca[:,1]
    return df

# ---------------- 6. RECOMMENDATIONS ----------------
def generate_recommendation(row):
    recs = []
    if row["tower_load_percent"] > 85 and row["download_speed_mbps"] < 20:
        recs.append("⚠️ Tower overloaded → Add capacity / load balancing")
    if row["latency_sec"] > 0.7:
        recs.append("⚡ High latency → Check backhaul/core network")
    if row["packet_loss_percent"] > 2:
        recs.append("📞 Packet loss high → Optimize VoIP routing/codecs")
    if "voip_metrics_jitter_ms" in row and row["voip_metrics_jitter_ms"] > 30:
        recs.append("📞 High jitter → Review routing paths / bandwidth")
    if row.get("maintenance_due", False) and row.get("tower_age_years", 0) > 10:
        recs.append("🛠️ Old tower + maintenance due → Replace/upgrade tower")
    if row["anomaly"] == -1:
        recs.append("🚨 Anomaly detected → Send field engineer")
    return "; ".join(recs) if recs else "✅ Tower healthy"

def add_recommendations(df):
    df["recommendation"] = df.apply(generate_recommendation, axis=1)
    return df

# ---------------- 7. RUN PIPELINE ----------------
def run_pipeline(json_file, output_csv="telecom_processed.csv"):
    df = load_and_clean(json_file)
    df, X_scaled, features = scale_features(df)
    df = cluster_towers(df, X_scaled)
    df = detect_anomalies(df, X_scaled)
    df = add_pca(df, X_scaled)
    df = add_recommendations(df)
    
    df.to_csv(output_csv, index=False)
    print(f"[✅] Pipeline complete. Processed data saved as {output_csv}")
    return df

if __name__ == "__main__":
    run_pipeline("telecom_tower_usaged.json")
