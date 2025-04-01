import os
import json
import pandas as pd
from scipy.stats import pearsonr
import logging

def compute_correlations(df_merged):
    """
    Given a merged DataFrame with performance and fairness metrics,
    compute Pearson correlations between each performance metric
    and each fairness metric.
    """

    # not gonna use time 
    performance_metrics = ["nDCG@10", "RR@10", "AP",]
    fairness_metrics = [
        "InterQuery.fairness_score",
        "LabelInversionRate",
        "IndividualFairnessViolation",
        "KendallsTauVsBaseline"
    ]
    
    results = []
    for p_col in performance_metrics:
        if p_col not in df_merged.columns or df_merged[p_col].dropna().empty:
            continue
        
        for f_col in fairness_metrics:
            if f_col not in df_merged.columns or df_merged[f_col].dropna().empty:
                continue
            
            df_valid = df_merged.dropna(subset=[p_col, f_col])
            if df_valid.empty:
                continue
            
            # Compute Pearson correlation
            corr, p_val = pearsonr(df_valid[p_col], df_valid[f_col])
            
            results.append({
                "PerformanceMetric": p_col,
                "FairnessMetric": f_col,
                "PearsonCorrelation": corr,
                "PValue": p_val
            })
    
    return pd.DataFrame(results)


def compute_pearson_coef():
    experiment_path = "experiments" 
    # for each dataset folder in experiment 
    for dataset_name in os.listdir(experiment_path):
        dataset_folder = os.path.join(experiment_path, dataset_name)
        
        if not os.path.isdir(dataset_folder):
            continue
        
        basic_eval_csv = os.path.join(dataset_folder, "basic_evaluations.csv")
        fairness_eval_json = os.path.join(dataset_folder, "fairness_evaluations.json")
        
        # display a warning if the files are missing, shouldn't be the case
        if not (os.path.isfile(basic_eval_csv) and os.path.isfile(fairness_eval_json)):
            logging.warning(f"Skipping {dataset_name}: missing CSV or JSON file.")
            continue
        
        logging.info(f"Processing dataset: {dataset_name}")
        
        df_perf = pd.read_csv(basic_eval_csv)  
        
        with open(fairness_eval_json, "r") as f:
            fairness_data = json.load(f)
        
        # flatten json to dataframe
        df_fair = pd.json_normalize(fairness_data, max_level=1)
        
        if 'name' not in df_perf.columns or 'model' not in df_fair.columns:
            logging.warning(f"Skipping {dataset_name}: 'name' or 'model' column missing.")
            continue
        
        df_merged = pd.merge(df_perf, df_fair, left_on='name', right_on='model', how='inner')
        
        if df_merged.empty:
            logging.warning(f"No matching models in CSV and JSON for {dataset_name}.")
            continue
        
        df_corr = compute_correlations(df_merged)
        
        # first save all correlation results 
        output_path = os.path.join(dataset_folder, "correlations.csv")
        df_corr.to_csv(output_path, index=False)
        
        # also save only relevant correlations, where p<0.05
        df_corr_significant = df_corr[df_corr["PValue"] < 0.05]
        output_path_significant = os.path.join(dataset_folder, "correlations_significant.csv")

        df_corr_significant.to_csv(output_path_significant, index=False)
