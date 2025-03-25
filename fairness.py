import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_df(ltr_model, test_queries, test_qrels):
  """
  Compute the dataframe for a given model. It will be used for the fairness evaluation.
  ltr_model = trained ltr model,
  - test_queries = pt.get_dataset("irds:nfcorpus/test/nontopic").get_topics()
  - test_qrels = pt.get_dataset("irds:nfcorpus/test/nontopic").get_qrels()
  - or both from pt.get_dataset("irds:nfcorpus/test/nontopic").get_topicsqrels()
  """

  res_ltr = ltr_model.transform(test_queries)

  combined_df = pd.merge(
    res_ltr,
    test_qrels[["qid", "docno", "label"]],
    on=["qid", "docno"],
    how="left"
  )

  combined_df.dropna()

  # combined_df.columns should be Index(['qid', 'docid', 'docno', 'title', 'abstract', 'url', 
  # 'score', 'query','features', 'rank', 'label'],
    
    
  return combined_df

def dcg_at_k(rels, k):
    """
    Compute Discounted Cumulative Gain (DCG) at rank k.
    """
    rels = np.asfarray(rels)[:k]
    if rels.size:
        return np.sum((2 ** rels - 1) / np.log2(np.arange(2, rels.size + 2)))
    return 0.0

def ndcg_at_k(rels, k=10):
    """
    Compute Normalized Discounted Cumulative Gain (nDCG) at rank k.
    """
    ideal_rels = sorted(rels, reverse=True)
    ideal_dcg = dcg_at_k(ideal_rels, k)
    actual_dcg = dcg_at_k(rels, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def compute_query_ndcg(df, k=10):
    """
    For each query in df, compute nDCG@k.
    Expects df to have columns: 'qid', 'score', and 'label'.
    Returns a dictionary mapping each qid to its nDCG@k.
    """
    ndcg_per_qid = {}
    for qid, group_df in df.groupby("qid"):
        group_df = group_df.sort_values("score", ascending=False)
        rels = group_df["label"].tolist()
        ndcg_per_qid[qid] = ndcg_at_k(rels, k)
    return ndcg_per_qid


def inter_query_fairness(df, k=10):
    """
    Computes inter-query fairness metrics:
      - Mean nDCG across queries
      - Standard deviation of nDCG
      - Range (max - min) of nDCG values
      - A fairness score defined as 1 - (std/mean) (clipped to [0,1])
    Input:
      df: DataFrame with columns 'qid', 'score', and 'label'
      k: Cutoff for nDCG calculation
    Returns:
      A dictionary with keys: mean_nDCG, std_nDCG, range_nDCG, fairness_score.
    """
    ndcg_scores = compute_query_ndcg(df, k)
    values = list(ndcg_scores.values())
    if len(values) == 0:
        return {"mean_nDCG": 0, "std_nDCG": 0, "range_nDCG": 0, "fairness_score": 0}
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    range_val = max(values) - min(values)
    fairness_score = max(0, 1 - (std_val / mean_val)) if mean_val > 0 else 0
    return {
        "mean_nDCG": round(mean_val, 4),
        "std_nDCG": round(std_val, 4),
        "range_nDCG": round(range_val, 4),
        "fairness_score": round(fairness_score, 4)
    }

def cosine_sim_func(textA, textB):
    """
    Compute cosine similarity between two texts using TF-IDF.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([textA, textB])
    sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return sim

def individual_fairness_violation(df, text_field, sim_threshold=0.8):
    """
    For each query, compares all document pairs. For each pair:
      - If similarity >= sim_threshold and the documents' rank difference is > 1,
        count it as a fairness violation.
    Inputs:
      df: DataFrame with columns 'qid', 'docno', 'score', 'abstract'. 
      sim_threshold: The similarity threshold above which two docs are considered "similar."
      text_field: The column name to use for text comparison.
    Returns:
      Average violation rate across all comparable pairs.
    """
    violations = []
    for qid, group_df in df.groupby("qid"):
        group_df = group_df.sort_values("score", ascending=False).reset_index(drop=True)
        # Create mappings from docno to rank and to text from the specified field.
        ranks = {row["docno"]: idx for idx, row in group_df.iterrows()}
        texts = {row["docno"]: row[text_field] for idx, row in group_df.iterrows()}
        docnos = group_df["docno"].tolist()
        # Create a mapping from docno to its rank (0-based)
        ranks = {docno: rank for rank, docno in enumerate(docnos)}
        n = len(docnos)
        for i in range(n):
            for j in range(i+1, n):
                dA, dB = docnos[i], docnos[j]
                rank_diff = abs(ranks[dA] - ranks[dB])
                sim = cosine_sim_func(texts[dA], texts[dB])
                # If similar but not adjacent (i.e., rank difference > 1), count as a violation
                if sim >= sim_threshold and rank_diff > 1:
                    violations.append(1)
                else:
                    violations.append(0)
    return np.mean(violations) if violations else 0.0


def label_inversion_rate(df):
    """
    For each query, computes the rate of label inversions in the ranking.
    A label inversion is a pair of documents where a document with a lower relevance label
    is ranked above a document with a higher relevance label.
    Input:
      df: DataFrame with columns 'qid', 'score', and 'label'
    Returns:
      Average inversion rate across queries.
    """
    inversion_rates = []
    for qid, group_df in df.groupby("qid"):
        group_df = group_df.sort_values("score", ascending=False).reset_index(drop=True)
        labels = group_df["label"].tolist()
        inversions = 0
        total_pairs = 0
        n = len(labels)
        for i in range(n):
            for j in range(i+1, n):
                total_pairs += 1
                if labels[i] < labels[j]:
                    inversions += 1
        if total_pairs > 0:
            inversion_rates.append(inversions / total_pairs)
    return np.mean(inversion_rates) if inversion_rates else 0.0

def compute_kendalls_tau(df_model, df_baseline):
    """
    Computes the average Kendall's Tau between the ranking orders of two DataFrames (model vs. baseline)
    for each query.
    Inputs:
      df_model: DataFrame containing the model's ranking results (must have 'qid', 'docno', 'score')
      df_baseline: DataFrame containing the baseline's ranking results (same required columns)
    Returns:
      The average Kendall's Tau across queries.
    """
    taus = []
    for qid, df_q_model in df_model.groupby("qid"):
        df_q_baseline = df_baseline[df_baseline["qid"] == qid]
        if df_q_baseline.empty:
            continue
        # Sort both by descending score
        df_q_model = df_q_model.sort_values("score", ascending=False).reset_index(drop=True)
        df_q_baseline = df_q_baseline.sort_values("score", ascending=False).reset_index(drop=True)
        # Create a mapping from docno to rank
        model_rank = {docno: rank for rank, docno in enumerate(df_q_model["docno"])}
        baseline_rank = {docno: rank for rank, docno in enumerate(df_q_baseline["docno"])}
        common_docnos = set(model_rank.keys()).intersection(baseline_rank.keys())
        if len(common_docnos) < 2:
            continue
        model_ranks = [model_rank[d] for d in common_docnos]
        baseline_ranks = [baseline_rank[d] for d in common_docnos]
        tau, _ = kendalltau(model_ranks, baseline_ranks)
        taus.append(tau)
    return np.mean(taus) if taus else 0.0



def fairness_evaluation(df_model, df_baseline=None, k=10, sim_threshold=0.8,  text_field="abstract"):
    """
    Evaluates fairness metrics for a ranking system.
    Inputs:
      - df_model: DataFrame with the model's ranking results (must contain 'qid', 'docno', 'score', 'label')
      - df_baseline: (Optional) DataFrame with baseline ranking results (for Kendall's Tau)
      - k: Cutoff for nDCG-related metrics
      - sim_threshold: Similarity threshold for considering documents as "similar"
      - text_field: The text column to use for similarity (e.g., "abstract" or "title")
    Returns:
      A dictionary with:
        - 'InterQuery': inter-query fairness metrics (mean_nDCG, std_nDCG, range_nDCG, fairness_score)
        - 'IndividualFairnessViolation': average violation rate
        - 'LabelInversionRate': average label inversion rate
        - 'KendallsTauVsBaseline': average Kendall's Tau
    """
    inter_query = inter_query_fairness(df_model, k)
    indiv_fair = individual_fairness_violation(df_model, text_field, sim_threshold) 
    inversion = label_inversion_rate(df_model)
    ktau = compute_kendalls_tau(df_model, df_baseline) if df_baseline is not None else None

    result = {
        "InterQuery": inter_query,
        "LabelInversionRate": round(inversion, 4),
        "IndividualFairnessViolation" : round(indiv_fair, 4)
    }
    if ktau is not None:
        result["KendallsTauVsBaseline"] = round(ktau, 4)
    return result