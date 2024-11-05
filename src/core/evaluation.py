def calculate_metrics(interactions_labels, prediction_col):
    precision_at_10 = interactions_labels.apply(
        lambda row: len(list(set(row["contentId"]).intersection(row[prediction_col])))
        / len(row[prediction_col]),
        axis=1,
    ).mean()
    recall_at_10 = interactions_labels.apply(
        lambda row: len(list(set(row["contentId"]).intersection(row[prediction_col])))
        / len(row["contentId"]),
        axis=1,
    ).mean()
    return {"precision_at_10": precision_at_10, "recall_at_10": recall_at_10}
