def calculate_metrics(interactions_labels, prediction_col):
    """
    Summary:
    This function calculates the precision at 10 and recall at 10 for a given set of interactions and predicted content.

    Args:
        interactions_labels (pd.DataFrame): A DataFrame containing the actual interactions, where each row contains a
                                             list of content IDs that the user has interacted with.
        prediction_col (str): The name of the column in the DataFrame that contains the predicted content IDs.

    Returns:
        dict: A dictionary containing the precision at 10 and recall at 10 metrics.
    """
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
