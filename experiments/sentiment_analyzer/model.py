from transformers import DistilBertForSequenceClassification, DistilBertConfig

def get_model(num_labels=2):
    """
    Initializes a DistilBERT model for sequence classification.
    Default is 2 labels (Negative, Positive) for IMDB.
    """
    config = DistilBertConfig.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        config=config
    )
    return model
