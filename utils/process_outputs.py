def process_outputs(task, outputs, labels=None):
    """
    Take raw tensor model output in and return processed numpy array

    Parameters
    ----------
    task: str
        CLASSIFICATION OR REGRESSION
    outputs: torch tensor
        raw tensor model outputs
    labels:
        tensor of labels

    Return
    ------
    preds: dict
        dictionnary of predictions (and prediction scores - probability like)
    labels: numpy array
        numpy array of labels
    """
    
    if task == "REGRESSION":
        preds = outputs.cpu().detach().numpy()
        if labels is not None:
            labels = labels.cpu().detach().numpy()
        return {"preds": preds,}, labels
    
    elif task == "CLASSIFICATION":
        preds = outputs.argmax(axis=1).cpu().detach().numpy()
        preds_score = outputs.softmax(dim=1)[:, 1].cpu().detach().numpy()
        if labels is not None:
            labels = labels.cpu().detach().numpy()

        return {"preds": preds, "preds_score": preds_score,}, labels
    else:
        raise Exception("task not supported")
