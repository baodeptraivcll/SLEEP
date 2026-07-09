# Wrapper for compatibility
from common.evaluate import evaluate_model as common_evaluate

def evaluate_model(model, loader, device='cuda', verbose=True):
    res = common_evaluate(model, loader, device)
    # Return backward compatible tuple: acc, macro_f1, kappa, cm
    return res["accuracy"], res["macro_f1"], res["kappa"], res["confusion_matrix"]
