# prediction_app/utils.py
import numpy as np

COCO_CATEGORIES = [
    "ConsolePort_Connected",
    "ConsolePort_NotConnected",
    "LAN1_Port_Connected_Lit",
    "LAN1_Port_Connected_NotLit",
    "LAN1_Port_NotConnected",
    "LAN2_Port_Connected_Lit",
    "LAN2_Port_Connected_NotLit",
    "LAN2_Port_NotConnected",
    "LAN3_Port_Connected_Lit",
    "LAN3_Port_Connected_NotLit",
    "LAN3_Port_NotConnected",
    "LAN4_Port_Connected_Lit",
    "LAN4_Port_Connected_NotLit",
    "LAN4_Port_NotConnected",
    "PowerCable_Connected",
    "PowerCable_NotConnected",
    "WAN1_Port_Connected_Lit",
    "WAN1_Port_Connected_NotLit",
    "WAN1_Port_NotConnected",
    "WAN2_Port_Connected_Lit",
    "WAN2_Port_Connected_NotLit",
    "WAN2_Port_NotConnected"
]

def select_top_predictions_per_group(predictions, n=8):
    """
    Select the top prediction for each port/cable type, ensuring diverse category representation.
    
    Parameters:
    - predictions: The raw output scores from the model.
    - n: The desired number of top predictions.

    Returns:
    - A list of the top predicted category names, respecting the port/cable groupings.
    """
    predictions = np.array(predictions)
    if predictions.ndim > 1:
        predictions = predictions.squeeze()
    
    top_predictions_per_group = {}
    for score, category in sorted(zip(predictions, COCO_CATEGORIES), reverse=True):
        base_name = "_".join(category.split("_")[:-1])
        if base_name not in top_predictions_per_group:
            top_predictions_per_group[base_name] = category
        if len(top_predictions_per_group) >= n:
            break
    
    selected_categories = list(top_predictions_per_group.values())
    return selected_categories
