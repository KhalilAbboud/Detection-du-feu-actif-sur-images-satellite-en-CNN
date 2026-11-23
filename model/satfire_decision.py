import numpy as np

def decide(probs, fire_factor=1.0, forest_factor=1.0):
    labels = np.argmax(probs, axis=-1).astype("uint8")
    active_group = (labels == 1) | (labels == 2)
    forest_group = ~active_group
    if fire_factor != 1.0:
        fire_wins = probs[..., 1] * fire_factor >= probs[..., 2]
        labels[active_group] = np.where(fire_wins[active_group], 1, 2)
    if forest_factor != 1.0:
        forest_wins = probs[..., 0] * forest_factor >= probs[..., 3]
        labels[forest_group] = np.where(forest_wins[forest_group], 0, 3)
    return labels
