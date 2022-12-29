import numpy as np
from sklearn.linear_model import LogisticRegression

from difficulty import convert_difficulties_to_quantiles


def get_logistic_agent_characteristic(
    instance_difficulties: dict[str, float],
    correct: dict[str, int],
    quantiles: bool = True,
):
    if quantiles:
        x_for_fit = convert_difficulties_to_quantiles(instance_difficulties)
        x_for_acc = np.linspace(0, 100, 100)
    else:
        x_for_fit = np.array(list(instance_difficulties.values()))
        x_for_acc = np.linspace(0, 1, 100)
    x_for_fit = np.expand_dims(x_for_fit, 1)
    y_for_fit = np.array(list(correct.values()))
    clf = LogisticRegression(penalty=None).fit(x_for_fit, y_for_fit)
    y_for_acc = clf.predict_proba(np.expand_dims(x_for_acc, 1))[:, 1]
    return x_for_acc, y_for_acc
