from enum import Enum
from collections import defaultdict
import re
from itertools import permutations
from typing import Optional

from load import load_model_task_data, load_tasks_data


class Split(Enum):
    TEST = "test"
    VALID = "valid"


def get_accuracy_per_model(task_name: str, split: Optional[Split] = None):
    tasks = load_tasks_data()
    task = tasks[task_name]
    instance_accuracy_per_model = {}
    for model_name in task["models"]:
        model_results = load_model_task_data(task_name, model_name)
        instance_accuracy_per_model[
            model_name
        ] = _get_instance_accuracy_for_single_model(
            model_results["request_states"], task_name, split
        )

    return instance_accuracy_per_model


def get_accuracy_per_trial(instance_results_per_model: dict):
    result = defaultdict(list)
    for model_name, model_results in instance_results_per_model.items():
        for instance_result in model_results:
            result[f'{instance_result["id"]}_{instance_result["trial"]}'].append(
                {
                    "model": model_name,
                    "is_correct": instance_result["is_correct"],
                }
            )
    return result


def _response_is_exact_match(response: dict):
    expected, completion = _get_expected_and_completion(response)
    return int(expected == completion.strip()), expected, completion


def _response_is_correct_choice(response: dict):
    expected, completion = _get_expected_and_completion(response)
    prediction = response["output_mapping"].get(completion, "")
    return int(expected == prediction.strip()), expected, completion


def _response_is_exact_match_up_to_symbol_permutation(response: dict):
    expected, completion = _get_expected_and_completion(response)
    return (
        int(_exact_match_up_to_symbol_permutation(expected, completion)),
        expected,
        completion,
    )


def _exact_match_of_boxed_expression(response: dict):
    expected, completion = _get_expected_and_completion(response)
    expected = re.search(r"boxed{(.*)}", expected).group(1)
    completion = re.search(r"boxed{(.*)}", completion)
    if completion is not None:
        completion = completion.group(1)
    return int(expected == completion), expected, completion


def _exact_match_of_answer(response: dict):
    expected, completion = _get_expected_and_completion(response)
    expected = re.search(r"The answer is (.*).", expected).group(1)
    completion = re.search(r"The answer is (.*).", completion)
    if completion is not None:
        completion = completion.group(1)
    return int(expected == completion), expected, completion


def _exact_match_up_to_symbol_permutation(expected: str, completion: str):
    original_order = ("X", "Y", "Z")
    for permutation in permutations(original_order):
        permuted_response = ""
        for character in completion:
            if character in original_order:
                permuted_response += permutation[original_order.index(character)]
            else:
                permuted_response += character
        if permuted_response == expected:
            return True
    return False


def _get_expected_and_completion(response: dict):
    correct_references = [
        reference
        for reference in response["instance"]["references"]
        if "correct" in reference["tags"]
    ]
    assert len(correct_references) == 1
    expected = correct_references[0]["output"].strip()
    completions = response["result"]["completions"]
    assert len(completions) == 1
    completion = completions[0]["text"].strip()
    return expected, completion


def _get_instance_accuracy_for_single_model(
    model_responses: dict, task_name: str, split: Optional[Split] = None
):
    result = []
    if task_name == "synthetic_reasoning_induction":
        correctness_function = _response_is_exact_match_up_to_symbol_permutation
    elif task_name == "math_cot":
        correctness_function = _exact_match_of_boxed_expression
    elif task_name == "gsm8k":
        correctness_function = _exact_match_of_answer
    elif "output_mapping" in model_responses[0]:
        correctness_function = _response_is_correct_choice
    else:
        correctness_function = _response_is_exact_match
    for response in model_responses:
        is_correct, expected, prediction = correctness_function(response)
        if split is not None and response["instance"]["split"] != split.value:
            continue
        result.append(
            {
                "id": f'{response["instance"]["id"]}_{response["train_trial_index"]}',
                "trial": response["train_trial_index"],
                "is_correct": is_correct,
                "expected": expected,
                "actual": prediction,
            }
        )
    return result


def normalize_accuracy(accuracy: float, num_options: int):
    return (accuracy - 1 / num_options) * num_options
