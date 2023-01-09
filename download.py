import json
import os
import requests

DEFAULT_MODELS = [
    "ai21_j1-jumbo",
    "ai21_j1-large",
    "ai21_j1-grande",
    "together_bloom",
    "together_t0pp",
    "anthropic_stanford-online-all-v4-s3",
    "cohere_xlarge-20220609",
    "cohere_large-20220720",
    "cohere_medium-20220720",
    "cohere_small-20220720",
    "together_gpt-j-6b",
    "together_gpt-neox-20b",
    "together_t5-11b",
    "together_ul2",
    "together_opt-175b",
    "together_opt-66b",
    "microsoft_TNLGv2_530B",
    "microsoft_TNLGv2_7B",
    "openai_davinci",
    "openai_curie",
    "openai_babbage",
    "openai_ada",
    "openai_text-davinci-002",
    "openai_text-curie-001",
    "openai_text-babbage-001",
    "openai_text-ada-001",
    "openai_code-davinci-002",
    "openai_code-cushman-001",
    "together_glm",
    "together_yalm",
]

DEFAULT_URL_EXTRAS = {
    "together_t0pp": "stop=hash",
    "together_t5-11b": "stop=hash",
    "together_ul2": "stop=hash,global_prefix=nlg",
    "together_glm": "stop=hash",
}

TASKS = {
    "synthetic_reasoning_pattern_match": {
        "url_param": "synthetic_reasoning:mode=pattern_match,",
        "models": DEFAULT_MODELS,
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "synthetic_reasoning_variable_substitution": {
        "url_param": "synthetic_reasoning:mode=variable_substitution,",
        "models": DEFAULT_MODELS,
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "synthetic_reasoning_induction": {
        "url_param": "synthetic_reasoning:mode=induction,",
        "models": DEFAULT_MODELS,
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "synthetic_reasoning_natural_easy": {
        "url_param": "synthetic_reasoning_natural:difficulty=easy,",
        "models": DEFAULT_MODELS,
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "synthetic_reasoning_natural_hard": {
        "url_param": "synthetic_reasoning_natural:difficulty=hard,",
        "models": DEFAULT_MODELS,
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "babi_qa_all": {
        "url_param": "babi_qa:task=all,",
        "models": DEFAULT_MODELS,
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "babi_qa_3": {
        "url_param": "babi_qa:task=3,",
        "models": DEFAULT_MODELS,
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "babi_qa_15": {
        "url_param": "babi_qa:task=15,",
        "models": DEFAULT_MODELS,
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "babi_qa_19": {
        "url_param": "babi_qa:task=19,",
        "models": DEFAULT_MODELS,
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "dyck": {
        "url_param": "dyck_language_np=3:",
        "models": DEFAULT_MODELS,
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "gsm8k": {
        "url_param": "gsm:",
        "models": DEFAULT_MODELS,
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "math": {
        "url_param": "math:subject=all,level=1,use_official_examples=True,use_chain_of_thought=False,",
        "models": DEFAULT_MODELS,
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "math_cot": {
        "url_param": "math:subject=all,level=1,use_official_examples=False,use_chain_of_thought=True,",
        "models": DEFAULT_MODELS,
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "lsat_qa": {
        "url_param": "lsat_qa:task=all,method=multiple_choice_joint,",
        "models": DEFAULT_MODELS,
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "legal_support": {
        "url_param": "legal_support,method=multiple_choice_joint:",
        "models": DEFAULT_MODELS,
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "data_imputation_buy": {
        "url_param": "entity_data_imputation:dataset=Buy,",
        "models": list(
            set(DEFAULT_MODELS)
            - set(["openai_code-davinci-002", "openai_code-cushman-001"])
        ),
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "data_imputation_restaurant": {
        "url_param": "entity_data_imputation:dataset=Restaurant,",
        "models": list(
            set(DEFAULT_MODELS)
            - set(["openai_code-davinci-002", "openai_code-cushman-001"])
        ),
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "entity_matching_beer": {
        "url_param": "entity_matching:dataset=Beer,",
        "models": list(
            set(DEFAULT_MODELS)
            - set(["openai_code-davinci-002", "openai_code-cushman-001"])
        ),
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "entity_matching_abt_buy": {
        "url_param": "entity_matching:dataset=Abt_Buy,",
        "models": list(
            set(DEFAULT_MODELS)
            - set(["openai_code-davinci-002", "openai_code-cushman-001"])
        ),
        "url_extras": DEFAULT_URL_EXTRAS,
    },
    "entity_matching_dirty_itunes_amazon": {
        "url_param": "entity_matching:dataset=Dirty_iTunes_Amazon,",
        "models": list(
            set(DEFAULT_MODELS)
            - set(["openai_code-davinci-002", "openai_code-cushman-001"])
        ),
        "url_extras": DEFAULT_URL_EXTRAS,
    },
}

BASE_URL = (
    "https://storage.googleapis.com/"
    "crfm-helm-public/benchmark_output/runs/v1.0/"
    "{}"
    "{}/scenario_state_slim.json"
)

TASKS_TO_DOWNLOAD = [
    "synthetic_reasoning_pattern_match",
    "synthetic_reasoning_variable_substitution",
    "synthetic_reasoning_induction",
    "synthetic_reasoning_natural_easy",
    "synthetic_reasoning_natural_hard",
    "babi_qa_all",
    "babi_qa_3",
    "babi_qa_15",
    "babi_qa_19",
    "dyck",
    "gsm8k",
    "math",
    "math_cot",
    "lsat_qa",
    "legal_support",
    "data_imputation_buy",
    "data_imputation_restaurant",
    "entity_matching_beer",
    "entity_matching_abt_buy",
    "entity_matching_dirty_itunes_amazon",
]


def main():
    output_dir = os.environ["HELM_DATA_DIR"]
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "tasks.json"), "w") as f:
        json.dump(TASKS, f)
    for task_name in TASKS_TO_DOWNLOAD:
        task = TASKS[task_name]
        for model_name in task["models"]:
            url_insert = f"model={model_name}"
            if model_name in task["url_extras"]:
                url_insert = f"{url_insert},{task['url_extras'][model_name]}"
            url = BASE_URL.format(task["url_param"], url_insert)
            os.makedirs(os.path.join(output_dir, task_name), exist_ok=True)
            response = requests.get(url)
            response.raise_for_status()
            with open(
                os.path.join(output_dir, task_name, f"{model_name}.json"), "w"
            ) as f:
                json.dump(response.json(), f)


if __name__ == "__main__":
    main()
