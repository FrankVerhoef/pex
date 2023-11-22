import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
import random
import utils.logging as logging
from utils.general import prettydict

def do_tune(train_fn, run_config):
    ray.init(
        configure_logging=True,
        logging_level="warning",
        )
    search_space = {
        "seed": tune.grid_search([42, 2206, 1968, 524, 123, 55]),
        # "context_option": tune.grid_search([3, 4]),
        # "config_option": tune.grid_search([0, 3]),
        # "persona_selector": tune.sample_from(lambda spec: {
        #     3: 'preprocessed:trained_base_reweighted_bart',
        #     4: None
        #     }[spec.config.context_option]),
        # "sessionbreak_token": tune.sample_from(lambda spec: {
        #     0: None,
        #     3: '<session>'
        # }[spec.config.config_option])
        # "prefix_aggr": tune.grid_search(["concat", "max", "avg"]),
        # "lm_loss_factor": tune.sample_from(lambda spec: random.random()),
        "speaker_prefixes": tune.grid_search([None, ["[other]", "[self]"]]),
        "nofact_token": tune.sample_from(lambda spec: "" if spec.config.speaker_prefixes is None else "[nofact]"),
        "add_tokens": tune.sample_from(
            lambda spec: 
                spec.config.speaker_prefixes 
                if spec.config.speaker_prefixes is None 
                else spec.config.speaker_prefixes + [spec.config.nofact_token]
            ),
        # "learning_rate": tune.grid_search([1e-5, 1e-4, 1e-3]),
        # "batch_size": tune.grid_search([32, 64, 128]),
        # "prefix_size": tune.grid_search([0, 5]),
        # # If there is a prefix, then freeze all Bert layers
        # # If the is no prefix, then vary the number of frozen layers
        # "freeze": tune.sample_from(
        #         lambda spec: {
        #             0: None, 
        #             1: 8, 
        #             2: 12
        #         }[random.randint(0,2)]
        #         if spec.config.prefix_size == 0 else 12
        #     ),
    }

    tuner = tune.Tuner(
        trainable=train_fn, 
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=HyperBandScheduler(),
            metric="valid_loss", 
            mode="min",
            num_samples=2,
            max_concurrent_trials=8
        ),
        run_config = run_config
    )
    results = tuner.fit()
    best_result = results.get_best_result() 
    logging.success(prettydict(best_result.config, "Best config"))
    logging.success(prettydict(best_result.metrics, "Best metrics"))
    return results

