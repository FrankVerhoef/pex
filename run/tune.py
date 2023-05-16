import ray
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
import random
import utils.logging as logging

def do_grid_search(train_fn):
    ray.init(
        configure_logging=True,
        logging_level="warning",
        )
    search_space = {
        "seed": tune.grid_search([42, 123, 1968, 2206]),
        "lm_loss_factor": tune.sample_from(lambda spec: random.random()),
        # "prefix_aggr": tune.grid_search(["concat", "max", "avg"]),
        # "speaker_prefixes": tune.grid_search([None, ["<self>", "<other>"]]),
        # "nofact_token": tune.sample_from(lambda spec: "" if spec.config.speaker_prefixes is None else "<nofact>"),
        # "add_tokens": tune.sample_from(
        #     lambda spec: 
        #         spec.config.speaker_prefixes 
        #         if spec.config.speaker_prefixes is None 
        #         else spec.config.speaker_prefixes + [spec.config.nofact_token]
        #     ),
        # "learning_rate": tune.grid_search([1e-5, 1e-4, 1e-3]),
        # "batch_size": tune.grid_search([16, 32, 64]),
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
    trainable_with_resources = tune.with_resources(train_fn, {"gpu": 1})
    tuner = tune.Tuner(
        trainable=trainable_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=HyperBandScheduler(),
            metric="loss", 
            mode="min",
            num_samples=5,
            max_concurrent_trials=8
        ),
        run_config = air.RunConfig(
            verbose=3,
        )
    )
    results = tuner.fit()
    best_result = results.get_best_result() 
    logging.success("BEST RESULTS: {}".format(best_result.config))
    logging.success("BEST METRICS: {:.2%}".format(best_result.metrics["valid_acc"]))
    return results

