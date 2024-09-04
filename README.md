# Running the application

To run the application start by installing and activating the environment:

```shell
conda env create -f env.yaml
conda activate whatif
```

We can then use the following command to run the application

```shell
bokeh serve src/ --args <config-path>
```

where the `config_path` is the path to a config.yaml file in the experiments folder.
As a concrete example, this command will run the application using a simple dense network on the electricity dataset:
```shell
bokeh serve src/ --args experiments/electricity_nips/feedforward/config.yaml
```


# Running the application for new datasets and/or models
To run the application with a new model and/or dataset create a config.yaml file in `experiments/<dataset>/<model>`. Then train the model can be trained using by running
```shell
python scripts/fit.py <config_path>
```

During training, the script saves batches of training data to ``data/<dataset>/training_data`. We can calculate features from this data like this:
```shell
python scripts/calculate_features.py <config_path>
```

To calculate the features for the predefined test data from GluonTS use the optional `--test-data` argument:
```shell
python scripts/calculate_features.py <config_path> --test-data=1
```

Evaluating new models can be done like this:
```shell
python scripts/evaluate.py <config_path>
```

Once all of the above has been completed for the new dataset and/or model, the application can be run as usual.