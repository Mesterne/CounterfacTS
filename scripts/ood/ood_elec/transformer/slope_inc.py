import logging
import os
import sys

import random
import time
import yaml
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.time_feature import (
    HourOfDay,
    DayOfWeek,
    DayOfMonth,
    DayOfYear,
    MonthOfYear
)
from gluonts.torch.batchify import batchify
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    ValidationSplitSampler
)
from statsmodels.tsa.seasonal import STL
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

sys.path.append(".")
from src.models.utils import get_model
from src.utils.transformations import manipulate_trend_component


def create_train_dataloader(dataset, context_length, prediction_length, batch_size, num_batches_per_epoch):
    transformation = Chain([
        AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        ),
        AddTimeFeatures(
            start_field=FieldName.START,
            target_field=FieldName.TARGET,
            output_field=FieldName.FEAT_TIME,
            pred_length=prediction_length,
            time_features=[HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear(), MonthOfYear()]
        ),
        InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=ExpectedNumInstanceSampler(num_instances=1, min_past=context_length,
                                                        min_future=prediction_length),
            past_length=context_length,
            future_length=prediction_length,
            time_series_fields=[FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES]
        )
    ])
    dataloader = TrainDataLoader(
        dataset,
        batch_size=batch_size,
        stack_fn=batchify,
        transform=transformation,
        num_batches_per_epoch=num_batches_per_epoch,
        num_workers=1
    )
    return dataloader


def create_validation_dataloader(dataset, context_length, prediction_length, batch_size, num_batches_per_epoch):
    transformation = Chain([
        AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        ),
        AddTimeFeatures(
            start_field=FieldName.START,
            target_field=FieldName.TARGET,
            output_field=FieldName.FEAT_TIME,
            pred_length=prediction_length,
            time_features=[HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear(), MonthOfYear()]
        ),
        InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=ValidationSplitSampler(min_future=prediction_length),
            past_length=context_length,
            future_length=prediction_length,
            time_series_fields=[FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES]
        )
    ])
    dataloader = ValidationDataLoader(
        dataset,
        batch_size=batch_size,
        stack_fn=batchify,
        transform=transformation,
        num_workers=1
    )
    return dataloader


def get_train_and_val_data(dataset, num_validation_windows, context_length, prediction_length):
    train_data = ListDataset(list(iter(dataset.train)), freq=dataset.metadata.freq)
    validation_data = []
    for i in range(num_validation_windows):
        for ts in train_data.list_data:
            # only add time series long enough that we can remove one horizon and still have context_length +
            # prediction_length values left
            if len(ts["target"]) <= context_length + prediction_length * (i + 2):
                continue

            val_ts = deepcopy(ts)
            val_ts["target"] = val_ts["target"][:-prediction_length * i if i > 0 else None]
            val_ts["target"] = val_ts["target"][-(context_length + prediction_length):]
            validation_data.append(val_ts)

            # slice off the validation data from the training data
            if i == num_validation_windows - 1:
                ts["target"] = ts["target"][:-prediction_length * (i + 1)]
    
    return train_data, ListDataset(validation_data, freq=dataset.metadata.freq)


def add_to_dataset(original_data, generated_data):
    for generated, original in zip(generated_data, original_data):
        entry = {"start": generated.index[0], "target": generated.values,
                 "feat_static_cat": original["feat_static_cat"], "item_id": original["item_id"]}
        original_data.list_data.append(entry)
    
    return original_data


def change_slope(data, sp, m):
    generated = []
    for ts in tqdm(data):
        index = pd.date_range(start=ts["start"], freq=ts["start"].freq, periods=len(ts["target"]))
        ts = pd.Series(data=ts["target"], index=index)
        decomp = STL(ts, period=sp).fit()
        
        new_trend = manipulate_trend_component(decomp.trend, f=1, g=1, h=1, m=m / len(ts))
        new_ts = new_trend + decomp.seasonal + decomp.resid
        generated.append(new_ts)
    
    return generated


if __name__ == '__main__':
    # configure script
    model_name = "transformer"
    dataset = "electricity_nips"
    folder_suffix = "slope_inc"

    datadir = f"data/{dataset}"
    experiment_dir = f"experiments/{dataset}/{model_name}"
    generated_datadir = os.path.join(datadir, "generated", "test")

    # load original config, replace the experient path, create directory and dump config
    with open(os.path.join(experiment_dir, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    experiment_dir = f"experiments/{config['dataset']}/{config['model_name']}_gen_{folder_suffix}"
    os.makedirs(experiment_dir, exist_ok=True)
    config["path"] = experiment_dir

    with open(os.path.join(experiment_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)

    # load new config
    with open(os.path.join(experiment_dir, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # create logger
    logging.basicConfig(filename=os.path.join(config["path"], "train.log"), level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler())  # print to stdout as well as file
    logger = logging.getLogger("trainer")
    logger.info(f"Logger initialized. Training model {model_name}.")

    # initialize model
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(config["model_name"])(**config["model_args"], device=device, path=config["path"]).to(device)

    # load data and augment
    logger.info("Loading data and creating dataloaders")
    dataset = get_dataset(config["dataset"])
    train_data, validation_data = get_train_and_val_data(dataset, config["trainer_args"]["num_validation_windows"],
                                                        config["trainer_args"]["context_length"], config["trainer_args"]["prediction_length"])

    generated_train = change_slope(train_data, config["sp"], m=10)
    expanded_train = add_to_dataset(train_data, generated_train)

    generated_val = change_slope(validation_data, config["sp"], m=1)
    expanded_val = add_to_dataset(validation_data, generated_val)

    train_dataloader = create_train_dataloader(expanded_train, config["trainer_args"]["context_length"], config["trainer_args"]["prediction_length"],
                                            config["trainer_args"]["batch_size"], config["trainer_args"]["num_batches_per_epoch"])

    validation_dataloader = create_validation_dataloader(expanded_val, config["trainer_args"]["context_length"], config["trainer_args"]["prediction_length"],
                                                        config["trainer_args"]["batch_size"], config["trainer_args"]["num_batches_per_epoch"])

    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config["trainer_args"]["patience"],
                                                           factor=0.5, min_lr=5e-5, verbose=True)

    best_smape = None
    model.train()
    for epoch_no in range(1, config["trainer_args"]["epochs"] + 1):
        epoch_start = time.time()
        sum_epoch_loss = 0
        for batch_no, batch in enumerate(train_dataloader, start=1):
            # calculate loss
            loss = model.calculate_loss(batch)
            if torch.isnan(torch.sum(loss)):
                logger.critical(f"NaN loss value, epoch {epoch_no} batch {batch_no}")
                raise ValueError(f"NaN loss value, epoch {epoch_no} batch {batch_no}")

            # step
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            sum_epoch_loss += loss.detach().cpu().numpy().item()

        if epoch_no % 5 == 0:
            mases = []
            smapes = []
            mses = []
            model.eval()
            for batch in validation_dataloader:
                mase, smape, mse = model.validate(batch, sp=config["sp"])

                mases.append(mase)
                smapes.append(smape)
                mses.append(mse)

            model.train()
            val_mase = np.mean(mases)
            val_smape = np.mean(smapes)
            val_mse = np.mean(mses)

            logger.info(f"Epoch {epoch_no}, time spent: {round(time.time() - epoch_start, 1)}, "
                        f"average training loss: {sum_epoch_loss / config['trainer_args']['num_batches_per_epoch']}, validation scores: "
                        f"[MASE: {val_mase}, MSE: {val_mse}, sMAPE: {val_smape}]")

            scheduler.step(val_smape)
            if (best_smape is None or val_smape < best_smape):
                best_smape = val_smape
                torch.save(model.state_dict(), os.path.join(config["path"], "temp.pth"))
        else:
            logger.info(f"Epoch {epoch_no}, time spent: {round(time.time() - epoch_start, 1)}, "
                        f"average training loss: {sum_epoch_loss / config['trainer_args']['num_batches_per_epoch']}")

        if optimizer.param_groups[0]["lr"] == scheduler.min_lrs[0] and scheduler.num_bad_epochs == config["trainer_args"]["patience"]:
            logger.info("Stopping training due to lack of improvement in validation loss")
            break

    logger.info(f"Done training. Best validation sMAPE: {best_smape}")
    model.load_state_dict(torch.load(os.path.join(config["path"], "temp.pth")))
    os.remove(os.path.join(config["path"], "temp.pth"))

    torch.save(model.state_dict(), os.path.join(config["path"], "model.pth"))
