import subprocess


scripts = [
    # "scripts/ood_elec/nbeats/seasonality.py",
    # "scripts/ood_elec/nbeats/slope_inc.py",
    # "scripts/ood_elec/nbeats/slope_dec.py",
    "scripts/ood_elec/nbeats/all.py",

    # "scripts/ood_elec/feedforward/seasonality.py",
    # "scripts/ood_elec/feedforward/slope_inc.py",
    # "scripts/ood_elec/feedforward/slope_dec.py",
    "scripts/ood_elec/feedforward/all.py",

    # "scripts/ood_elec/seq2seq/seasonality.py",
    # "scripts/ood_elec/seq2seq/slope_inc.py",
    # "scripts/ood_elec/seq2seq/slope_dec.py",
    "scripts/ood_elec/seq2seq/all.py",

    # "scripts/ood_elec/tcn/seasonality.py",
    # "scripts/ood_elec/tcn/slope_inc.py",
    # "scripts/ood_elec/tcn/slope_dec.py",
    "scripts/ood_elec/tcn/all.py",

    # "scripts/ood_elec/transformer/seasonality.py",
    # "scripts/ood_elec/transformer/slope_inc.py",
    # "scripts/ood_elec/transformer/slope_dec.py",
    "scripts/ood_elec/transformer/all.py",
]

for scripts in scripts:
    subprocess.run(["python", scripts])


configs = [
    # "experiments/electricity_nips/nbeats_g_gen_seas_dec/config.yaml",
    # "experiments/electricity_nips/nbeats_g_gen_slope_inc/config.yaml",
    # "experiments/electricity_nips/nbeats_g_gen_slope_dec/config.yaml",
    "experiments/electricity_nips/nbeats_g_gen_all/config.yaml",

    # "experiments/electricity_nips/transformer_gen_seas_dec/config.yaml",
    # "experiments/electricity_nips/transformer_gen_slope_inc/config.yaml",
    # "experiments/electricity_nips/transformer_gen_slope_dec/config.yaml",
    "experiments/electricity_nips/transformer_g_gen_all/config.yaml",

    # "experiments/electricity_nips/tcn_gen_seas_dec/config.yaml",
    # "experiments/electricity_nips/tcn_gen_slope_inc/config.yaml",
    # "experiments/electricity_nips/tcn_gen_slope_dec/config.yaml",
    "experiments/electricity_nips/tcn_g_gen_all/config.yaml",

    # "experiments/electricity_nips/seq2seq_gen_seas_dec/config.yaml",
    # "experiments/electricity_nips/seq2seq_gen_slope_inc/config.yaml",
    # "experiments/electricity_nips/seq2seq_gen_slope_dec/config.yaml",
    "experiments/electricity_nips/seq2seq_g_gen_all/config.yaml",

    # "experiments/electricity_nips/feedforward_gen_seas_dec/config.yaml",
    # "experiments/electricity_nips/feedforward_gen_slope_inc/config.yaml",
    # "experiments/electricity_nips/feedforward_gen_slope_dec/config.yaml",
    "experiments/electricity_nips/feedforward_g_gen_all/config.yaml",
]


for config in configs:
    subprocess.run(["python", "scripts/evaluate.py", config])
