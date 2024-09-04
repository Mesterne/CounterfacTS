import subprocess


scripts = [
    # "scripts/ood_m4_hourly/nbeats/seasonality.py",
    # "scripts/ood_m4_hourly/nbeats/slope_inc.py",
    # "scripts/ood_m4_hourly/nbeats/slope_dec.py",
    "scripts/ood_m4_hourly/nbeats/all.py",

    # "scripts/ood_m4_hourly/feedforward/seasonality.py",
    # "scripts/ood_m4_hourly/feedforward/slope_inc.py",
    # "scripts/ood_m4_hourly/feedforward/slope_dec.py",
    "scripts/ood_m4_hourly/feedforward/all.py",

    # "scripts/ood_m4_hourly/seq2seq/seasonality.py",
    # "scripts/ood_m4_hourly/seq2seq/slope_inc.py",
    # "scripts/ood_m4_hourly/seq2seq/slope_dec.py",
    "scripts/ood_m4_hourly/seq2seq/all.py",

    # "scripts/ood_m4_hourly/tcn/seasonality.py",
    # "scripts/ood_m4_hourly/tcn/slope_inc.py",
    # "scripts/ood_m4_hourly/tcn/slope_dec.py",
    "scripts/ood_m4_hourly/tcn/all.py",

    # "scripts/ood_m4_hourly/transformer/seasonality.py",
    # "scripts/ood_m4_hourly/transformer/slope_inc.py",
    # "scripts/ood_m4_hourly/transformer/slope_dec.py",
    # "scripts/ood_m4_hourly/transformer/all.py",
]

for scripts in scripts:
    subprocess.run(["python", scripts])


configs = [
    # "experiments/m4_hourly/nbeats_g_gen_seas_dec/config.yaml",
    # "experiments/m4_hourly/nbeats_g_gen_slope_inc/config.yaml",
    # "experiments/m4_hourly/nbeats_g_gen_slope_dec/config.yaml",
    "experiments/m4_hourly/nbeats_g_gen_all/config.yaml",

    # "experiments/m4_hourly/transformer_gen_seas_dec/config.yaml",
    # "experiments/m4_hourly/transformer_gen_slope_inc/config.yaml",
    # "experiments/m4_hourly/transformer_gen_slope_dec/config.yaml",
    "experiments/m4_hourly/transformer_gen_all/config.yaml",

    # "experiments/m4_hourly/tcn_gen_seas_dec/config.yaml",
    # "experiments/m4_hourly/tcn_gen_slope_inc/config.yaml",
    # "experiments/m4_hourly/tcn_gen_slope_dec/config.yaml",
    "experiments/m4_hourly/tcn_gen_all/config.yaml",

    # "experiments/m4_hourly/seq2seq_gen_seas_dec/config.yaml",
    # "experiments/m4_hourly/seq2seq_gen_slope_inc/config.yaml",
    # "experiments/m4_hourly/seq2seq_gen_slope_dec/config.yaml",
    "experiments/m4_hourly/seq2seq_gen_all/config.yaml",

    # "experiments/m4_hourly/feedforward_gen_seas_dec/config.yaml",
    # "experiments/m4_hourly/feedforward_gen_slope_inc/config.yaml",
    # "experiments/m4_hourly/feedforward_gen_slope_dec/config.yaml",
    # "experiments/m4_hourly/feedforward_gen_all/config.yaml",
]


for config in configs:
    subprocess.run(["python", "scripts/evaluate.py", config])
