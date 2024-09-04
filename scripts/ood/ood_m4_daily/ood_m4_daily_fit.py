import subprocess


scripts = [
    # "scripts/ood_m4_daily/nbeats/slope_inc.py",
    # "scripts/ood_m4_daily/nbeats/slope_dec.py",
    "scripts/ood_m4_daily/nbeats/seas_inc.py",
    # "scripts/ood_m4_daily/nbeats/seas_dec.py",

    # "scripts/ood_m4_daily/feedforward/slope_inc.py",
    # "scripts/ood_m4_daily/feedforward/slope_dec.py",
    "scripts/ood_m4_daily/feedforward/seas_inc.py",
    # "scripts/ood_m4_daily/feedforward/seas_dec.py",

    # "scripts/ood_m4_daily/seq2seq/slope_inc.py",
    # "scripts/ood_m4_daily/seq2seq/slope_dec.py",
    "scripts/ood_m4_daily/seq2seq/seas_inc.py",
    # "scripts/ood_m4_daily/seq2seq/seas_dec.py",

    # "scripts/ood_m4_daily/tcn/slope_inc.py",
    # "scripts/ood_m4_daily/tcn/slope_dec.py",
    "scripts/ood_m4_daily/tcn/seas_inc.py",
    # "scripts/ood_m4_daily/tcn/seas_dec.py",

    # "scripts/ood_m4_daily/transformer/slope_inc.py",
    # "scripts/ood_m4_daily/transformer/slope_dec.py",
    "scripts/ood_m4_daily/transformer/seas_inc.py",
    # "scripts/ood_m4_daily/transformer/seas_dec.py",
]

for scripts in scripts:
    subprocess.run(["python", scripts])


configs = [
    # "experiments/m4_daily/nbeats_g_gen_slope_inc/config.yaml",
    # "experiments/m4_daily/nbeats_g_gen_slope_dec/config.yaml",
    "experiments/m4_daily/nbeats_g_gen_seas_inc/config.yaml",
    # "experiments/m4_daily/nbeats_g_gen_seas_dec/config.yaml",

    # "experiments/m4_daily/transformer_gen_slope_inc/config.yaml",
    # "experiments/m4_daily/transformer_gen_slope_dec/config.yaml",
    "experiments/m4_daily/transformer_gen_seas_inc/config.yaml",
    # "experiments/m4_daily/transformer_gen_seas_dec/config.yaml",

    # "experiments/m4_daily/tcn_gen_slope_inc/config.yaml",
    # "experiments/m4_daily/tcn_gen_slope_dec/config.yaml",
    "experiments/m4_daily/tcn_gen_seas_inc/config.yaml",
    # "experiments/m4_daily/tcn_gen_seas_dec/config.yaml",

    # "experiments/m4_daily/seq2seq_gen_slope_inc/config.yaml",
    # "experiments/m4_daily/seq2seq_gen_slope_dec/config.yaml",
    "experiments/m4_daily/seq2seq_gen_seas_inc/config.yaml",
    # "experiments/m4_daily/seq2seq_gen_seas_dec/config.yaml",

    # "experiments/m4_daily/feedforward_gen_slope_inc/config.yaml",
    # "experiments/m4_daily/feedforward_gen_slope_dec/config.yaml",
    "experiments/m4_daily/feedforward_gen_seas_inc/config.yaml",
    # "experiments/m4_daily/feedforward_gen_seas_dec/config.yaml",
]


for config in configs:
    subprocess.run(["python", "scripts/evaluate.py", config])
