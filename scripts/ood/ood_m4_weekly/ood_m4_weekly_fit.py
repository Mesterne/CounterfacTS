import subprocess


scripts = [
    # "scripts/ood_m4_weekly/nbeats/slope_inc.py",
    # "scripts/ood_m4_weekly/nbeats/slope_dec.py",
    # "scripts/ood_m4_weekly/nbeats/trend_str_dec.py",

    # "scripts/ood_m4_weekly/feedforward/slope_inc.py",
    # "scripts/ood_m4_weekly/feedforward/slope_dec.py",
    # "scripts/ood_m4_weekly/feedforward/trend_str_dec.py",

    # "scripts/ood_m4_weekly/seq2seq/slope_inc.py",
    # "scripts/ood_m4_weekly/seq2seq/slope_dec.py",
    # "scripts/ood_m4_weekly/seq2seq/trend_str_dec.py",

    # "scripts/ood_m4_weekly/tcn/slope_inc.py",
    # "scripts/ood_m4_weekly/tcn/slope_dec.py",
    # "scripts/ood_m4_weekly/tcn/trend_str_dec.py",

    # "scripts/ood_m4_weekly/transformer/slope_inc.py",
    # "scripts/ood_m4_weekly/transformer/slope_dec.py",
    # "scripts/ood_m4_weekly/transformer/trend_str_dec.py",
]

for scripts in scripts:
    subprocess.run(["python", scripts])


configs = [
    "experiments/m4_weekly/nbeats_g_gen_slope_inc/config.yaml",
    "experiments/m4_weekly/nbeats_g_gen_slope_dec/config.yaml",
    # "experiments/m4_weekly/nbeats_g_gen_trend_str_dec/config.yaml",

    "experiments/m4_weekly/transformer_gen_slope_inc/config.yaml",
    "experiments/m4_weekly/transformer_gen_slope_dec/config.yaml",
    # "experiments/m4_weekly/transformer_gen_trend_str_dec/config.yaml",

    "experiments/m4_weekly/tcn_gen_slope_inc/config.yaml",
    "experiments/m4_weekly/tcn_gen_slope_dec/config.yaml",
    # "experiments/m4_weekly/tcn_gen_trend_str_dec/config.yaml",

    "experiments/m4_weekly/seq2seq_gen_slope_inc/config.yaml",
    "experiments/m4_weekly/seq2seq_gen_slope_dec/config.yaml",
    # "experiments/m4_weekly/seq2seq_gen_trend_str_dec/config.yaml",

    "experiments/m4_weekly/feedforward_gen_slope_inc/config.yaml",
    "experiments/m4_weekly/feedforward_gen_slope_dec/config.yaml",
    # "experiments/m4_weekly/feedforward_gen_trend_str_dec/config.yaml",
]


for config in configs:
    subprocess.run(["python", "scripts/evaluate.py", config])
