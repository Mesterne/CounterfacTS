import subprocess


scripts = [
    # "scripts/ood_m4_monthly/nbeats/seasonality.py",
    # "scripts/ood_m4_monthly/nbeats/slope_inc.py",
    # "scripts/ood_m4_monthly/nbeats/slope_dec.py",
    "scripts/ood_m4_monthly/nbeats/lin_dec.py",
    "scripts/ood_m4_monthly/nbeats/trend_str_dec.py",
    # "scripts/ood_m4_monthly/nbeats/all.py",

    # "scripts/ood_m4_monthly/feedforward/seasonality.py",
    # "scripts/ood_m4_monthly/feedforward/slope_inc.py",
    # "scripts/ood_m4_monthly/feedforward/slope_dec.py",
    "scripts/ood_m4_monthly/feedforward/lin_dec.py",
    "scripts/ood_m4_monthly/feedforward/trend_str_dec.py",
    # "scripts/ood_m4_monthly/feedforward/all.py",

    # "scripts/ood_m4_monthly/seq2seq/seasonality.py",
    # "scripts/ood_m4_monthly/seq2seq/slope_inc.py",
    # "scripts/ood_m4_monthly/seq2seq/slope_dec.py",
    "scripts/ood_m4_monthly/seq2seq/lin_dec.py",
    "scripts/ood_m4_monthly/seq2seq/trend_str_dec.py",
    # "scripts/ood_m4_monthly/seq2seq/all.py",

    # "scripts/ood_m4_monthly/tcn/seasonality.py",
    # "scripts/ood_m4_monthly/tcn/slope_inc.py",
    # "scripts/ood_m4_monthly/tcn/slope_dec.py",
    "scripts/ood_m4_monthly/tcn/lin_dec.py",
    "scripts/ood_m4_monthly/tcn/trend_str_dec.py",
    # "scripts/ood_m4_monthly/tcn/all.py",

    # "scripts/ood_m4_monthly/transformer/seasonality.py",
    # "scripts/ood_m4_monthly/transformer/slope_inc.py",
    # "scripts/ood_m4_monthly/transformer/slope_dec.py",
    "scripts/ood_m4_monthly/transformer/lin_dec.py",
    "scripts/ood_m4_monthly/transformer/trend_str_dec.py",
    # "scripts/ood_m4_monthly/transformer/all.py",
]

for scripts in scripts:
    subprocess.run(["python", scripts])


configs = [
    # "experiments/m4_monthly/nbeats_g_gen_seas_dec/config.yaml",
    # "experiments/m4_monthly/nbeats_g_gen_slope_inc/config.yaml",
    # "experiments/m4_monthly/nbeats_g_gen_slope_dec/config.yaml",
    "experiments/m4_monthly/nbeats_g_gen_lin_dec/config.yaml",
    "experiments/m4_monthly/nbeats_g_gen_trend_str_dec/config.yaml",
    # "experiments/m4_monthly/nbeats_g_gen_all/config.yaml",

    # "experiments/m4_monthly/transformer_gen_seas_dec/config.yaml",
    # "experiments/m4_monthly/transformer_gen_slope_inc/config.yaml",
    # "experiments/m4_monthly/transformer_gen_slope_dec/config.yaml",
    "experiments/m4_monthly/transformer_gen_lin_dec/config.yaml",
    "experiments/m4_monthly/transformer_gen_trend_str_dec/config.yaml",
    # "experiments/m4_monthly/transformer_gen_all/config.yaml",

    # "experiments/m4_monthly/feedforward_gen_seas_dec/config.yaml",
    # "experiments/m4_monthly/feedforward_gen_slope_inc/config.yaml",
    # "experiments/m4_monthly/feedforward_gen_slope_dec/config.yaml",
    "experiments/m4_monthly/feedforward_gen_lin_dec/config.yaml",
    "experiments/m4_monthly/feedforward_gen_trend_str_dec/config.yaml",
    # "experiments/m4_monthly/feedforward_gen_all/config.yaml",

    # "experiments/m4_monthly/tcn_gen_seas_dec/config.yaml",
    # "experiments/m4_monthly/tcn_gen_slope_inc/config.yaml",
    # "experiments/m4_monthly/tcn_gen_slope_dec/config.yaml",
    "experiments/m4_monthly/tcn_gen_lin_dec/config.yaml",
    "experiments/m4_monthly/tcn_gen_trend_str_dec/config.yaml",
    # "experiments/m4_monthly/tcn_gen_all/config.yaml",

    # "experiments/m4_monthly/seq2seq_gen_seas_dec/config.yaml",
    # "experiments/m4_monthly/seq2seq_gen_slope_inc/config.yaml",
    # "experiments/m4_monthly/seq2seq_gen_slope_dec/config.yaml",
    "experiments/m4_monthly/seq2seq_gen_lin_dec/config.yaml",
    "experiments/m4_monthly/seq2seq_gen_trend_str_dec/config.yaml",
    # "experiments/m4_monthly/seq2seq_gen_all/config.yaml",
]


for config in configs:
    subprocess.run(["python", "scripts/evaluate.py", config])
