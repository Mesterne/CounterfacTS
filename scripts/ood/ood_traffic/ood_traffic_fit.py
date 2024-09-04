import subprocess


scripts = [
    # "scripts/ood_traffic/nbeats/seasonality.py",
    # "scripts/ood_traffic/nbeats/slope_inc.py",
    # "scripts/ood_traffic/nbeats/slope_dec.py",
    # "scripts/ood_traffic/nbeats/lin_dec.py",
    "scripts/ood_traffic/nbeats/trend_str_dec.py",
    # "scripts/ood_traffic/nbeats/all.py",

    # "scripts/ood_traffic/feedforward/seasonality.py",
    # "scripts/ood_traffic/feedforward/slope_inc.py",
    # "scripts/ood_traffic/feedforward/slope_dec.py",
    # "scripts/ood_traffic/feedforward/lin_dec.py",
    "scripts/ood_traffic/feedforward/trend_str_dec.py",
    # "scripts/ood_traffic/feedforward/all.py",

    # "scripts/ood_traffic/seq2seq/seasonality.py",
    # "scripts/ood_traffic/seq2seq/slope_inc.py",
    # "scripts/ood_traffic/seq2seq/slope_dec.py",
    # "scripts/ood_traffic/seq2seq/lin_dec.py",
    "scripts/ood_traffic/seq2seq/trend_str_dec.py",
    # "scripts/ood_traffic/seq2seq/all.py",

    # "scripts/ood_traffic/tcn/seasonality.py",
    # "scripts/ood_traffic/tcn/slope_inc.py",
    # "scripts/ood_traffic/tcn/slope_dec.py",
    # "scripts/ood_traffic/tcn/lin_dec.py",
    "scripts/ood_traffic/tcn/trend_str_dec.py",
    # "scripts/ood_traffic/tcn/all.py",

    # "scripts/ood_traffic/transformer/seasonality.py",
    # "scripts/ood_traffic/transformer/slope_inc.py",
    # "scripts/ood_traffic/transformer/slope_dec.py",
    # "scripts/ood_traffic/transformer/lin_dec.py",
    "scripts/ood_traffic/transformer/trend_str_dec.py",
    # "scripts/ood_traffic/transformer/all.py",
]

for scripts in scripts:
    subprocess.run(["python", scripts])


configs = [
    # "experiments/traffic_nips/nbeats_g_gen_seas_dec/config.yaml",
    # "experiments/traffic_nips/nbeats_g_gen_slope_inc/config.yaml",
    # "experiments/traffic_nips/nbeats_g_gen_slope_dec/config.yaml",
    # "experiments/traffic_nips/nbeats_g_gen_lin_dec/config.yaml",
    "experiments/traffic_nips/nbeats_g_gen_trend_str_dec/config.yaml",
    # "experiments/traffic_nips/nbeats_g_gen_all/config.yaml",

    # "experiments/traffic_nips/transformer_gen_seas_dec/config.yaml",
    # "experiments/traffic_nips/transformer_gen_slope_inc/config.yaml",
    # "experiments/traffic_nips/transformer_gen_slope_dec/config.yaml",
    # "experiments/traffic_nips/transformer_gen_lin_dec/config.yaml",
    "experiments/traffic_nips/transformer_gen_trend_str_dec/config.yaml",
    # "experiments/traffic_nips/transformer_gen_all/config.yaml",

    # "experiments/traffic_nips/feedforward_gen_seas_dec/config.yaml",
    # "experiments/traffic_nips/feedforward_gen_slope_inc/config.yaml",
    # "experiments/traffic_nips/feedforward_gen_slope_dec/config.yaml",
    # "experiments/traffic_nips/feedforward_gen_lin_dec/config.yaml",
    "experiments/traffic_nips/feedforward_gen_trend_str_dec/config.yaml",
    # "experiments/traffic_nips/feedforward_gen_all/config.yaml",

    # "experiments/traffic_nips/tcn_gen_seas_dec/config.yaml",
    # "experiments/traffic_nips/tcn_gen_slope_inc/config.yaml",
    # "experiments/traffic_nips/tcn_gen_slope_dec/config.yaml",
    # "experiments/traffic_nips/tcn_gen_lin_dec/config.yaml",
    "experiments/traffic_nips/tcn_gen_trend_str_dec/config.yaml",
    # "experiments/traffic_nips/tcn_gen_all/config.yaml",

    # "experiments/traffic_nips/seq2seq_gen_seas_dec/config.yaml",
    # "experiments/traffic_nips/seq2seq_gen_slope_inc/config.yaml",
    # "experiments/traffic_nips/seq2seq_gen_slope_dec/config.yaml",
    # "experiments/traffic_nips/seq2seq_gen_lin_dec/config.yaml",
    "experiments/traffic_nips/seq2seq_gen_trend_str_dec/config.yaml",
    # "experiments/traffic_nips/seq2seq_gen_all/config.yaml",
]


for config in configs:
    subprocess.run(["python", "scripts/evaluate.py", config])
