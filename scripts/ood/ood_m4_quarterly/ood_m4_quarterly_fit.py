import subprocess


scripts = [
    # "scripts/ood_m4_quarterly/feedforward/seas_inc.py",
    "scripts/ood_m4_quarterly/feedforward/lin_dec.py",
    # "scripts/ood_m4_quarterly/feedforward/seas_dec.py",
    "scripts/ood_m4_quarterly/feedforward/slope_dec.py",
    # "scripts/ood_m4_quarterly/feedforward/all.py",

    # "scripts/ood_m4_quarterly/nbeats/seas_inc.py",
    "scripts/ood_m4_quarterly/nbeats/lin_dec.py",
    # "scripts/ood_m4_quarterly/nbeats/seas_dec.py",
    "scripts/ood_m4_quarterly/nbeats/slope_dec.py",
    # "scripts/ood_m4_quarterly/nbeats/all.py",

    # "scripts/ood_m4_quarterly/seq2seq/seas_inc.py",
    "scripts/ood_m4_quarterly/seq2seq/lin_dec.py",
    # "scripts/ood_m4_quarterly/seq2seq/seas_dec.py",
    "scripts/ood_m4_quarterly/seq2seq/slope_dec.py",
    # "scripts/ood_m4_quarterly/seq2seq/all.py",

    # "scripts/ood_m4_quarterly/tcn/seas_inc.py",
    "scripts/ood_m4_quarterly/tcn/lin_dec.py",
    # "scripts/ood_m4_quarterly/tcn/seas_dec.py",
    "scripts/ood_m4_quarterly/tcn/slope_dec.py",
    # "scripts/ood_m4_quarterly/tcn/all.py",

    # "scripts/ood_m4_quarterly/transformer/seas_inc.py",
    "scripts/ood_m4_quarterly/transformer/lin_dec.py",
    # "scripts/ood_m4_quarterly/transformer/seas_dec.py",
    "scripts/ood_m4_quarterly/transformer/slope_dec.py",
    # "scripts/ood_m4_quarterly/transformer/all.py",
]

for scripts in scripts:
    subprocess.run(["python", scripts])


configs = [
    # "experiments/m4_quarterly/nbeats_g_gen_seas_dec/config.yaml",
    # "experiments/m4_quarterly/nbeats_g_gen_seas_inc/config.yaml",
    "experiments/m4_quarterly/nbeats_g_gen_slope_dec/config.yaml",
    "experiments/m4_quarterly/nbeats_g_gen_lin_dec/config.yaml",
    # "experiments/m4_quarterly/nbeats_g_gen_all/config.yaml",

    # "experiments/m4_quarterly/tcn_gen_seas_dec/config.yaml",
    # "experiments/m4_quarterly/tcn_gen_seas_inc/config.yaml",
    "experiments/m4_quarterly/tcn_gen_slope_dec/config.yaml",
    "experiments/m4_quarterly/tcn_get_lin_dec/config.yaml",
    # "experiments/m4_quarterly/tcn_gen_all/config.yaml",

    # "experiments/m4_quarterly/seq2seq_gen_seas_dec/config.yaml",
    # "experiments/m4_quarterly/seq2seq_gen_seas_inc/config.yaml",
    "experiments/m4_quarterly/seq2seq_gen_slope_dec/config.yaml",
    "experiments/m4_quarterly/seq2seq_gen_lin_dec/config.yaml",
    # "experiments/m4_quarterly/seq2seq_gen_all/config.yaml",

    # "experiments/m4_quarterly/transformer_gen_seas_dec/config.yaml",
    # "experiments/m4_quarterly/transformer_gen_seas_inc/config.yaml",
    "experiments/m4_quarterly/transformer_gen_slope_dec/config.yaml",
    "experiments/m4_quarterly/transformer_gen_lin_dec/config.yaml",
    # "experiments/m4_quarterly/transformer_gen_all/config.yaml",

    # "experiments/m4_quarterly/feedforward_gen_seas_dec/config.yaml",
    # "experiments/m4_quarterly/feedforward_gen_seas_inc/config.yaml",
    "experiments/m4_quarterly/feedforward_gen_slope_dec/config.yaml",
    "experiments/m4_quarterly/feedforward_gen_lin_dec/config.yaml",
    # "experiments/m4_quarterly/feedforward_gen_all/config.yaml",
]


for config in configs:
    subprocess.run(["python", "scripts/evaluate.py", config])
