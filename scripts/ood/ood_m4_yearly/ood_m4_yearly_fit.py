import subprocess


scripts = [
    "scripts/ood_m4_yearly/nbeats/slope_inc.py",
    "scripts/ood_m4_yearly/nbeats/slope_dec.py",
    "scripts/ood_m4_yearly/nbeats/lin_dec.py",

    "scripts/ood_m4_yearly/feedforward/slope_inc.py",
    "scripts/ood_m4_yearly/feedforward/slope_dec.py",
    "scripts/ood_m4_yearly/feedforward/lin_dec.py",

    "scripts/ood_m4_yearly/seq2seq/slope_inc.py",
    "scripts/ood_m4_yearly/seq2seq/slope_dec.py",
    "scripts/ood_m4_yearly/seq2seq/lin_dec.py",

    "scripts/ood_m4_yearly/tcn/slope_inc.py",
    "scripts/ood_m4_yearly/tcn/slope_dec.py",
    "scripts/ood_m4_yearly/tcn/lin_dec.py",

    "scripts/ood_m4_yearly/transformer/slope_inc.py",
    "scripts/ood_m4_yearly/transformer/slope_dec.py",
    "scripts/ood_m4_yearly/transformer/lin_dec.py",
]

for scripts in scripts:
    subprocess.run(["python", scripts])


configs = [
    "experiments/m4_yearly/nbeats_g_gen_slope_inc/config.yaml",
    "experiments/m4_yearly/nbeats_g_gen_slope_dec/config.yaml",
    "experiments/m4_yearly/nbeats_g_gen_lin_dec/config.yaml",

    "experiments/m4_yearly/seq2seq_gen_slope_inc/config.yaml",
    "experiments/m4_yearly/seq2seq_gen_slope_dec/config.yaml",
    "experiments/m4_yearly/seq2seq_gen_lin_dec/config.yaml",

    "experiments/m4_yearly/feedforward_gen_slope_inc/config.yaml",
    "experiments/m4_yearly/feedforward_gen_slope_dec/config.yaml",
    "experiments/m4_yearly/feedforward_gen_lin_dec/config.yaml",

    "experiments/m4_yearly/tcn_gen_slope_inc/config.yaml",
    "experiments/m4_yearly/tcn_gen_slope_dec/config.yaml",
    "experiments/m4_yearly/tcn_gen_lin_dec/config.yaml",

    "experiments/m4_yearly/transformer_gen_slope_inc/config.yaml",
    "experiments/m4_yearly/transformer_gen_slope_dec/config.yaml",
    "experiments/m4_yearly/transformer_gen_lin_dec/config.yaml",
]


for config in configs:
    subprocess.run(["python", "scripts/evaluate.py", config])
