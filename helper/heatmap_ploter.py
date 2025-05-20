import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

tamil_font_path = "NotoSansTamil-VariableFont_wdth,wght.ttf" 

tamil_font = fm.FontProperties(fname=tamil_font_path)
plt.rcParams['font.family'] = tamil_font.get_name()

with open("predictions_attention.json", "r", encoding='utf-8') as f:
    data = json.load(f)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i, ax in enumerate(axes.flat):
    if i >= len(data):
        ax.axis("off")
        continue

    sample = data[i]
    attention = np.squeeze(np.array(sample["attention"]))
    input_tokens = sample["input"]
    output_tokens = sample["prediction"]

    sns.heatmap(attention, ax=ax, cmap="YlGnBu",
                xticklabels=input_tokens, yticklabels=output_tokens,
                cbar=False)

    ax.set_title(f"Sample {i+1}", fontproperties=tamil_font)
    ax.set_xlabel("Input Tokens", fontproperties=tamil_font)
    ax.set_ylabel("Output Tokens", fontproperties=tamil_font)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

    ax.set_xticklabels(input_tokens, fontproperties=tamil_font)
    ax.set_yticklabels(output_tokens, fontproperties=tamil_font)

plt.tight_layout()
plt.savefig("attention_heatmaps_YlGnBu.png", dpi=300)
