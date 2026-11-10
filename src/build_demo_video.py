#!/usr/bin/env python3
"""Build a focused 60-second captioned demo video from project artifacts."""
from pathlib import Path
import json
import subprocess
import textwrap
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
OUT = ROOT.parent / "Fixed_Demo_Video"
OUT.mkdir(parents=True, exist_ok=True)
SLIDES = OUT / "slides"
SLIDES.mkdir(exist_ok=True)

metrics = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font = ImageFont.truetype(font_path, 28)
small = ImageFont.truetype(font_path, 22)
bold = ImageFont.truetype(bold_path, 44)
small_bold = ImageFont.truetype(bold_path, 26)
W, H = 1280, 720
NAVY = (17, 35, 62)
BLUE = (41, 104, 181)
ORANGE = (229, 126, 34)
LIGHT = (245, 248, 252)
DARK = (32, 43, 56)


def draw_wrapped(draw, xy, text, width, fill=DARK, fnt=font, spacing=8):
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    draw.multiline_text(xy, "\n".join(lines), font=fnt, fill=fill, spacing=spacing)


def slide(title, subtitle=""):
    im = Image.new("RGB", (W, H), LIGHT)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 112), fill=NAVY)
    d.text((48, 28), title, font=bold, fill="white")
    if subtitle:
        d.text((50, 132), subtitle, font=small, fill=(75, 88, 106))
    return im, d

slides = []
# 1
im, d = slide("Customer Churn Prediction Pipeline", "Machine-learning project | Telco Customer Churn educational dataset")
d.text((70, 235), "Goal", font=small_bold, fill=BLUE)
draw_wrapped(d, (70, 280), "Prioritize customers who are most likely to cancel so a retention team can focus limited outreach capacity.", 43, fnt=bold, spacing=12)
d.rounded_rectangle((760, 245, 1160, 520), radius=20, fill=(226, 237, 250), outline=BLUE, width=4)
d.text((805, 295), "7,043 rows", font=bold, fill=NAVY)
d.text((805, 370), "26.5% churn", font=bold, fill=ORANGE)
d.text((805, 445), "21 source columns", font=small_bold, fill=DARK)
slides.append(im)
# 2
im, d = slide("EDA and preprocessing", "Data quality and segment-level churn analysis")
chart = Image.open(FIGURES / "churn_rate_by_category.png").convert("RGB")
chart.thumbnail((650, 430))
im.paste(chart, (570, 190))
draw_wrapped(d, (65, 210), "• Converted TotalCharges to numeric\n• Replaced 11 blanks with 0 because tenure = 0\n• Removed customerID from model features\n• Added churn-rate views by contract, internet service, payment method, and tenure band", 34, fnt=font)
slides.append(im)
# 3
im, d = slide("Pipeline and model comparison", "Leakage-aware preprocessing and validation-based selection")
steps = ["Train / validation / test\n60% / 20% / 20%", "Preprocess inside\nscikit-learn Pipeline", "Compare baseline\nwith tuned Random Forest", "Select model by\nvalidation PR-AUC"]
for i, txt in enumerate(steps):
    x = 55 + i * 305
    d.rounded_rectangle((x, 215, x + 250, 445), radius=18, fill=(226, 237, 250), outline=BLUE, width=3)
    draw_wrapped(d, (x + 22, 270), txt, 15, fill=NAVY, fnt=small_bold, spacing=10)
    if i < 3:
        d.text((x + 258, 315), "→", font=bold, fill=ORANGE)
d.text((65, 520), "Logistic Regression reference model: validation PR-AUC 0.678", font=small_bold, fill=BLUE)
d.text((65, 570), "Tuned Random Forest validation PR-AUC: 0.673", font=small_bold, fill=ORANGE)
slides.append(im)
# 4
im, d = slide("Threshold choice and test results", "The threshold is selected on validation, then test is used once")
metric_text = (f"Threshold: {metrics['threshold']:.4f}\n"
               f"Precision: {metrics['precision']:.3f}\n"
               f"Recall: {metrics['recall']:.3f}\n"
               f"F1: {metrics['f1']:.3f}\n"
               f"PR-AUC: {metrics['pr_auc_average_precision']:.3f}\n"
               f"ROC-AUC: {metrics['roc_auc']:.3f}")
d.rounded_rectangle((70, 200, 570, 570), radius=20, fill=(226, 237, 250), outline=BLUE, width=3)
draw_wrapped(d, (110, 235), metric_text, 24, fill=NAVY, fnt=small_bold, spacing=14)
cm = Image.open(FIGURES / "test_confusion_matrix.png").convert("RGB")
cm.thumbnail((570, 400))
im.paste(cm, (650, 190))
d.text((690, 595), "Validation target: 70% | Test recall: 66%", font=small_bold, fill=ORANGE)
slides.append(im)
# 5
im, d = slide("What the model means for the business", "Use predictions to prioritize outreach—not to make automatic adverse decisions")
draw_wrapped(d, (80, 220), f"At this threshold, the holdout flags {metrics['flagged_customers_on_test']} customers.\n\nIt correctly identifies 247 churners, while missing {metrics['missed_churners_on_test']} churners.\n\nA real retention workflow should attach expected customer value and intervention cost to these outcomes, then recalibrate the threshold on current company data.", 42, fnt=small_bold, spacing=14)
slides.append(im)
# 6
im, d = slide("Explainability and reproducibility", "Complete artifacts are included for review")
imp = Image.open(FIGURES / "permutation_importance.png").convert("RGB")
imp.thumbnail((570, 410))
im.paste(imp, (60, 180))
draw_wrapped(d, (680, 205), "Included:\n• Reproducible training script\n• Colab notebook\n• Saved model and threshold\n• English and Arabic reports\n• Power BI CSV exports\n• EDA and permutation-importance charts\n\nPermutation importance describes predictive dependence; it does not prove causality.", 30, fnt=small_bold, spacing=10)
slides.append(im)

for i, im in enumerate(slides, 1):
    im.save(SLIDES / f"slide_{i:02d}.png")

concat = OUT / "slides.txt"
with concat.open("w", encoding="utf-8") as f:
    for i in range(1, len(slides) + 1):
        f.write(f"file '{(SLIDES / f'slide_{i:02d}.png').as_posix()}'\n")
        f.write("duration 10\n")
    f.write(f"file '{(SLIDES / 'slide_06.png').as_posix()}'\n")

video = ROOT.parent / "Videos" / "Customer_Churn_Demo_Fixed.mp4"
video.parent.mkdir(exist_ok=True)
subprocess.run([
    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
    "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
    "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(video)
