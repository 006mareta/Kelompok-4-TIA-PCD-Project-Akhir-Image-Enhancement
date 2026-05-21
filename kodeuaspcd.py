"""
===================================================
  Dental X-Ray — Gaussian Filter + Histogram Equalization
  Segmen 1 : Citra Asli
  Segmen 2 : Noisy + Hasil Kernel 3×3 · 5×5 · 7×7  (2×2)
  Segmen 3 : Evaluasi Metrik PSNR · MSE · SSIM
  (Mode Batch Otomatis + Rata-rata Metrik + Konsisten/Sama)
===================================================
"""

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim


# ─────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────
DATASET_PATH = "Dental X-ray"
OUTPUT_PATH  = "Hasil_Denoising"  # Folder tempat menyimpan hasil otomatis
NOISE_MEAN   = 0
NOISE_STD    = 25
RANDOM_SEED  = 42

KERNELS = {
    "3×3": (3, 3),
    "5×5": (5, 5),
    "7×7": (7, 7),
}

STYLE = {
    "bg"     : "#0f1117",
    "panel"  : "#1a1d27",
    "border" : "#2a2d3e",
    "accent" : "#4fc3f7",
    "text"   : "#e0e6f0",
    "subtext": "#7a84a0",
    "good"   : "#69f0ae",
    "warn"   : "#ffd740",
    "neutral": "#ce93d8",
    "bars"   : ["#4fc3f7", "#69f0ae", "#ffd740"],
    "ker_col": ["#4fc3f7", "#69f0ae", "#ffd740"],
}

plt.rcParams.update({
    "figure.facecolor": STYLE["bg"],
    "axes.facecolor"  : STYLE["panel"],
    "text.color"      : STYLE["text"],
    "axes.labelcolor" : STYLE["text"],
    "xtick.color"     : STYLE["subtext"],
    "ytick.color"     : STYLE["subtext"],
    "axes.edgecolor"  : STYLE["border"],
    "font.family"     : "monospace",
    "font.size"       : 9,
})


# ─────────────────────────────────────────────────────────
# PEMROSESAN
# ─────────────────────────────────────────────────────────

def load_image_paths(dataset_path: str) -> list[str]:
    extensions = (".png", ".jpg", ".jpeg")
    paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(dataset_path)
        for f in files
        if f.lower().endswith(extensions)
    ]
    if not paths:
        raise FileNotFoundError(f"Tidak ada gambar di: {dataset_path!r}")
    return paths


def add_gaussian_noise(image: np.ndarray, mean: float = 0, std: float = 25) -> np.ndarray:
    """Tambahkan Gaussian noise dengan mengunci seed agar polanya selalu konsisten."""
    np.random.seed(RANDOM_SEED)  # <── DIKUNCI DI SINI: Agar pola noise selalu sama per eksekusi
    img_f = image.astype(np.float32)
    noise = np.random.normal(mean, std, img_f.shape).astype(np.float32)
    return np.clip(img_f + noise, 0, 255).astype(np.uint8)


def process(noisy: np.ndarray, ksize: tuple) -> np.ndarray:
    """Gaussian Blur → Histogram Equalization."""
    blurred  = cv2.GaussianBlur(noisy, ksize, sigmaX=1)
    enhanced = cv2.equalizeHist(blurred)
    return enhanced


def compute_metrics(original: np.ndarray, enhanced: np.ndarray) -> dict:
    return {
        "psnr": psnr(original, enhanced, data_range=255),
        "mse" : float(mse(original, enhanced)),
        "ssim": ssim(original, enhanced, data_range=255),
    }


# ─────────────────────────────────────────────────────────
# HELPER VISUALISASI
# ─────────────────────────────────────────────────────────

def _frame(ax, color: str, lw: float = 2.0) -> None:
    """Border berwarna di sekeliling axes."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(color)
        sp.set_linewidth(lw)


def _imshow(ax, img: np.ndarray, title: str, title_color: str,
            badge: str = None, badge_color: str = None,
            metrics: dict = None) -> None:
    """Tampilkan citra dengan judul, badge, dan opsional overlay metrik."""
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, color=title_color, fontsize=10,
                 fontweight="bold", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])

    if badge:
        ax.text(0.03, 0.97, badge, transform=ax.transAxes,
                ha="left", va="top", fontsize=8, fontweight="bold",
                color="#0f1117",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=badge_color or title_color,
                          edgecolor="none", alpha=0.92))

    if metrics:
        txt = (f"PSNR : {metrics['psnr']:.2f} dB\n"
               f"MSE  : {metrics['mse']:.2f}\n"
               f"SSIM : {metrics['ssim']:.4f}")
        ax.text(0.97, 0.03, txt, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5,
                color=STYLE["text"],
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="#0d1020",
                          edgecolor=STYLE["border"], alpha=0.90),
                linespacing=1.65)


# ─────────────────────────────────────────────────────────
# SEGMEN VISUALISASI & SAVE
# ─────────────────────────────────────────────────────────

def segmen1_original(original: np.ndarray, fname: str, save_prefix: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    fig.patch.set_facecolor(STYLE["bg"])

    _imshow(ax, original,
            title="Citra Asli (Original)",
            title_color=STYLE["good"],
            badge="ORIGINAL", badge_color=STYLE["good"])
    _frame(ax, STYLE["good"], lw=2.5)

    fig.suptitle(
        "SEGMEN 1  ·  Input Citra Dental X-Ray",
        color=STYLE["text"], fontsize=13, fontweight="bold", y=1.01,
    )
    fig.text(0.5, -0.01, f"File : {fname}",
             ha="center", color=STYLE["subtext"], fontsize=8)
    plt.tight_layout()
    
    plt.savefig(f"{save_prefix}_segmen1.png", dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)


def segmen2_results(noisy: np.ndarray, results: dict, save_prefix: str) -> None:
    names    = list(results.keys())
    ker_col  = STYLE["ker_col"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.06})
    fig.patch.set_facecolor(STYLE["bg"])

    _imshow(axes[0, 0], noisy,
            title=f"Citra Noisy  (μ={NOISE_MEAN}, σ={NOISE_STD})",
            title_color=STYLE["warn"],
            badge="NOISY", badge_color=STYLE["warn"])
    _frame(axes[0, 0], STYLE["warn"])

    positions = [(0, 1), (1, 0), (1, 1)]
    for idx, (name, (row, col)) in enumerate(zip(names, positions)):
        r = results[name]
        _imshow(
            axes[row, col],
            r["img"],
            title=f"Gaussian {name}  +  Histogram Equalization",
            title_color=ker_col[idx],
            badge=f"KERNEL {name}", badge_color=ker_col[idx],
            metrics=r["metrics"],
        )
        _frame(axes[row, col], ker_col[idx])

    fig.suptitle(
        "SEGMEN 2  ·  Gaussian Filter + Histogram Equalization  ·  Kernel 3×3 · 5×5 · 7×7",
        color=STYLE["text"], fontsize=12, fontweight="bold", y=1.005,
    )
    fig.text(
        0.5, -0.008,
        f"Noise : Gaussian  μ={NOISE_MEAN}  σ={NOISE_STD}  "
        f"·  Filter diterapkan pada citra noisy",
        ha="center", color=STYLE["subtext"], fontsize=8,
    )
    plt.tight_layout()
    
    plt.savefig(f"{save_prefix}_segmen2.png", dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)


def segmen3_metrics(results: dict, save_prefix: str) -> None:
    names   = list(results.keys())
    colors  = STYLE["bars"]
    metrics = [
        ("psnr", "PSNR (dB)",  "↑ Lebih tinggi = lebih baik"),
        ("mse",  "MSE",        "↓ Lebih rendah = lebih baik"),
        ("ssim", "SSIM",       "↑ Lebih tinggi = lebih baik"),
    ]

    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor(STYLE["bg"])

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.60, wspace=0.35,
                           top=0.88, bottom=0.15,
                           left=0.06, right=0.97)

    for j, (key, label, hint) in enumerate(metrics):
        ax   = fig.add_subplot(gs[0, j])
        vals = [results[n]["metrics"][key] for n in names]
        x    = np.arange(len(names))
        ymax = max(vals)

        bars = ax.bar(x, vals, width=0.48, color=colors,
                      edgecolor=STYLE["border"], linewidth=0.9, zorder=3)

        for bar, val in zip(bars, vals):
            fmt = f"{val:.4f}" if key == "ssim" else f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.03,
                    fmt, ha="center", va="bottom",
                    color=STYLE["text"], fontsize=9, fontweight="bold")

        ax.set_title(label, color=STYLE["accent"],
                     fontsize=11, fontweight="bold", pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Kernel\n{n}" for n in names], fontsize=9)
        ax.set_ylim(0, ymax * 1.28)
        ax.tick_params(left=False)
        ax.set_yticklabels([])
        ax.grid(axis="y", color=STYLE["border"], linewidth=0.8, zorder=0)
        ax.spines[:].set_visible(False)
        ax.text(0.5, -0.24, hint, transform=ax.transAxes,
                ha="center", color=STYLE["subtext"],
                fontsize=7.5, style="italic")

    ax_t = fig.add_subplot(gs[1, :])
    ax_t.set_facecolor(STYLE["bg"])
    ax_t.axis("off")

    best     = max(names, key=lambda k: results[k]["metrics"]["ssim"])
    ranks    = sorted(names, key=lambda k: results[k]["metrics"]["ssim"], reverse=True)
    rank_map = {n: i + 1 for i, n in enumerate(ranks)}

    col_labels = ["Kernel", "PSNR (dB)", "MSE", "SSIM", "Ranking SSIM"]
    rows = []
    for n in names:
        m   = results[n]["metrics"]
        tag = f"#{rank_map[n]}" + (" ✔ Terbaik" if n == best else "")
        rows.append([n, f"{m['psnr']:.2f}", f"{m['mse']:.2f}",
                     f"{m['ssim']:.4f}", tag])

    tbl = ax_t.table(cellText=rows, colLabels=col_labels,
                     cellLoc="center", loc="center",
                     bbox=[0.05, 0.0, 0.90, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(STYLE["border"])
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#1e2235")
            cell.set_text_props(color=STYLE["accent"], fontweight="bold")
        elif names[row - 1] == best:
            cell.set_facecolor("#1a2e22")
            cell.set_text_props(color=STYLE["good"])
        else:
            cell.set_facecolor(STYLE["panel"])
            cell.set_text_props(color=STYLE["text"])

    fig.suptitle(
        "SEGMEN 3  ·  Evaluasi Metrik — PSNR · MSE · SSIM",
        color=STYLE["text"], fontsize=12, fontweight="bold",
    )
    
    plt.savefig(f"{save_prefix}_segmen3.png", dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)


# ─────────────────────────────────────────────────────────
# PRINT RINGKASAN DATA KESELURUHAN (RATA-RATA)
# ─────────────────────────────────────────────────────────

def print_final_summary(summary_metrics: dict, total_processed: int) -> None:
    W = 65
    print("\n" + "═"*W)
    print(f"  REKAPITULASI RATA-RATA METRIK (TOTAL: {total_processed} CITRA DENTAL X-RAY)")
    print("═"*W)
    print(f"  {'Kernel':<8} {'Rerata PSNR (dB)':>18} {'Rerata MSE':>14} {'Rerata SSIM':>14}")
    print(f"  {'─'*61}")
    
    avg_results = {}
    for name in KERNELS.keys():
        avg_psnr = np.mean(summary_metrics[name]["psnr"])
        avg_mse  = np.mean(summary_metrics[name]["mse"])
        avg_ssim = np.mean(summary_metrics[name]["ssim"])
        avg_results[name] = {"psnr": avg_psnr, "mse": avg_mse, "ssim": avg_ssim}
        
    best_avg_kernel = max(avg_results, key=lambda k: avg_results[k]["ssim"])
    
    for name, avg in avg_results.items():
        marker = " ✔" if name == best_avg_kernel else "  "
        print(f"{marker} {name:<8} {avg['psnr']:>18.2f} {avg['mse']:>14.2f} {avg['ssim']:>14.4f}")
        
    print(f"  {'─'*61}")
    print(f"  ✔ KESIMPULAN BATCH: Kernel terbaik secara umum adalah {best_avg_kernel}")
    print("═"*W + "\n")


# ─────────────────────────────────────────────────────────
# MAIN PROGRAM (KONSISTEN / DETERMINISTIK)
# ─────────────────────────────────────────────────────────

def main() -> None:
    # Mengunci global seed bawaan python dan numpy
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print(f"Membuat folder penyimpanan baru: '{OUTPUT_PATH}'")

    try:
        image_paths = load_image_paths(DATASET_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    total_images = len(image_paths)
    print(f"Total gambar ditemukan : {total_images}")
    
    target_proses = min(50, total_images)
    print(f"Memproses {target_proses} citra secara berurutan & konsisten...\n")

    # MODIFIKASI: Mengurutkan file secara alfabetis (Bukan random shuffle)
    image_paths.sort()  # <── DIUBAH DI SINI: Urutan pembacaan file selalu sama

    summary_metrics = {name: {"psnr": [], "mse": [], "ssim": []} for name in KERNELS.keys()}
    citra_berhasil = 0

    for i in range(target_proses):
        img_path = image_paths[i]
        fname = os.path.basename(img_path)
        name_only, _ = os.path.splitext(fname)
        
        print(f"» [{i+1}/{target_proses}] Memproses & Menyimpan: {fname}")

        original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            print(f"   Gagal membaca file: {img_path}. Dilanjutkan ke file berikutnya.")
            continue

        # Proses noise (Hasil noise terjamin konsisten karena seed dikunci di dalam fungsi)
        noisy = add_gaussian_noise(original, NOISE_MEAN, NOISE_STD)

        results = {}
        for name, ksize in KERNELS.items():
            enhanced      = process(noisy, ksize)
            m             = compute_metrics(original, enhanced)
            results[name] = {
                "img"    : enhanced,
                "metrics": m,
            }
            summary_metrics[name]["psnr"].append(m["psnr"])
            summary_metrics[name]["mse"].append(m["mse"])
            summary_metrics[name]["ssim"].append(m["ssim"])

        citra_berhasil += 1
        save_prefix = os.path.join(OUTPUT_PATH, f"{i+1:02d}_{name_only}")

        segmen1_original(original, fname, save_prefix)
        segmen2_results(noisy, results, save_prefix)
        segmen3_metrics(results, save_prefix)
        
        plt.close('all')

    if citra_berhasil > 0:
        print_final_summary(summary_metrics, citra_berhasil)
    else:
        print("Tidak ada citra yang berhasil diproses.")

    print("═"*60)
    print(f" SELESAI! Hasil plot dan rata-rata metrik dijamin 100% konsisten di setiap run.")
    print("═"*60)


if __name__ == "__main__":
    main()