# reportes/reportes_pdf.py
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def generate_pdf_report(
    output_pdf_path: str,
    image_paths: list[str],
    title: str = "Informe de Resultados",
    metadata: dict | None = None,
):
    """
    Crea un PDF con:
      - Portada (título + metadatos clave)
      - 1 página por imagen de `image_paths` (se escala automáticamente)
    No requiere reportlab; usa matplotlib.PdfPages.
    """
    _ensure_dir(output_pdf_path)
    meta = metadata or {}

    with PdfPages(output_pdf_path) as pdf:
        # ===== Portada =====
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 aprox.
        fig.suptitle(title, fontsize=18, y=0.95)
        txt_lines = [
            f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        # Metadatos clave
        for k in ["simbolo", "timeframe", "cantidad_datos", "modo", "outdir"]:
            if k in meta:
                txt_lines.append(f"{k}: {meta[k]}")
        txt = "\n".join(txt_lines)
        plt.text(0.05, 0.85, txt, fontsize=11, va="top")

        # Pequeña leyenda de qué contiene el PDF
        legend = (
            "Este informe incluye:\n"
            "• Gráfico nivel: real vs pronósticos\n"
            "• Gráficos de error rolling por modelo\n"
            "• Gráfico volatilidad: RV vs HAR vs GARCH"
        )
        plt.text(0.05, 0.55, legend, fontsize=11, va="top")
        plt.axis("off")
        pdf.savefig(fig); plt.close(fig)

        # ===== Páginas por imagen =====
        for p in image_paths:
            if not os.path.isfile(p):
                continue
            img = plt.imread(p)
            h, w = img.shape[:2]
            # Mantener proporción en página A4
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
            ax.imshow(img)
            ax.set_title(os.path.basename(p), fontsize=12)
            ax.axis("off")
            pdf.savefig(fig); plt.close(fig)
