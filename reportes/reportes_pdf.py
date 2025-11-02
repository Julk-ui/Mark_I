# reportes_pdf.py
# Versión refactorizada para reportes EDA de alta calidad.

import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os

# ---------------------------------------------------------------------
# MEJORA 1: Clase de Reporte Robusta (Maneja Header/Footer)
# ---------------------------------------------------------------------

class PDF(FPDF):
    """
    Clase personalizada que hereda de FPDF para crear cabeceras
    y pies de página automáticamente en cada página.
    """
    def header(self):
        # Logo (opcional, si tiene uno)
        # self.image('logo.png', 10, 8, 33)
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Reporte de Análisis Exploratorio de Datos (EDA)', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 5, f'Activo: EUR/USD | Fecha de generación: {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'C')
        self.ln(10) # Salto de línea

    def footer(self):
        self.set_y(-15) # Posición 1.5 cm desde el fondo
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', 0, 0, 'C')

    def chapter_title(self, title):
        """Crea un título de sección estandarizado."""
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230) # Gris claro
        self.cell(0, 8, title, 0, 1, 'L', fill=True)
        self.ln(4)

    def chapter_body(self, body_text):
        """Inserta texto de párrafo estándar."""
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body_text)
        self.ln()

# ---------------------------------------------------------------------
# MEJORA 2: Función para Formatear Tablas (Pandas -> PDF)
# ---------------------------------------------------------------------

def _add_df_to_pdf(pdf: PDF, df: pd.DataFrame):
    """
    Helper para dibujar un DataFrame de Pandas como una tabla
    estilizada en el PDF.
    """
    # Guardar estado de fuente
    pdf.set_font('Arial', 'B', 9) # Fuente Negrita para Header
    line_height = pdf.font_size * 1.5
    
    # Calcular ancho de columnas (simple, se puede mejorar)
    col_width = (pdf.w - pdf.l_margin - pdf.r_margin) / (len(df.columns) + 1) # +1 para el índice
    
    # --- Cabecera de la Tabla ---
    # Índice
    pdf.cell(col_width, line_height, df.index.name or 'Index', border=1, ln=0, align='C', fill=True)
    # Columnas
    for col in df.columns:
        pdf.cell(col_width, line_height, str(col), border=1, ln=0, align='C', fill=True)
    pdf.ln(line_height)

    # --- Cuerpo de la Tabla ---
    pdf.set_font('Arial', '', 9) # Fuente normal para datos
    
    for i in range(len(df)):
        # Índice
        pdf.cell(col_width, line_height, str(df.index[i]), border=1, ln=0, align='L')
        # Celdas de datos
        for col in df.columns:
            cell_text = str(df.iloc[i][col])
            # Acortar texto si es muy largo para la celda
            if len(cell_text) > 30: 
                cell_text = cell_text[:27] + "..."
            pdf.cell(col_width, line_height, cell_text, border=1, ln=0, align='L')
        pdf.ln(line_height)
    
    pdf.ln(5) # Espacio después de la tabla

# ---------------------------------------------------------------------
# MEJORA 3: Función para Incrustar Imágenes
# ---------------------------------------------------------------------

def _add_image_to_pdf(pdf: PDF, img_path: str, title: str = ""):
    """
    Helper para insertar una imagen, centrada y respetando
    el ancho de la página.
    """
    if not os.path.exists(img_path):
        pdf.set_font('Arial', 'I', 10)
        pdf.set_text_color(255, 0, 0) # Rojo
        pdf.cell(0, 10, f"[ERROR] No se encontró la imagen: {img_path}", 0, 1)
        pdf.set_text_color(0, 0, 0) # Reset color
        return

    if title:
        pdf.set_font('Arial', 'I', 11)
        pdf.cell(0, 8, title, 0, 1, 'C')

    # Ancho de página disponible
    page_width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.image(img_path, w=page_width * 0.9, x=pdf.l_margin + (page_width * 0.05))
    pdf.ln(5)

# =====================================================================
# FUNCIÓN PRINCIPAL (MODIFICADA)
# =Telemetria (EDA)
# =====================================================================

def generar_reporte_eda_pdf(data_dict: dict, ruta_salida: str = "EDA_informe.pdf"):
    """
    Genera el reporte EDA en PDF usando la nueva clase y helpers.

    Args:
        data_dict (dict): Un diccionario que debe contener:
            'narrativa': (dict) Textos descriptivos.
            'tablas': (dict) DataFrames de Pandas.
            'graficos': (dict) Rutas a los archivos .png.
        ruta_salida (str): Path donde se guardará el PDF.
    """
    
    print(f"📄 Iniciando generación de Reporte EDA PDF en: {ruta_salida}")

    pdf = PDF()
    pdf.alias_nb_pages() # Habilita el conteo de páginas {nb}
    pdf.add_page()

    # --- 1. Resumen y Calidad de Datos ---
    pdf.chapter_title("1. Resumen y Calidad de Datos")
    
    if 'narrativa' in data_dict and 'resumen' in data_dict['narrativa']:
        pdf.chapter_body(data_dict['narrativa']['resumen'])
    
    if 'tablas' in data_dict and 'data_quality' in data_dict['tablas']:
        _add_df_to_pdf(pdf, data_dict['tablas']['data_quality'])
        
    if 'graficos' in data_dict and 'missing_data' in data_dict['graficos']:
        _add_image_to_pdf(pdf, data_dict['graficos']['missing_data'], "Visualización de Datos Faltantes")

    # --- 2. Análisis de Estacionariedad y Retornos ---
    pdf.add_page()
    pdf.chapter_title("2. Análisis de Estacionariedad y Retornos")

    if 'narrativa' in data_dict and 'estacionariedad' in data_dict['narrativa']:
        pdf.chapter_body(data_dict['narrativa']['estacionariedad'])
        
    if 'tablas' in data_dict and 'stationarity_test' in data_dict['tablas']:
        _add_df_to_pdf(pdf, data_dict['tablas']['stationarity_test'])

    if 'graficos' in data_dict and 'precio_close' in data_dict['graficos']:
        _add_image_to_pdf(pdf, data_dict['graficos']['precio_close'], "Serie de Precios (Close)")
        
    if 'graficos' in data_dict and 'retornos' in data_dict['graficos']:
        _add_image_to_pdf(pdf, data_dict['graficos']['retornos'], "Serie de Retornos y Clústeres de Volatilidad")

    # --- 3. Distribución y Autocorrelación ---
    pdf.add_page()
    pdf.chapter_title("3. Distribución y Autocorrelación")
    
    if 'narrativa' in data_dict and 'distribucion' in data_dict['narrativa']:
        pdf.chapter_body(data_dict['narrativa']['distribucion'])

    if 'graficos' in data_dict and 'histograma_retornos' in data_dict['graficos']:
        _add_image_to_pdf(pdf, data_dict['graficos']['histograma_retornos'], "Histograma y QQ-Plot de Retornos")

    if 'graficos' in data_dict and 'acf_pacf' in data_dict['graficos']:
        _add_image_to_pdf(pdf, data_dict['graficos']['acf_pacf'], "Análisis de Autocorrelación (ACF/PACF)")

    # --- 4. Selección de Modelos (ARIMA/SARIMA) ---
    pdf.add_page()
    pdf.chapter_title("4. Candidatos de Modelos")
    
    if 'narrativa' in data_dict and 'modelos' in data_dict['narrativa']:
        pdf.chapter_body(data_dict['narrativa']['modelos'])
        
    if 'tablas' in data_dict and 'arima_candidates' in data_dict['tablas']:
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 8, "Candidatos ARIMA", 0, 1, 'L')
        _add_df_to_pdf(pdf, data_dict['tablas']['arima_candidates'])
        
    if 'tablas' in data_dict and 'sarima_candidates' in data_dict['tablas']:
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 8, "Candidatos SARIMA", 0, 1, 'L')
        _add_df_to_pdf(pdf, data_dict['tablas']['sarima_candidates'])

    # --- Guardar el PDF ---
    try:
        pdf.output(ruta_salida)
        print(f"✅ Reporte EDA PDF generado exitosamente.")
    except Exception as e:
        print(f"❌ Error al guardar el PDF: {e}")

# =====================================================================
# OTRAS FUNCIONES (Ej. Reporte de Inversión)
# =====================================================================

def generar_reporte_inversion_pdf(predicciones, senal, capital, operacion, umbral, ruta):
    """
    Esta es la función que genera su OTRO reporte.
    LA DEJAMOS INTÁCTA tal como usted la tenga.
    """
    # ... (Su código original para el reporte de inversión va aquí)
    # Ejemplo de cómo podría ser:
    print(f"ℹ️ (Simulando) Generando reporte de INVERSIÓN en {ruta}...")
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Reporte de Recomendación de Inversión", ln=True, align='C')
        pdf.multi_cell(0, 10, f"Señal: {senal}\nCapital: {capital}\nUmbral: {umbral}\nOperación: {operacion}")
        
        # Asumimos que el gráfico de predicción existe
        if os.path.exists('outputs/grafico_prediccion.png'):
             pdf.image('outputs/grafico_prediccion.png', w=190)
             
        pdf.output(ruta)
        print(f"✅ Reporte de INVERSIÓN generado.")
    except Exception as e:
        print(f"❌ Error en reporte de INVERSIÓN: {e}")


# =====================================================================
# Ejemplo de uso (para probar este script)
# =====================================================================
if __name__ == "__main__":
    
    # --- Prueba del Reporte de Inversión (simulado) ---
    generar_reporte_inversion_pdf(
        predicciones=None,
        senal="comprar",
        capital=100.50,
        operacion={'tipo': 'compra', 'resultado': 'simulada'},
        umbral=0.0003,
        ruta="outputs/TEST_Reporte_Inversion.pdf"
    )

    # --- Prueba del Reporte EDA (simulado) ---
    # `eda_crispdm.py` debería construir un diccionario así:
    
    # 1. Crear datos falsos de tablas
    df_quality = pd.DataFrame({
        'Nulos': [0, 0, 0], 
        'Duplicados': [0, 0, 0],
        'Outliers (IQR)': [10, 12, 8]
    }, index=['Open', 'High', 'Low'])
    
    df_stationarity = pd.DataFrame({
        'ADF Statistic': [-1.2, -9.8],
        'p-value': [0.85, 0.001],
        'Estacionaria': [False, True]
    }, index=['Precio (Close)', 'Retornos'])

    # 2. Crear imágenes falsas (placeholders)
    # (Asegúrese de que existan o comente estas líneas)
    img_dir = "outputs/eda_graficos_falsos"
    os.makedirs(img_dir, exist_ok=True)
    
    # Creamos imágenes dummy para la prueba
    pd.Series(range(100)).plot().get_figure().savefig(f"{img_dir}/dummy_plot.png")

    test_data_dict = {
        'narrativa': {
            'resumen': "Análisis inicial de 10,000 velas M15. No se encontraron nulos ni duplicados, pero se detectaron outliers.",
            'estacionariedad': "La prueba ADF confirma que el precio no es estacionario (p>0.05), pero los retornos sí lo son (p<0.05).",
            'distribucion': "Los retornos muestran colas pesadas (leptokurtosis) y clústeres de volatilidad.",
            'modelos': "Basado en ACF/PACF, se sugieren modelos ARIMA/SARIMA."
        },
        'tablas': {
            'data_quality': df_quality,
            'stationarity_test': df_stationarity,
            'arima_candidates': pd.DataFrame({'p': [1,2], 'd': [1,1], 'q': [1,2], 'bic': [100, 102]}),
            'sarima_candidates': pd.DataFrame({'p': [1], 'd': [1], 'q': [1], 's': [12], 'bic': [90]})
        },
        'graficos': {
            'missing_data': f"{img_dir}/dummy_plot.png",
            'precio_close': f"{img_dir}/dummy_plot.png",
            'retornos': f"{img_dir}/dummy_plot.png",
            'histograma_retornos': f"{img_dir}/dummy_plot.png",
            'acf_pacf': f"{img_dir}/dummy_plot.png"
        }
    }
    
    generar_reporte_eda_pdf(test_data_dict, ruta_salida="outputs/TEST_EDA_Informe_Mejorado.pdf")