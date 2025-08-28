import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Example dataframe
df = pd.read_csv("telecom_processed.csv")

# PDF generation
def generate_pdf(df, filename="recommendations.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    normal_style = ParagraphStyle(
        "Normal",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.white,
        backColor=colors.HexColor("#1717a1"),
        leading=14,
        borderPadding=(8, 8, 8),
        spaceAfter=10,
    )
    
    warning_style = ParagraphStyle(
        "Warning",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.black,
        backColor=colors.HexColor("#ffcccc"),
        leading=14,
        borderPadding=(8, 8, 8),
        spaceAfter=10,
    )
    
    # Add recommendations
    for _, row in df.iterrows():
        style = warning_style if "⚠️" in row["recommendation"] or "🚨" in row["recommendation"] else normal_style
        text = f"<b>Tower {row['tower_id']} ({row['operator']} - {row['network_type']})</b><br/>{row['recommendation']}"
        elements.append(Paragraph(text, style))
        elements.append(Spacer(1, 6))
    
    doc.build(elements)

# Run function
generate_pdf(df)
print("✅ PDF saved as recommendations.pdf")
