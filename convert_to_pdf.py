"""
Script to convert HTML documentation to PDF
"""
import os
import sys

def convert_html_to_pdf():
    """Convert the HTML documentation to PDF using available methods"""
    
    html_file = r"c: \Users\hp\AI Nutrition\project_documentation.html"
    pdf_file = r"c:\Users\hp\AI Nutrition\AI_Nutrition_Project_Documentation.pdf"
    
    # Method 1: Try using weasyprint
    try:
        from weasyprint import HTML
        print("Using WeasyPrint to convert HTML to PDF...")
        HTML(filename=html_file).write_pdf(pdf_file)
        print(f"✓ PDF created successfully: {pdf_file}")
        return True
    except ImportError:
        print("WeasyPrint not installed. Trying next method...")
    except Exception as e:
        print(f"WeasyPrint error: {e}")
    
    # Method 2: Try using pdfkit (requires wkhtmltopdf)
    try:
        import pdfkit
        print("Using pdfkit to convert HTML to PDF...")
        pdfkit.from_file(html_file, pdf_file)
        print(f"✓ PDF created successfully: {pdf_file}")
        return True
    except ImportError:
        print("pdfkit not installed. Trying next method...")
    except Exception as e:
        print(f"pdfkit error: {e}")
    
    # Method 3: Try using reportlab with html
    try:
        from xhtml2pdf import pisa
        print("Using xhtml2pdf to convert HTML to PDF...")
        
        with open(html_file, 'r', encoding='utf-8') as source_file:
            source_html = source_file.read()
        
        with open(pdf_file, 'wb') as output_file:
            pisa_status = pisa.CreatePDF(source_html, dest=output_file)
        
        if not pisa_status.err:
            print(f"✓ PDF created successfully: {pdf_file}")
            return True
        else:
            print(f"xhtml2pdf error occurred")
    except ImportError:
        print("xhtml2pdf not installed.")
    except Exception as e:
        print(f"xhtml2pdf error: {e}")
    
    print("\n" + "="*60)
    print("No PDF library available. Installing weasyprint...")
    print("="*60)
    os.system("pip install weasyprint")
    
    # Try weasyprint again after installation
    try:
        from weasyprint import HTML
        print("\nConverting with newly installed WeasyPrint...")
        HTML(filename=html_file).write_pdf(pdf_file)
        print(f"✓ PDF created successfully: {pdf_file}")
        return True
    except Exception as e:
        print(f"Error even after installation: {e}")
        return False

if __name__ == "__main__":
    success = convert_html_to_pdf()
    if not success:
        print("\n⚠ Could not create PDF automatically.")
        print("Please open project_documentation.html in a browser and print to PDF manually.")
        sys.exit(1)
    else:
        print("\n✓ All done!")
