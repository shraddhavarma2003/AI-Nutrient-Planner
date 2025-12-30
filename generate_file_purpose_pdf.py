import os
import sys
print('--- Starting PDF generation script ---')
from fpdf import FPDF
from utils.file_purpose_extractor import extract_purpose

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'AI Nutrition Project - File Purpose Documentation', ln=True, align='C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def is_excluded_dir(dirname):
    # Exclude virtual env and common large dirs
    exclude = {'venv', '__pycache__', 'node_modules', '.git'}
    return os.path.basename(dirname) in exclude

def collect_files(root_dir):
    file_entries = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # modify dirnames in-place to skip excluded dirs
        dirnames[:] = [d for d in dirnames if not is_excluded_dir(d)]
        for fname in filenames:
            if fname.endswith('.py'):
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, root_dir)
                purpose = extract_purpose(full_path)
                file_entries.append((rel_path, purpose))
    return sorted(file_entries)

def generate_pdf(entries, output_path='File_Purpose_Documentation.pdf'):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Helvetica', '', 12)
    # Table of contents
    pdf.cell(0, 10, 'Table of Contents', ln=True)
    for i, (path, _) in enumerate(entries, 1):
        pdf.cell(0, 8, f'{i}. {path}', ln=True)
    pdf.add_page()
    # Detailed sections
    for i, (path, purpose) in enumerate(entries, 1):
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, f'{i}. {path}', ln=True)
        pdf.set_font('Helvetica', '', 12)
        pdf.multi_cell(0, 8, f'Purpose: {purpose}')
        pdf.ln(5)
    pdf.output(output_path)
    print(f'PDF generated at {output_path}')

if __name__ == '__main__':
    try:
        print('Collecting files...')
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        entries = collect_files(project_root)
        print(f'Collected {len(entries)} python files')
        generate_pdf(entries)
        print('PDF generation completed')
    except Exception as e:
        print('Error during PDF generation:', e)
        raise
