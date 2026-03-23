import pypandoc
import re
import os

print("Starting DOCX Conversion...")

# Load the markdown file
md_file = 'lung_disease_basic_analysis_report.md'
with open(md_file, 'r', encoding='utf-8') as f:
    text = f.read()

# The Markdown currently uses absolute file:/// paths to the brain folder.
# We need to strip these down to just the local filenames so Pandoc can embed the PNGs from the current directory CWD
text = re.sub(r'\/C:\/Users\/akabi\/\.gemini\/antigravity\/brain\/[a-zA-Z0-9\-]+\/', '', text)

# Save a temporary clean markdown
with open('temp_report.md', 'w', encoding='utf-8') as f:
    f.write(text)

# Convert to DOCX
output_file = 'Lung_Disease_XV_Analysis_Report.docx'
pypandoc.convert_file('temp_report.md', 'docx', outputfile=output_file)

# Cleanup
if os.path.exists('temp_report.md'):
    os.remove('temp_report.md')

print(f"Successfully generated {output_file} with embedded figures!")
