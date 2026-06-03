from pypdf import PdfReader
import json

reader = PdfReader("data/linkedin.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()
with open("data/linkedin.json", "w") as f:
    json.dump({"linkedin_text": text}, f)