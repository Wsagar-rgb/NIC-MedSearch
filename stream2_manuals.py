# ─────────────────────────────────────────────────────────────
# stream2_manuals_enhanced.py — PDF Extraction + Tables + Images
# Enhanced with:
#   - Table extraction → HTML → LLM summarization (Groq)
#   - Image extraction → Vision description (Groq Vision API)
#   - Separate doc_types for tables and images
# Usage: python stream2_manuals_enhanced.py
# ─────────────────────────────────────────────────────────────

import os
import json
import logging
import hashlib
import base64
import io
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF
from PIL import Image
import groq

from config import MANUALS_DIR, MANUAL_JSONL, PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Groq API Setup ────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set. See README for setup.")

groq_client = groq.Groq(api_key=GROQ_API_KEY)

# ── Image storage directory ───────────────────────────────────
IMAGES_DIR = PROCESSED_DIR / "extracted_images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ── Utility functions ─────────────────────────────────────────

def compute_hash(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_tables_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract tables from PDF using PyMuPDF's table detection.
    Returns list of dicts: {page_num, table_index, table_image, table_bbox}
    """
    tables = []
    doc = fitz.open(pdf_path)
    
    for page_num, page in enumerate(doc):
        try:
            # Detect tables on page
            table_list = page.find_tables()
            
            if not table_list:
                continue
            
            for table_idx, table in enumerate(table_list.tables):
                try:
                    # Extract table area as image
                    bbox = table.bbox
                    # Expand bbox slightly for better context
                    expanded_bbox = fitz.Rect(
                        max(0, bbox.x0 - 10),
                        max(0, bbox.y0 - 10),
                        min(page.rect.width, bbox.x1 + 10),
                        min(page.rect.height, bbox.y1 + 10)
                    )
                    
                    pix = page.get_pixmap(clip=expanded_bbox, matrix=fitz.Matrix(2, 2))
                    table_image = pix.tobytes("png")
                    
                    # Extract text from table for metadata
                    table_text = table.extract()
                    
                    tables.append({
                        "page_num": page_num,
                        "table_index": table_idx,
                        "bbox": bbox,
                        "image": table_image,
                        "raw_text": table_text,
                        "pdf_name": Path(pdf_path).stem,
                    })
                    
                    log.info(f"  ✓ Extracted table {table_idx+1} from page {page_num+1}")
                    
                except Exception as e:
                    log.warning(f"Failed to extract table {table_idx} on page {page_num+1}: {e}")
                    continue
        
        except Exception as e:
            log.warning(f"Error processing page {page_num+1}: {e}")
            continue
    
    doc.close()
    return tables


def extract_images_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract images from PDF pages.
    Returns list of dicts: {page_num, image_index, image_data, image_size}
    """
    images = []
    doc = fitz.open(pdf_path)
    
    for page_num, page in enumerate(doc):
        try:
            # Get all images on the page
            image_list = page.get_images(full=True)
            
            for img_idx, img_ref in enumerate(image_list):
                try:
                    xref = img_ref[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Convert to RGB if necessary
                    if pix.n - pix.alpha < 4:  # Gray or RGB
                        image_data = pix.tobytes("png")
                    else:  # CMYK
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        image_data = pix.tobytes("png")
                    
                    images.append({
                        "page_num": page_num,
                        "image_index": img_idx,
                        "image_data": image_data,
                        "image_size": (pix.width, pix.height),
                        "pdf_name": Path(pdf_path).stem,
                    })
                    
                    log.info(f"  ✓ Extracted image {img_idx+1} from page {page_num+1}")
                    
                except Exception as e:
                    log.warning(f"Failed to extract image {img_idx} on page {page_num+1}: {e}")
                    continue
        
        except Exception as e:
            log.warning(f"Error processing page {page_num+1}: {e}")
            continue
    
    doc.close()
    return images


def summarize_table_with_llm(table_image: bytes, table_text: str, pdf_name: str) -> str:
    """
    Use Groq Vision API to describe table content in detail.
    """
    try:
        # Encode image to base64
        image_base64 = base64.standard_b64encode(table_image).decode("utf-8")
        
        prompt = f"""You are a technical documentation expert. I am extracting structured data from medical equipment manuals for a retrieval-augmented generation system.

Analyze this table image carefully and provide a detailed, natural-language description that would help someone find this table when searching.

IMPORTANT: 
- Write as descriptive sentences, not bullet points
- Include row headers, column headers, and all key values
- Explain what each column means and why the data matters
- Format: "This table shows [what it shows]. It contains columns for [headers]. For example, [specific entries]."

Source: {pdf_name}

Provide ONLY the description, no extra text."""

        response = groq_client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        
        description = response.choices[0].message.content
        return description.strip()
    
    except Exception as e:
        log.error(f"Error summarizing table: {e}")
        # Fallback: return raw table text
        return f"Table data: {table_text[:500]}"


def describe_image_with_llm(image_data: bytes, pdf_name: str) -> str:
    """
    Use Groq Vision API to describe image content for RAG.
    """
    try:
        image_base64 = base64.standard_b64encode(image_data).decode("utf-8")
        
        prompt = """You are a technical documentation expert helping index images from medical equipment manuals for a search system.

Analyze this image and provide a clear, detailed description that would help someone find it when searching for related equipment or procedures.

IMPORTANT:
- Describe what the image shows (diagram, flowchart, photo, graph, etc.)
- For diagrams/flowcharts: describe the structure, components, and flow
- For graphs/charts: describe axes, data trends, and key insights
- For photos: describe the equipment, parts, and context
- Include any text, labels, or annotations visible in the image
- Write naturally, not as bullet points

Provide ONLY the description, no extra commentary."""

        response = groq_client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=600,
        )
        
        description = response.choices[0].message.content
        return description.strip()
    
    except Exception as e:
        log.error(f"Error describing image: {e}")
        return "Image description unavailable"


def save_image_file(image_data: bytes, pdf_name: str, img_type: str, index: int) -> str:
    """
    Save extracted image to disk. Returns relative path for storage in payload.
    """
    try:
        filename = f"{pdf_name}_{img_type}_{index:03d}.png"
        filepath = IMAGES_DIR / filename
        
        with open(filepath, "wb") as f:
            f.write(image_data)
        
        # Return path relative to PROCESSED_DIR for portability
        return f"extracted_images/{filename}"
    
    except Exception as e:
        log.error(f"Error saving image file: {e}")
        return None


def process_pdf(pdf_path: str) -> list[dict]:
    """
    Process a single PDF and extract text + tables + images.
    Returns list of records ready for indexing.
    """
    records = []
    pdf_name = Path(pdf_path).stem
    
    log.info(f"\n📄 Processing: {pdf_path}")
    
    # ── Extract text sections (existing logic) ────────────────
    doc = fitz.open(pdf_path)
    
    # Join pages first (fixes cross-page procedure splits)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    
    # Split by headings
    heading_pattern = r"^[A-Z][A-Z\s]*$"
    import re
    
    sections = []
    current_heading = None
    current_content = []
    
    for line in full_text.split("\n"):
        stripped = line.strip()
        
        if re.match(heading_pattern, stripped) and len(stripped) < 100:
            if current_heading and current_content:
                sections.append((current_heading, "\n".join(current_content)))
            current_heading = stripped
            current_content = []
        else:
            if current_heading:
                current_content.append(line)
    
    if current_heading and current_content:
        sections.append((current_heading, "\n".join(current_content)))
    
    # Create records from text sections
    for section_idx, (heading, content) in enumerate(sections):
        content_clean = content.strip()
        if not content_clean or len(content_clean) < 50:
            continue
        
        # Chunk large sections
        chunk_size = 800
        chunks = [content_clean[i:i+chunk_size] for i in range(0, len(content_clean), chunk_size)]
        
        for chunk_idx, chunk in enumerate(chunks):
            rec = {
                "id": f"{pdf_name}_section_{section_idx}_{chunk_idx}",
                "doc_type": "manual_section",
                "source_pdf": pdf_name,
                "source_file": Path(pdf_path).name,
                "title": heading,
                "content": chunk,
                "content_hash": compute_hash(chunk),
                "timestamp": datetime.now().isoformat(),
            }
            records.append(rec)
    
    doc.close()
    
    log.info(f"  ✓ Extracted {len(records)} text sections")
    
    # ── Extract and process tables ────────────────────────────
    log.info("  🔄 Extracting tables …")
    tables = extract_tables_from_pdf(pdf_path)
    
    for table in tables:
        description = summarize_table_with_llm(
            table["image"],
            table["raw_text"],
            pdf_name
        )
        
        # Save table image
        image_path = save_image_file(
            table["image"],
            pdf_name,
            "table",
            table["table_index"]
        )
        
        rec = {
            "id": f"{pdf_name}_table_{table['table_index']}",
            "doc_type": "table_description",
            "source_pdf": pdf_name,
            "source_file": Path(pdf_path).name,
            "page_num": table["page_num"],
            "title": f"Table on page {table['page_num']+1}",
            "content": description,  # This is embedded
            "image_path": image_path,  # Original image for display
            "content_hash": compute_hash(description),
            "timestamp": datetime.now().isoformat(),
        }
        records.append(rec)
        log.info(f"    ✓ Summarized table {table['table_index']+1}")
    
    # ── Extract and process images ────────────────────────────
    log.info("  🔄 Extracting images …")
    images = extract_images_from_pdf(pdf_path)
    
    for image in images:
        description = describe_image_with_llm(image["image_data"], pdf_name)
        
        # Save image
        image_path = save_image_file(
            image["image_data"],
            pdf_name,
            "img",
            image["image_index"]
        )
        
        rec = {
            "id": f"{pdf_name}_image_{image['image_index']}",
            "doc_type": "image_description",
            "source_pdf": pdf_name,
            "source_file": Path(pdf_path).name,
            "page_num": image["page_num"],
            "title": f"Image/Diagram on page {image['page_num']+1}",
            "content": description,  # This is embedded
            "image_path": image_path,  # Original image for display
            "image_size": image["image_size"],
            "content_hash": compute_hash(description),
            "timestamp": datetime.now().isoformat(),
        }
        records.append(rec)
        log.info(f"    ✓ Described image {image['image_index']+1}")
    
    return records


def main():
    log.info("=" * 60)
    log.info("NIC MedSearch — Enhanced PDF Processing (Text + Tables + Images)")
    log.info("=" * 60)
    
    pdf_files = sorted(MANUALS_DIR.glob("*.pdf"))
    
    if not pdf_files:
        log.warning(f"No PDFs found in {MANUALS_DIR}")
        return
    
    all_records = []
    
    for pdf_path in pdf_files:
        try:
            records = process_pdf(str(pdf_path))
            all_records.extend(records)
        except Exception as e:
            log.error(f"Failed to process {pdf_path.name}: {e}")
            continue
    
    # ── Save to JSONL ─────────────────────────────────────────
    log.info(f"\n📝 Saving {len(all_records)} records to {MANUAL_JSONL}")
    
    with open(MANUAL_JSONL, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")
    
    log.info("✓ Done!")
    log.info(f"\n📊 Summary:")
    log.info(f"  Text sections: {sum(1 for r in all_records if r['doc_type'] == 'manual_section')}")
    log.info(f"  Table descriptions: {sum(1 for r in all_records if r['doc_type'] == 'table_description')}")
    log.info(f"  Image descriptions: {sum(1 for r in all_records if r['doc_type'] == 'image_description')}")
    log.info(f"  Total records: {len(all_records)}")
    log.info(f"\n  Images saved to: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
