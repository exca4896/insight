"""
pipeline.py
-----------
Unified Franklin Templeton ingestion pipeline.

Flow:
  1. Scrape new articles from franklintempleton.com/insights
     (skip any URL already recorded in the JSON registry).
  2. Save each new article as a PDF in `complete_pdfs/`.
  3. Update the JSON registry with the newly scraped metadata.
  4. Convert ONLY the newly created PDFs to Markdown in `md_files/`.
  5. Chunk the new Markdown files and upsert them into ChromaDB.
"""

# ── Standard library ────────────────────────────────────────────────────────
import json
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

# ── Third-party ──────────────────────────────────────────────────────────────
import requests
from bs4 import BeautifulSoup
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from fpdf import FPDF
from langchain_community.docstore.document import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# ── Configuration ────────────────────────────────────────────────────────────
load_dotenv()

JSON_FILE   = "franklin_templeton_insights.json"
PDF_DIR     = "complete_pdfs"
MD_DIR      = "md_files"
CHROMA_PATH = "chroma3"


# ════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Scrape & PDF  (from page_extraction.py)
# ════════════════════════════════════════════════════════════════════════════

def load_json_data(json_file: str):
    """Return parsed JSON list, or None if the file does not exist / is invalid."""
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def search_url(url: str, json_data: list) -> bool:
    """Return True if *url* is already recorded in *json_data*."""
    return any(item["url"] == url for item in json_data)


def scrape_franklin_templeton_insights(num_pages: int, json_data) -> list:
    """
    Scrape article metadata from franklintempleton.com/insights.
    Articles whose URL is already in *json_data* are skipped.

    Returns a list of new article dicts.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36"
    )

    driver = webdriver.Chrome(options=chrome_options)
    articles_data = []

    try:
        driver.get("https://www.franklintempleton.com/insights")
        wait = WebDriverWait(driver, 10)

        for page_num in range(1, num_pages + 1):
            print(f"Scraping page {page_num}...")
            time.sleep(3)

            try:
                wait.until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "article, .article, .insight-card, [class*='card']")
                    )
                )
            except TimeoutException:
                print(f"Timeout waiting for articles on page {page_num}")

            articles = driver.find_elements(By.CSS_SELECTOR, "a.card-article__inner")

            if not articles:
                print(f"No articles found on page {page_num}")
                break

            print(f"Found {len(articles)} articles on page {page_num}")

            for article in articles:
                try:
                    # Title
                    title = None
                    try:
                        title = article.get_attribute("title")
                        if not title:
                            title = article.find_element(By.CSS_SELECTOR, "h4").text.strip()
                    except NoSuchElementException:
                        pass

                    # URL
                    article_url = article.get_attribute("href") or None
                    if article_url and article_url.startswith("/"):
                        article_url = "https://www.franklintempleton.com" + article_url

                    # Skip duplicates
                    if json_data is not None and search_url(article_url, json_data):
                        continue

                    # Tags / date
                    tags = []
                    try:
                        date_text = article.find_element(
                            By.CSS_SELECTOR, ".card-article__date"
                        ).text.strip()
                        if date_text:
                            tags.append(date_text)
                    except NoSuchElementException:
                        pass

                    try:
                        logo_alt = article.find_element(
                            By.CSS_SELECTOR, ".card-article__logo-img img"
                        ).get_attribute("alt")
                        if logo_alt and logo_alt not in tags:
                            tags.append(logo_alt)
                    except NoSuchElementException:
                        pass

                    if title or article_url:
                        articles_data.append(
                            {
                                "title": title or "N/A",
                                "url": article_url or "N/A",
                                "tags": tags,
                                "scraped_timestamp": datetime.now().isoformat(),
                            }
                        )
                        print(f"  - Scraped: {title}")

                except Exception as exc:
                    print(f"Error extracting article data: {exc}")

            # Paginate
            if page_num < num_pages:
                try:
                    time.sleep(2)
                    pagination_pages = driver.find_elements(
                        By.CSS_SELECTOR, "a.pagination__page"
                    )
                    if not pagination_pages:
                        print(f"No pagination elements found on page {page_num}")
                        break

                    clicked = False
                    for page_elem in pagination_pages:
                        try:
                            page_text = (
                                page_elem.text.strip().replace("Go to page", "").strip()
                            )
                            if page_text == str(page_num + 1):
                                driver.execute_script(
                                    "arguments[0].scrollIntoView({block: 'center'});",
                                    page_elem,
                                )
                                time.sleep(1)
                                driver.execute_script("arguments[0].click();", page_elem)
                                print(f"Clicked pagination page {page_num + 1}")
                                time.sleep(4)
                                clicked = True
                                break
                        except Exception as exc:
                            print(f"Error clicking pagination element: {exc}")

                    if not clicked:
                        print(f"Could not find pagination button for page {page_num + 1}")
                        break

                except Exception as exc:
                    print(f"Error navigating to next page: {exc}")
                    break

    finally:
        driver.quit()

    return articles_data


class FullWebpagePDFExtractor:
    """
    Extracts full article content from a URL and saves it as a PDF.
    Returns the list of PDF file paths that were successfully created.
    """

    def __init__(self, output_dir: str = PDF_DIR, use_selenium: bool = True):
        self.output_dir = output_dir
        self.use_selenium = use_selenium
        os.makedirs(output_dir, exist_ok=True)

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
        }
        self.driver = None
        if use_selenium:
            self._init_selenium()

    # ── Selenium setup ───────────────────────────────────────────────────────
    def _init_selenium(self):
        try:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument(f'user-agent={self.headers["User-Agent"]}')
            self.driver = webdriver.Chrome(options=options)
            print("✓ Selenium initialized successfully\n")
        except ImportError:
            print("⚠️  Selenium not installed – falling back to requests.\n")
            self.use_selenium = False
        except Exception as exc:
            print(f"⚠️  Selenium initialization failed: {exc} – falling back to requests.\n")
            self.use_selenium = False

    # ── Public API ───────────────────────────────────────────────────────────
    def process_urls(self, articles: list, delay: int = 3) -> list:
        """
        Download and save PDFs for *articles*.

        Args:
            articles: list of dicts with at least ``url`` and ``tags`` keys.
            delay:    seconds to wait between requests.

        Returns:
            List of file paths for successfully created PDFs.
        """
        pdf_paths = []
        total = len(articles)

        print(f"\n{'='*80}")
        print(f"BULK EXTRACTION: {total} URLs")
        print(f"Method: {'Selenium' if self.use_selenium else 'Requests'}")
        print(f"{'='*80}")

        for i, article_meta in enumerate(articles, 1):
            print(f"\n[{i}/{total}]")

            published_date = (
                article_meta["tags"][0]
                if article_meta.get("tags") and article_meta["tags"]
                else None
            )

            article = self.extract_article(article_meta["url"], published_date=published_date)

            if article and len(article["content"]) > 100:
                try:
                    pdf_path = self.save_as_pdf(article)
                    if pdf_path:
                        pdf_paths.append(pdf_path)
                except Exception as exc:
                    print(f"❌ PDF creation failed: {exc}")
            else:
                print("❌ Extraction failed or insufficient content")

            if i < total:
                print(f"⏳ Waiting {delay}s...")
                time.sleep(delay)

        print(f"\n{'='*80}")
        print(f"Created {len(pdf_paths)} / {total} PDFs in: {self.output_dir}/")
        print(f"{'='*80}")
        return pdf_paths

    def extract_article(self, url: str, published_date=None):
        try:
            if self.use_selenium and self.driver:
                return self._extract_with_selenium(url, published_date)
            return self._extract_with_requests(url, published_date)
        except Exception as exc:
            print(f"❌ Error: {exc}")
            return None

    def close(self):
        if self.driver:
            self.driver.quit()
            print("\n✓ Selenium driver closed")

    # ── Private extraction helpers ───────────────────────────────────────────
    def _extract_with_selenium(self, url: str, published_date=None):
        print("🌐 Loading page with Selenium...")
        try:
            self.driver.get(url)
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "article"))
                )
            except Exception:
                pass

            time.sleep(3)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            self.driver.execute_script("window.scrollTo(0, 0);")

            html = self.driver.page_source.encode("utf-8", errors="replace").decode("utf-8")
            soup = BeautifulSoup(html, "html.parser")

            title = self._sanitize_text(self._get_title(soup))
            if not published_date:
                published_date = self._scrape_date_from_page(soup)
            date = self._sanitize_text(published_date)

            content = self._sanitize_text(self._get_complete_content(soup))
            print(f"✓ Extracted {len(content)} characters")

            return {
                "title": title,
                "url": url,
                "content": content,
                "date": date,
                "extraction_method": "selenium",
            }
        except Exception as exc:
            print(f"❌ Selenium extraction failed: {exc}")
            return None

    def _extract_with_requests(self, url: str, published_date=None):
        print("🌐 Loading page with requests...")
        response = requests.get(url, headers=self.headers, timeout=30)
        html = response.content.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        title = self._sanitize_text(self._get_title(soup))
        if not published_date:
            published_date = self._scrape_date_from_page(soup)
        date = self._sanitize_text(published_date)

        content = self._sanitize_text(self._get_complete_content(soup))
        print(f"✓ Extracted {len(content)} characters")

        return {
            "title": title,
            "url": url,
            "content": content,
            "date": date,
            "extraction_method": "requests",
        }

    # ── Content helpers ──────────────────────────────────────────────────────
    def _get_title(self, soup):
        for selector in [
            lambda s: s.find("h1"),
            lambda s: s.find("meta", {"property": "og:title"}),
            lambda s: s.find("title"),
        ]:
            elem = selector(soup)
            if elem:
                return (
                    elem.get("content", "") if elem.name == "meta" else elem.get_text(strip=True)
                )
        return "Untitled Article"

    def _scrape_date_from_page(self, soup):
        time_elem = soup.find("time")
        if time_elem:
            return time_elem.get("datetime", "") or time_elem.get_text(strip=True)

        date_meta = soup.find("meta", {"property": "article:published_time"})
        if date_meta:
            return date_meta.get("content", "")

        for selector in [
            ".card-article__date", ".article__date",
            ".publish-date", '[class*="date"]',
        ]:
            elem = soup.select_one(selector)
            if elem:
                return elem.get_text(strip=True)

        return "Unknown"

    def _get_complete_content(self, soup):
        print("\n📝 Extracting content...")
        for tag in soup(["script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "button"]):
            tag.decompose()

        container = soup.find("article") or soup.find("main")

        if not container:
            all_divs = soup.find_all("div")
            best, max_score = None, 0
            for div in all_divs:
                score = len(div.find_all("p")) * 100 + len(div.get_text(strip=True)) / 10
                if score > max_score:
                    max_score, best = score, div
            container = best or soup.find("body")

        content = self._extract_all_text_from_element(container) if container else ""
        content = self._clean_text(content)
        content = self._remove_duplicate_paragraphs(content)
        print(f"✓ Final content: {len(content)} characters")
        return content

    def _extract_all_text_from_element(self, element):
        if not element:
            return ""
        text_parts, processed = [], set()
        for child in element.descendants:
            if not hasattr(child, "name"):
                continue
            if child.name not in ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]:
                continue
            parent = child.find_parent(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"])
            if parent and parent in processed:
                continue
            text = child.get_text(strip=True)
            if not text or len(text) < 5 or self._is_noise(text):
                continue
            processed.add(child)
            if child.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                text_parts.append(f"\n{text}\n")
            else:
                text_parts.append(text)
        return "\n\n".join(text_parts)

    def _remove_duplicate_paragraphs(self, text):
        paragraphs, deduplicated, seen = text.split("\n\n"), [], set()
        for para in paragraphs:
            h = hash(para.strip())
            if h not in seen:
                deduplicated.append(para)
                seen.add(h)
            elif len(deduplicated) > 10:
                seen.discard(h)
                deduplicated.append(para)
                seen.add(h)
        return "\n\n".join(deduplicated)

    def _is_noise(self, text):
        tl = text.lower().strip()
        if len(text) < 5:
            return True
        noise = [
            "cookie", "menu", "navigation", "skip to", "sign in",
            "log in", "subscribe", "follow us", "share this",
            "print", "download", "privacy policy", "terms of use",
            "back to top", "read more", "learn more", "close",
        ]
        return any(tl == p or tl.startswith(p) for p in noise)

    def _sanitize_text(self, text):
        if not text:
            return ""
        replacements = {
            "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " ",
            "\u200b": "", "\u2122": "", "\u00ae": "", "\u00a9": "",
            "\u2039": "<", "\u203a": ">", "\u201e": '"', "\u201a": "'",
            "\u009d": "",
        }
        for char, rep in replacements.items():
            text = text.replace(char, rep)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    def _clean_text(self, text):
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    # ── PDF creation ─────────────────────────────────────────────────────────
    def save_as_pdf(self, article: dict) -> str | None:
        if not article:
            return None

        safe_title = re.sub(r"[^\w\s-]", "", article["title"])[:50]
        safe_title = re.sub(r"[-\s]+", "_", safe_title)
        filepath = os.path.join(self.output_dir, f"{safe_title}.pdf")

        print(f"\n📄 Creating PDF: {os.path.basename(filepath)}")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        pdf.set_font("Arial", "B", 16)
        pdf.multi_cell(0, 10, self._sanitize_text(article["title"]))
        pdf.ln(5)

        pdf.set_font("Arial", "I", 9)
        if article.get("date"):
            pdf.cell(0, 5, f"Published: {self._sanitize_text(article['date'])}")
            pdf.ln()
        pdf.cell(0, 5, f"Source: {article['url']}")
        pdf.ln()
        pdf.cell(0, 5, f"Extraction method: {article.get('extraction_method', 'unknown')}")
        pdf.ln(10)

        pdf.set_font("Arial", "", 10)
        for line in article["content"].split("\n"):
            line = self._sanitize_text(line.strip())
            if not line:
                pdf.ln(3)
                continue
            if len(line) < 100 and not line.endswith((".", "!", "?", ":", ";", ",")):
                pdf.set_font("Arial", "B", 12)
                pdf.multi_cell(0, 6, line)
                pdf.set_font("Arial", "", 10)
            else:
                pdf.multi_cell(0, 5, line)
            pdf.ln(2)

        pdf.output(filepath)
        print(f"✓ PDF saved: {filepath} ({os.path.getsize(filepath)/1024:.1f} KB)")
        return filepath


# ════════════════════════════════════════════════════════════════════════════
# STAGE 2 — PDF → Markdown  (from doc_converter.py)
# ════════════════════════════════════════════════════════════════════════════

def convert_pdfs_to_markdown(pdf_paths: list, output_dir: str = MD_DIR) -> list:
    """
    Convert a list of PDF file paths to Markdown files.

    Args:
        pdf_paths:  List of absolute/relative paths to PDF files.
        output_dir: Directory where .md files will be written.

    Returns:
        List of Path objects for successfully created Markdown files.
    """
    if not pdf_paths:
        print("No PDFs to convert.")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    converter = DocumentConverter()
    md_paths = []

    print(f"\n{'='*80}")
    print(f"CONVERTING {len(pdf_paths)} PDFs → Markdown")
    print(f"{'='*80}")

    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        try:
            result = converter.convert(pdf_path)
            md_file = output_dir / f"{pdf_path.stem}.md"
            md_file.write_text(result.document.export_to_markdown(), encoding="utf-8")
            md_paths.append(md_file)
            print(f"✓ Converted: {pdf_path.name} → {md_file.name}")
        except Exception as exc:
            print(f"❌ Failed to convert {pdf_path.name}: {exc}")

    print(f"\n✓ {len(md_paths)} Markdown files written to: {output_dir}/")
    return md_paths


# ════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Markdown → ChromaDB  (from store_chroma_updated.py)
# ════════════════════════════════════════════════════════════════════════════

def extract_metadata_and_content(file_path: Path) -> tuple[dict, str]:
    """
    Parse metadata header lines and main content from a Markdown file.
    Returns (metadata dict, cleaned content string).
    """
    metadata = {
        "source": str(file_path),
        "filename": file_path.name,
        "published_date": None,
        "source_url": None,
        "extraction_method": None,
        "title": None,
    }

    try:
        full_content = file_path.read_text(encoding="utf-8")
        lines = full_content.split("\n")
        content_start_idx = 0

        for idx, line in enumerate(lines[:20]):
            if re.match(r"Published:", line, re.IGNORECASE):
                metadata["published_date"] = re.sub(
                    r"Published:\s*", "", line, flags=re.IGNORECASE
                ).strip()
                content_start_idx = max(content_start_idx, idx + 1)

            elif re.match(r"Source:", line, re.IGNORECASE):
                m = re.search(r"https?://[^\s]+", line)
                if m:
                    metadata["source_url"] = m.group(0).strip()
                content_start_idx = max(content_start_idx, idx + 1)

            elif re.match(r"Extraction method:", line, re.IGNORECASE):
                metadata["extraction_method"] = re.sub(
                    r"Extraction method:\s*", "", line, flags=re.IGNORECASE
                ).strip()
                content_start_idx = max(content_start_idx, idx + 1)

            elif line.startswith("##") and not metadata["title"]:
                metadata["title"] = line.strip("#").strip()

        # Skip leftover metadata/blank lines at the top
        skip_prefixes = ("Published:", "Source:", "Extraction method:")
        while content_start_idx < len(lines) and (
            not lines[content_start_idx].strip()
            or lines[content_start_idx].startswith(skip_prefixes)
        ):
            content_start_idx += 1

        content = "\n".join(lines[content_start_idx:])

        # Strip boilerplate legal sections
        for marker in [
            "WHAT ARE THE RISKS?",
            "IMPORTANT LEGAL INFORMATION",
            "This material is intended to be of general interest only",
            "CONTRIBUTORS",
        ]:
            pos = content.find(marker)
            if pos != -1:
                content = content[:pos]
                break

        return metadata, content.strip()

    except Exception as exc:
        print(f"Error processing {file_path}: {exc}")
        return metadata, ""


def load_md_files(md_paths: list) -> list[Document]:
    """
    Load a specific list of Markdown file paths into LangChain Documents.
    Only files with non-empty content are returned.
    """
    docs = []
    for path in md_paths:
        path = Path(path)
        metadata, content = extract_metadata_and_content(path)
        if content:
            docs.append(Document(page_content=content, metadata=metadata))
            print(f"Loaded: {path.name} ({len(content)} chars)")
        else:
            print(f"Warning: No content extracted from {path.name}")
    return docs


def split_documents(docs: list) -> list:
    """Split documents into overlapping chunks for vector storage."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"\nSplit {len(docs)} documents into {len(chunks)} chunks.")
    return chunks


def upsert_to_vectordb(chunks: list, chroma_path: str = CHROMA_PATH):
    """
    Add *chunks* to an existing ChromaDB collection, or create one if absent.
    Uses upsert semantics so re-running the pipeline never duplicates content.
    """
    if not chunks:
        print("No chunks to store.")
        return

    embeddings = OpenAIEmbeddings()

    if os.path.exists(chroma_path):
        print(f"\nUpdating existing ChromaDB at '{chroma_path}'...")
        db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
        db.add_documents(chunks)
    else:
        print(f"\nCreating new ChromaDB at '{chroma_path}'...")
        db = Chroma.from_documents(chunks, embeddings, persist_directory=chroma_path)

    db.persist()
    print(f"✓ Upserted {len(chunks)} chunks into ChromaDB.")


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("FRANKLIN TEMPLETON INGESTION PIPELINE")
    print("=" * 80)

    # ── Stage 1a: Load existing registry ────────────────────────────────────
    print("\n[Stage 1] Scraping new articles...")
    existing_data = load_json_data(JSON_FILE)

    # ── Stage 1b: Scrape new article metadata ────────────────────────────────
    new_articles = scrape_franklin_templeton_insights(
        num_pages=10, json_data=existing_data
    )

    if not new_articles:
        print("No new articles found. Pipeline complete.")
        return

    print(f"\n→ {len(new_articles)} new article(s) found.")

    # ── Stage 1c: Download content and save PDFs ─────────────────────────────
    extractor = FullWebpagePDFExtractor(output_dir=PDF_DIR, use_selenium=True)
    try:
        pdf_paths = extractor.process_urls(new_articles, delay=3)
    finally:
        extractor.close()

    # ── Stage 1d: Persist updated JSON registry ───────────────────────────────
    combined = (existing_data or []) + new_articles
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\n✓ JSON registry updated: {JSON_FILE} ({len(combined)} total articles)")

    if not pdf_paths:
        print("No PDFs were created. Pipeline complete.")
        return

    # ── Stage 2: Convert new PDFs to Markdown ────────────────────────────────
    print("\n[Stage 2] Converting PDFs to Markdown...")
    md_paths = convert_pdfs_to_markdown(pdf_paths, output_dir=MD_DIR)

    if not md_paths:
        print("No Markdown files created. Pipeline complete.")
        return

    # ── Stage 3: Chunk and store in ChromaDB ─────────────────────────────────
    print("\n[Stage 3] Loading and chunking Markdown files...")
    docs = load_md_files(md_paths)

    if not docs:
        print("No documents loaded from Markdown files. Pipeline complete.")
        return

    chunks = split_documents(docs)
    upsert_to_vectordb(chunks, chroma_path=CHROMA_PATH)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print(f"  New articles scraped : {len(new_articles)}")
    print(f"  PDFs created         : {len(pdf_paths)}")
    print(f"  Markdown files       : {len(md_paths)}")
    print(f"  Chunks stored        : {len(chunks)}")
    print("=" * 80)


if __name__ == "__main__":
    main()