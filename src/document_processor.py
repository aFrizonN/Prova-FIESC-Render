"""
Technical document ingestion, OCR, chunking, and ChromaDB vector indexing.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
import pymupdf  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR
import chromadb
import logging

import config
from src.constants import DOCUMENT_MAPPING
from src.database import get_db_connection

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles PDF extraction (including OCR), semantic chunking, and vector storage."""

    def __init__(self):
        self.ocr_engine: Optional[RapidOCR] = None
        self.chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def _get_ocr_engine(self) -> RapidOCR:
        if self.ocr_engine is None:
            self.ocr_engine = RapidOCR()
        return self.ocr_engine

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extracts text from a PDF file. Automatically uses OCR if text layer is empty."""
        doc = pymupdf.open(pdf_path)
        full_text = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_text = page.get_text().strip()

            if len(page_text) > 30:
                # Digital text layer exists
                full_text.append(f"\n--- Página {page_idx + 1} ---\n" + page_text)
            else:
                # Scanned image page -> run OCR
                logger.info(f"Running OCR on {pdf_path.name} - Page {page_idx + 1}...")
                pix = page.get_pixmap(dpi=150)
                temp_img_path = config.BASE_DIR / f"temp_page_{page_idx}.png"
                pix.save(temp_img_path)

                ocr = self._get_ocr_engine()
                result, _ = ocr(str(temp_img_path))
                if result:
                    ocr_text = "\n".join([line[1] for line in result])
                    full_text.append(f"\n--- Página {page_idx + 1} (OCR) ---\n" + ocr_text)

                if temp_img_path.exists():
                    temp_img_path.unlink()

        doc.close()
        return "\n".join(full_text)

    def chunk_text(self, text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> List[str]:
        """Splits document text into overlapping chunks."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # Keep last paragraph for overlap
                current_chunk = [current_chunk[-1], para] if len(current_chunk) > 1 else [para]
                current_len = sum(len(p) for p in current_chunk)
            else:
                current_chunk.append(para)
                current_len += para_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        # Fallback if text has no double newlines
        if not chunks and text:
            for i in range(0, len(text), chunk_size - overlap):
                chunks.append(text[i:i + chunk_size])

        return chunks

    def process_all_initial_documents(self) -> Dict[str, Any]:
        """Processes all Doc1.pdf - Doc6.pdf from Prova folder and indexes into ChromaDB and SQLite."""
        prova_dir = config.PROVA_DIR
        processed_dir = config.DOCS_DIR / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        all_ids = []
        all_documents = []
        all_metadatas = []

        # Find all Doc*.pdf
        pdf_files = sorted(list(prova_dir.glob("Doc*.pdf")))
        logger.info(f"Found {len(pdf_files)} initial technical documentation PDFs.")

        conn = get_db_connection()
        cursor = conn.cursor()

        for pdf_path in pdf_files:
            doc_name = pdf_path.name
            logger.info(f"Processing technical manual: {doc_name}...")
            
            # Extract text
            extracted_text = self.extract_text_from_pdf(pdf_path)
            
            # Save processed text file
            txt_output = processed_dir / f"{pdf_path.stem}.txt"
            txt_output.write_text(extracted_text, encoding="utf-8")

            # Determine corresponding categories
            matched_categories = [
                cat for cat, info in DOCUMENT_MAPPING.items()
                if info.get("doc_id") == doc_name
            ]

            chunks = self.chunk_text(extracted_text)
            logger.info(f"{doc_name}: Generated {len(chunks)} text chunks.")

            for cat in matched_categories:
                doc_info = DOCUMENT_MAPPING.get(cat, {})
                title = doc_info.get("title", doc_name)

                # Save or update SQLite
                cursor.execute("""
                INSERT INTO fault_documents (fault_category, document_name, document_path, title, text_content, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(fault_category) DO UPDATE SET
                    document_name=excluded.document_name,
                    document_path=excluded.document_path,
                    title=excluded.title,
                    text_content=excluded.text_content,
                    chunk_count=excluded.chunk_count,
                    uploaded_at=CURRENT_TIMESTAMP
                """, (cat, doc_name, str(pdf_path), title, extracted_text, len(chunks)))

                # Add chunks to ChromaDB batch
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"{doc_name}_{cat}_chunk_{chunk_idx}"
                    all_ids.append(chunk_id)
                    all_documents.append(chunk)
                    all_metadatas.append({
                        "source_doc": doc_name,
                        "fault_category": cat,
                        "title": title,
                        "chunk_index": chunk_idx
                    })

            results[doc_name] = {
                "categories": matched_categories,
                "chunks": len(chunks),
                "text_length": len(extracted_text)
            }

        conn.commit()
        conn.close()

        # Upsert into ChromaDB
        if all_documents:
            logger.info(f"Indexing {len(all_documents)} chunks into ChromaDB...")
            self.collection.upsert(
                ids=all_ids,
                documents=all_documents,
                metadatas=all_metadatas
            )
            logger.info("ChromaDB vector indexing completed.")

        return results

    def add_new_document(self, uploaded_file_bytes: bytes, filename: str, fault_category: str, custom_title: str) -> Dict[str, Any]:
        """Allows dynamic upload of new technical documentation for unmapped or new fault types."""
        save_path = config.DOCS_DIR / filename
        with open(save_path, "wb") as f:
            f.write(uploaded_file_bytes)

        extracted_text = self.extract_text_from_pdf(save_path)
        chunks = self.chunk_text(extracted_text)

        # Update SQLite
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO fault_documents (fault_category, document_name, document_path, title, text_content, chunk_count)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(fault_category) DO UPDATE SET
            document_name=excluded.document_name,
            document_path=excluded.document_path,
            title=excluded.title,
            text_content=excluded.text_content,
            chunk_count=excluded.chunk_count,
            uploaded_at=CURRENT_TIMESTAMP
        """, (fault_category, filename, str(save_path), custom_title, extracted_text, len(chunks)))
        conn.commit()
        conn.close()

        # Index in ChromaDB
        ids = [f"{filename}_{fault_category}_chunk_{i}" for i in range(len(chunks))]
        metas = [{
            "source_doc": filename,
            "fault_category": fault_category,
            "title": custom_title,
            "chunk_index": i
        } for i in range(len(chunks))]

        self.collection.upsert(ids=ids, documents=chunks, metadatas=metas)
        logger.info(f"Dynamically added new document '{filename}' for category '{fault_category}'.")

        return {
            "status": "success",
            "filename": filename,
            "fault_category": fault_category,
            "chunks_indexed": len(chunks)
        }

    def query_chunks(self, query_text: str, fault_category: Optional[str] = None, n_results: int = 4) -> List[Dict[str, Any]]:
        """Retrieves top relevant documentation chunks from ChromaDB."""
        where_filter = {"fault_category": fault_category} if fault_category else None
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )

        matched_chunks = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            dists = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                matched_chunks.append({
                    "text": doc,
                    "metadata": meta,
                    "score": round(float(1.0 - dist), 3) if dist is not None else 1.0
                })

        return matched_chunks


# Global document processor singleton
document_processor_instance = DocumentProcessor()


def get_document_processor() -> DocumentProcessor:
    """Returns singleton document processor."""
    return document_processor_instance


if __name__ == "__main__":
    dp = DocumentProcessor()
    res = dp.process_all_initial_documents()
    print("Documents processed:", res)
