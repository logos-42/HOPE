pub mod epub_parser;
pub mod ocr;
pub mod pdf_parser;
pub mod text_processor;

pub use epub_parser::extract_text_from_epub;
pub use ocr::{auto_ocr_if_needed, is_scanned_pdf, ocr_pdf_with_tesseract};
pub use pdf_parser::extract_text_from_pdf;
pub use text_processor::{clean_text, add_structure_markers};

