use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;
use tracing::{info, warn};

/// Check if a PDF is likely scanned (has no extractable text)
pub fn is_scanned_pdf(path: &Path) -> Result<bool> {
    let content = crate::utils::pdf_parser::extract_text_from_pdf(path)?;
    Ok(!content.has_text)
}

/// Perform OCR on a PDF file using Tesseract (external tool)
/// 
/// Note: This requires Tesseract to be installed on the system.
/// Install: 
/// - Windows: https://github.com/UB-Mannheim/tesseract/wiki
/// - Linux: sudo apt-get install tesseract-ocr
/// - Mac: brew install tesseract
pub fn ocr_pdf_with_tesseract(path: &Path) -> Result<String> {
    info!("Performing OCR on PDF: {:?}", path);
    
    // Check if tesseract is available
    let tesseract_check = Command::new("tesseract")
        .arg("--version")
        .output();
    
    if tesseract_check.is_err() {
        anyhow::bail!(
            "Tesseract OCR is not installed or not in PATH. \
             Please install Tesseract: https://github.com/tesseract-ocr/tesseract"
        );
    }
    
    // Create temporary directory for images
    let temp_dir = std::env::temp_dir().join(format!("hope_ocr_{}", 
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    ));
    
    std::fs::create_dir_all(&temp_dir)?;
    
    // Convert PDF to images using pdftoppm (part of poppler-utils)
    info!("Converting PDF to images...");
    let output = Command::new("pdftoppm")
        .arg("-png")
        .arg(path)
        .arg(temp_dir.join("page"))
        .output();
    
    if output.is_err() {
        warn!("pdftoppm not found. Trying alternative method...");
        // Cleanup and return error
        let _ = std::fs::remove_dir_all(&temp_dir);
        anyhow::bail!(
            "PDF to image conversion failed. Install poppler-utils: \
             Linux: sudo apt-get install poppler-utils, \
             Mac: brew install poppler, \
             Windows: download from https://github.com/oschwartz10612/poppler-windows/releases/"
        );
    }
    
    // Run OCR on each image
    let mut all_text = String::new();
    let mut page_count = 0;
    
    for entry in std::fs::read_dir(&temp_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("png") {
            page_count += 1;
            info!("OCR processing page {}...", page_count);
            
            let output_base = temp_dir.join(format!("ocr_page_{}", page_count));
            
            let output = Command::new("tesseract")
                .arg(&path)
                .arg(&output_base)
                .arg("-l")
                .arg("eng")  // Language: English (change as needed)
                .output()?;
            
            if !output.status.success() {
                warn!("Tesseract failed for page {}", page_count);
                continue;
            }
            
            // Read the output text file
            let text_file = output_base.with_extension("txt");
            if text_file.exists() {
                let page_text = std::fs::read_to_string(&text_file)?;
                all_text.push_str(&page_text);
                all_text.push_str("\n\n");
            }
        }
    }
    
    // Cleanup
    let _ = std::fs::remove_dir_all(&temp_dir);
    
    info!("OCR completed: {} pages processed", page_count);
    
    if all_text.is_empty() {
        anyhow::bail!("OCR produced no text");
    }
    
    Ok(all_text)
}

/// Perform OCR using an external API (placeholder for future implementation)
pub fn ocr_pdf_with_api(path: &Path, api_key: &str) -> Result<String> {
    // This is a placeholder for cloud OCR services like:
    // - Google Cloud Vision API
    // - Azure Computer Vision
    // - AWS Textract
    
    warn!("API-based OCR not yet implemented for: {:?}", path);
    warn!("API key provided: {}", if api_key.is_empty() { "none" } else { "yes" });
    
    anyhow::bail!("API-based OCR not yet implemented. Use Tesseract OCR instead.")
}

/// Auto-detect and perform OCR if needed
pub fn auto_ocr_if_needed(path: &Path) -> Result<String> {
    // First try to extract text normally
    match crate::utils::pdf_parser::extract_text_from_pdf(path) {
        Ok(content) if content.has_text => {
            info!("PDF has extractable text, no OCR needed");
            return Ok(content.text);
        }
        _ => {
            info!("PDF appears to be scanned, attempting OCR...");
        }
    }
    
    // Try OCR with Tesseract
    ocr_pdf_with_tesseract(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ocr_availability() {
        let result = Command::new("tesseract").arg("--version").output();
        
        if result.is_ok() {
            println!("Tesseract is available");
        } else {
            println!("Tesseract is not installed");
        }
    }
}

