use anyhow::{Context, Result};
use pdf_extract::extract_text;
use std::path::Path;
use tracing::{info, warn};

/// Structured content from PDF
#[derive(Debug, Clone)]
pub struct PdfContent {
    pub text: String,
    pub pages: Vec<String>,
    pub has_text: bool,
}

/// Extract text from a PDF file
pub fn extract_text_from_pdf(path: &Path) -> Result<PdfContent> {
    info!("Extracting text from PDF: {:?}", path);
    
    let text = extract_text(path)
        .with_context(|| format!("Failed to extract text from PDF: {:?}", path))?;
    
    let has_text = !text.trim().is_empty();
    
    if !has_text {
        warn!("PDF appears to be scanned or has no extractable text: {:?}", path);
    }
    
    // Split by page breaks (heuristic - look for form feed characters or multiple newlines)
    let pages: Vec<String> = text
        .split("\x0C")  // Form feed character
        .map(|s| s.to_string())
        .collect();
    
    info!("Extracted {} pages from PDF", pages.len());
    
    Ok(PdfContent {
        text,
        pages,
        has_text,
    })
}

/// Extract structured content with chapter/section detection
pub fn extract_structured_content(path: &Path) -> Result<Vec<(String, String)>> {
    let content = extract_text_from_pdf(path)?;
    
    if !content.has_text {
        anyhow::bail!("PDF has no extractable text. OCR may be required.");
    }
    
    // Simple heuristic: detect chapters by looking for lines that:
    // 1. Start with "Chapter" or numbers
    // 2. Are short (likely titles)
    // 3. Are followed by content
    
    let mut sections = Vec::new();
    let mut current_title = String::from("Introduction");
    let mut current_content = String::new();
    
    for line in content.text.lines() {
        let trimmed = line.trim();
        
        // Check if this looks like a chapter/section heading
        if is_likely_heading(trimmed) {
            // Save previous section
            if !current_content.is_empty() {
                sections.push((current_title.clone(), current_content.trim().to_string()));
                current_content.clear();
            }
            current_title = trimmed.to_string();
        } else {
            // Add to current section content
            if !trimmed.is_empty() {
                current_content.push_str(line);
                current_content.push('\n');
            }
        }
    }
    
    // Add final section
    if !current_content.is_empty() {
        sections.push((current_title, current_content.trim().to_string()));
    }
    
    info!("Detected {} sections in PDF", sections.len());
    
    Ok(sections)
}

/// Heuristic to detect if a line is likely a heading
fn is_likely_heading(line: &str) -> bool {
    if line.is_empty() {
        return false;
    }
    
    // Check for common heading patterns
    let patterns = [
        line.starts_with("Chapter "),
        line.starts_with("CHAPTER "),
        line.starts_with("Section "),
        line.starts_with("Part "),
        line.chars().next().map(|c| c.is_numeric()).unwrap_or(false) && line.len() < 50,
        line.chars().all(|c| c.is_uppercase() || c.is_whitespace()) && line.len() < 80,
    ];
    
    patterns.iter().any(|&p| p)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_is_likely_heading() {
        assert!(is_likely_heading("Chapter 1: Introduction"));
        assert!(is_likely_heading("INTRODUCTION"));
        assert!(is_likely_heading("1. Getting Started"));
        assert!(!is_likely_heading("This is a normal paragraph with some text."));
    }
}

