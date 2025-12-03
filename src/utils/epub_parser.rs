use anyhow::{Context, Result};
use epub::doc::EpubDoc;
use std::path::Path;
use tracing::info;

/// Structured content from EPUB
#[derive(Debug, Clone)]
pub struct EpubContent {
    pub title: String,
    pub author: String,
    pub chapters: Vec<(String, String)>,  // (title, content)
}

/// Extract text from an EPUB file
pub fn extract_text_from_epub(path: &Path) -> Result<EpubContent> {
    info!("Extracting text from EPUB: {:?}", path);
    
    let mut doc = EpubDoc::new(path)
        .with_context(|| format!("Failed to open EPUB file: {:?}", path))?;
    
    // Get metadata
    let title = doc.mdata("title").unwrap_or_else(|| "Unknown".to_string());
    let author = doc.mdata("creator").unwrap_or_else(|| "Unknown".to_string());
    
    info!("EPUB: {} by {}", title, author);
    
    // Extract chapters
    let mut chapters = Vec::new();
    
    // Get the spine (reading order)
    let spine_len = doc.spine.len();
    
    for i in 0..spine_len {
        doc.set_current_page(i);
        
        if let Some((content_bytes, _mime)) = doc.get_current_str() {
            // Parse HTML content
            let content = strip_html_tags(&content_bytes);
            
            if !content.trim().is_empty() {
                // Try to extract chapter title from the content
                let chapter_title = extract_chapter_title(&content)
                    .unwrap_or_else(|| format!("Chapter {}", i + 1));
                
                chapters.push((chapter_title, content));
            }
        }
    }
    
    info!("Extracted {} chapters from EPUB", chapters.len());
    
    Ok(EpubContent {
        title,
        author,
        chapters,
    })
}

/// Strip HTML tags from text (simple implementation)
fn strip_html_tags(html: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;
    let mut in_script_or_style = false;
    
    let mut chars = html.chars().peekable();
    
    while let Some(ch) = chars.next() {
        if ch == '<' {
            in_tag = true;
            
            // Check if this is a script or style tag
            let remaining: String = chars.clone().take(10).collect();
            if remaining.to_lowercase().starts_with("script") || 
               remaining.to_lowercase().starts_with("style") {
                in_script_or_style = true;
            }
        } else if ch == '>' {
            in_tag = false;
            
            // Check if this closes script or style
            if in_script_or_style {
                in_script_or_style = false;
            }
        } else if !in_tag && !in_script_or_style {
            result.push(ch);
        }
    }
    
    // Clean up extra whitespace
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Extract chapter title from content (first heading or first line)
fn extract_chapter_title(content: &str) -> Option<String> {
    // Look for the first non-empty line as potential title
    for line in content.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() && trimmed.len() < 100 {
            return Some(trimmed.to_string());
        }
    }
    
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_strip_html_tags() {
        let html = "<p>Hello <b>World</b>!</p>";
        let text = strip_html_tags(html);
        assert_eq!(text, "Hello World !");
    }
    
    #[test]
    fn test_strip_script_tags() {
        let html = "<p>Text</p><script>alert('hi');</script><p>More text</p>";
        let text = strip_html_tags(html);
        assert!(text.contains("Text"));
        assert!(text.contains("More text"));
        assert!(!text.contains("alert"));
    }
}

