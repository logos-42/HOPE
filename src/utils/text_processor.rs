use regex::Regex;

/// Clean text by removing extra whitespace and special characters
pub fn clean_text(text: &str) -> String {
    // Remove multiple spaces
    let re_spaces = Regex::new(r"\s+").unwrap();
    let text = re_spaces.replace_all(text, " ");
    
    // Remove multiple newlines (keep paragraph breaks)
    let re_newlines = Regex::new(r"\n{3,}").unwrap();
    let text = re_newlines.replace_all(&text, "\n\n");
    
    // Remove page numbers (heuristic: standalone numbers)
    let re_page_nums = Regex::new(r"(?m)^\s*\d+\s*$").unwrap();
    let text = re_page_nums.replace_all(&text, "");
    
    // Remove common headers/footers patterns
    let re_headers = Regex::new(r"(?m)^(Page \d+|Chapter \d+)\s*$").unwrap();
    let text = re_headers.replace_all(&text, "");
    
    text.trim().to_string()
}

/// Add structure markers to text
pub fn add_structure_markers(sections: Vec<(String, String)>) -> String {
    let mut result = String::new();
    
    for (title, content) in sections {
        // Add chapter marker
        result.push_str("<CHAPTER>");
        result.push_str(&title);
        result.push_str("</CHAPTER>\n");
        
        // Split content into paragraphs and add markers
        for paragraph in content.split("\n\n") {
            let cleaned = paragraph.trim();
            if !cleaned.is_empty() {
                result.push_str("<PARAGRAPH>");
                result.push_str(cleaned);
                result.push_str("</PARAGRAPH>\n");
            }
        }
        
        result.push('\n');
    }
    
    result
}

/// Extract plain text without structure markers
pub fn remove_structure_markers(text: &str) -> String {
    let re = Regex::new(r"</?(?:CHAPTER|PARAGRAPH)>").unwrap();
    re.replace_all(text, "").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_clean_text() {
        let text = "Hello    World\n\n\n\nTest";
        let cleaned = clean_text(text);
        assert_eq!(cleaned, "Hello World\n\nTest");
    }
    
    #[test]
    fn test_add_structure_markers() {
        let sections = vec![
            ("Chapter 1".to_string(), "This is content.\n\nSecond paragraph.".to_string()),
        ];
        
        let marked = add_structure_markers(sections);
        assert!(marked.contains("<CHAPTER>"));
        assert!(marked.contains("<PARAGRAPH>"));
    }
    
    #[test]
    fn test_remove_structure_markers() {
        let text = "<CHAPTER>Title</CHAPTER><PARAGRAPH>Content</PARAGRAPH>";
        let cleaned = remove_structure_markers(text);
        assert_eq!(cleaned, "TitleContent");
    }
}

