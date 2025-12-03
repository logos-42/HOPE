use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;
use walkdir::WalkDir;

// Import from the main crate (we'll need to adjust paths)
use hope_model::data::{CharTokenizer, Tokenizer};
use hope_model::utils::{auto_ocr_if_needed, extract_text_from_epub, extract_text_from_pdf};
use hope_model::utils::{add_structure_markers, clean_text};

#[derive(Debug, Parser)]
#[command(author, version, about = "Preprocess books (PDF/EPUB) for training")]
struct Args {
    /// Input directory containing PDF/EPUB files
    #[arg(short, long)]
    input: PathBuf,
    
    /// Output directory for preprocessed files
    #[arg(short, long)]
    output: PathBuf,
    
    /// Whether to preserve structure markers
    #[arg(long, default_value = "true")]
    preserve_structure: bool,
    
    /// Enable OCR for scanned PDFs
    #[arg(long, default_value = "false")]
    enable_ocr: bool,
    
    /// Build vocabulary from scratch
    #[arg(long, default_value = "true")]
    build_vocab: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct DocumentMetadata {
    filename: String,
    file_type: String,
    character_count: usize,
    token_count: usize,
    processed_at: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct CorpusMetadata {
    total_documents: usize,
    total_characters: usize,
    total_tokens: usize,
    vocab_size: usize,
    documents: Vec<DocumentMetadata>,
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    
    let args = Args::parse();
    
    info!("Starting book preprocessing");
    info!("Input directory: {:?}", args.input);
    info!("Output directory: {:?}", args.output);
    
    // Create output directory
    fs::create_dir_all(&args.output)
        .with_context(|| format!("Failed to create output directory: {:?}", args.output))?;
    
    // Find all book files
    let mut book_files = Vec::new();
    
    for entry in WalkDir::new(&args.input)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        
        if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy().to_lowercase();
            
            if ext_str == "pdf" || ext_str == "epub" {
                book_files.push(path.to_path_buf());
            }
        }
    }
    
    info!("Found {} book files", book_files.len());
    
    if book_files.is_empty() {
        anyhow::bail!("No book files found in {:?}", args.input);
    }
    
    // Process each book
    let mut all_text = String::new();
    let mut documents = Vec::new();
    
    for (idx, book_path) in book_files.iter().enumerate() {
        info!("Processing {}/{}: {:?}", idx + 1, book_files.len(), book_path);
        
        match process_book(book_path, args.preserve_structure, args.enable_ocr) {
            Ok(text) => {
                let char_count = text.len();
                
                // Save individual document
                let filename = book_path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");
                
                let doc_path = args.output.join(format!("{}.txt", filename));
                fs::write(&doc_path, &text)
                    .with_context(|| format!("Failed to write document: {:?}", doc_path))?;
                
                documents.push(DocumentMetadata {
                    filename: filename.to_string(),
                    file_type: book_path.extension()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string(),
                    character_count: char_count,
                    token_count: 0,  // Will be filled later
                    processed_at: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                });
                
                all_text.push_str(&text);
                all_text.push_str("\n\n");
            }
            Err(e) => {
                warn!("Failed to process {:?}: {}", book_path, e);
            }
        }
    }
    
    if all_text.is_empty() {
        anyhow::bail!("No text extracted from any books");
    }
    
    info!("Total text length: {} characters", all_text.len());
    
    // Build or load tokenizer
    let tokenizer = if args.build_vocab {
        info!("Building vocabulary from corpus...");
        CharTokenizer::from_text(&all_text)
    } else {
        // Try to load existing tokenizer
        let tokenizer_path = args.output.join("vocab.json");
        if tokenizer_path.exists() {
            info!("Loading existing tokenizer...");
            CharTokenizer::load(&tokenizer_path)?
        } else {
            info!("No existing tokenizer found, building new one...");
            CharTokenizer::from_text(&all_text)
        }
    };
    
    info!("Vocabulary size: {}", tokenizer.vocab_size());
    
    // Save tokenizer
    let tokenizer_path = args.output.join("vocab.json");
    tokenizer.save(&tokenizer_path)?;
    info!("Tokenizer saved to: {:?}", tokenizer_path);
    
    // Tokenize the entire corpus
    info!("Tokenizing corpus...");
    let tokens = tokenizer.encode(&all_text);
    info!("Total tokens: {}", tokens.len());
    
    // Save corpus as JSONL
    let corpus_path = args.output.join("corpus.jsonl");
    let mut corpus_file = fs::File::create(&corpus_path)?;
    
    use std::io::Write;
    for (idx, doc_meta) in documents.iter_mut().enumerate() {
        let doc_path = args.output.join(format!("{}.txt", doc_meta.filename));
        let doc_text = fs::read_to_string(&doc_path)?;
        let doc_tokens = tokenizer.encode(&doc_text);
        
        doc_meta.token_count = doc_tokens.len();
        
        let json_line = serde_json::json!({
            "id": idx,
            "filename": doc_meta.filename,
            "text": doc_text,
            "tokens": doc_tokens,
        });
        
        writeln!(corpus_file, "{}", serde_json::to_string(&json_line)?)?;
    }
    
    info!("Corpus saved to: {:?}", corpus_path);
    
    // Save metadata
    let metadata = CorpusMetadata {
        total_documents: documents.len(),
        total_characters: all_text.len(),
        total_tokens: tokens.len(),
        vocab_size: tokenizer.vocab_size(),
        documents,
    };
    
    let metadata_path = args.output.join("metadata.json");
    let metadata_json = serde_json::to_string_pretty(&metadata)?;
    fs::write(&metadata_path, metadata_json)?;
    info!("Metadata saved to: {:?}", metadata_path);
    
    info!("Preprocessing complete!");
    info!("Summary:");
    info!("  - Documents: {}", metadata.total_documents);
    info!("  - Characters: {}", metadata.total_characters);
    info!("  - Tokens: {}", metadata.total_tokens);
    info!("  - Vocabulary size: {}", metadata.vocab_size);
    
    Ok(())
}

fn process_book(path: &Path, preserve_structure: bool, enable_ocr: bool) -> Result<String> {
    let ext = path.extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    
    let text = match ext.as_str() {
        "pdf" => {
            if enable_ocr {
                // Try OCR if needed
                auto_ocr_if_needed(path)?
            } else {
                let content = extract_text_from_pdf(path)?;
                
                if !content.has_text {
                    anyhow::bail!("PDF has no extractable text (enable OCR with --enable-ocr)");
                }
                
                if preserve_structure {
                    match hope_model::utils::pdf_parser::extract_structured_content(path) {
                        Ok(sections) => add_structure_markers(sections),
                        Err(_) => clean_text(&content.text),
                    }
                } else {
                    clean_text(&content.text)
                }
            }
        }
        "epub" => {
            let content = extract_text_from_epub(path)?;
            
            if preserve_structure {
                add_structure_markers(content.chapters)
            } else {
                content.chapters
                    .into_iter()
                    .map(|(_, text)| clean_text(&text))
                    .collect::<Vec<_>>()
                    .join("\n\n")
            }
        }
        _ => {
            anyhow::bail!("Unsupported file format: {}", ext);
        }
    };
    
    Ok(text)
}

