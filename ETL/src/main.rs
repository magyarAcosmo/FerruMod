use anyhow::Result;
use noodles::bam;
use noodles::sam::alignment::record::cigar::op::Kind;
use noodles::sam::alignment::record::data::field::Value;
use noodles::sam::alignment::record::data::field::value::Array;
use noodles::sam::alignment::record::Record;
use polars::prelude::*;
use rayon::prelude::*;
use std::fs::File;

const ML_THRESHOLD: u8 = 0; // Set to 0 to include all mods, adjust as needed

struct FerruRecord {
    chrom: String,
    pos: i64,
    strand: i32,
    cigar: String,
    mod_types: Vec<i32>,
    mod_offsets: Vec<u32>,
    mod_probs: Vec<u32>,
}

fn encode_mod_type(m: u8) -> i32 {
    match m {
        b'm' => 1, b'a' => 0, b'h' => 2, _ => -1,
    }
}

// Custom CIGAR char mapping to avoid Display trait errors
fn kind_to_char(k: Kind) -> char {
    match k {
        Kind::Match => 'M', Kind::Insertion => 'I', Kind::Deletion => 'D',
        Kind::Skip => 'N', Kind::SoftClip => 'S', Kind::HardClip => 'H',
        Kind::Pad => 'P', Kind::SequenceMatch => '=', Kind::SequenceMismatch => 'X',
    }
}

fn process_read(record: &bam::Record, header: &noodles::sam::Header) -> Result<FerruRecord> {
    let pos = record.alignment_start().transpose()?.map(|p| p.get() as i64).unwrap_or(-1);
    
    let chrom = record
        .reference_sequence(header)
        .transpose()?
        .map(|(name, _)| String::from_utf8_lossy(name).to_string())
        .unwrap_or_else(|| "unmapped".to_string());

    let strand = if record.flags().is_reverse_complemented() { 1 } else { 0 };
    
    let mut cigar_str = String::new();
    for op_res in record.cigar().iter() {
        let op = op_res?;
        cigar_str.push_str(&format!("{}{}", op.len(), kind_to_char(op.kind())));
    }

    let mut mod_types = Vec::new();
    let mut mod_offsets = Vec::new();
    let mut mod_probs = Vec::new();

    // 1. Manually extract the raw MM String
    let mut mm_str = String::new();
    if let Some(Ok(Value::String(s))) = record.data().get(&[b'M', b'M']) {
        mm_str = String::from_utf8_lossy(s).into_owned();
    }

    // 2. Manually extract the raw ML Byte Array
    let mut ml_vec: Vec<u8> = Vec::new();
    if let Some(Ok(Value::Array(Array::UInt8(arr)))) = record.data().get(&[b'M', b'L']) {
        for val in arr.iter() {
            if let Ok(v) = val { ml_vec.push(v); }
        }
    }

    // 3. Custom String Parser (Immune to API changes)
    if !mm_str.is_empty() && !ml_vec.is_empty() {
        let mut ml_iter = ml_vec.iter();
        let sequence = record.sequence();

        // Parse "C+m,5,0,9;"
        for group_str in mm_str.split(';') {
            if group_str.is_empty() { continue; }
            let mut parts = group_str.split(',');
            
            let header = parts.next().unwrap_or("");
            if header.len() < 3 { continue; }
            let mod_char = header.chars().next().unwrap_or('N');
            let mod_id_char = header.chars().nth(2).unwrap_or('m');
            let m_id = encode_mod_type(mod_id_char as u8);

            let skips: Vec<usize> = parts.filter_map(|s| s.parse().ok()).collect();
            let mut skip_iter = skips.iter();
            let mut current_skip = skip_iter.next().copied();
            
            let mut read_cursor = 0;
            let mut genome_cursor = pos;

            for op_res in record.cigar().iter() {
                let op = op_res?;
                let len = op.len();
                match op.kind() {
                    Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch => {
                        for _ in 0..len {
                            if let Some(base_byte) = sequence.get(read_cursor) {
                                if (base_byte as char) == mod_char {
                                    if let Some(0) = current_skip {
                                        if let Some(&prob) = ml_iter.next() {
                                            if prob >= ML_THRESHOLD {
                                                mod_types.push(m_id);
                                                mod_offsets.push((genome_cursor - pos) as u32);
                                                mod_probs.push(prob as u32);
                                            }
                                        }
                                        current_skip = skip_iter.next().copied();
                                    } else if let Some(ref mut s) = current_skip {
                                        if *s > 0 { *s -= 1; }
                                    }
                                }
                            }
                            read_cursor += 1;
                            genome_cursor += 1;
                        }
                    }
                    Kind::Insertion => {
                        for _ in 0..len {
                            if let Some(base_byte) = sequence.get(read_cursor) {
                                if (base_byte as char) == mod_char {
                                    if let Some(0) = current_skip {
                                        let _ = ml_iter.next();
                                        current_skip = skip_iter.next().copied();
                                    } else if let Some(ref mut s) = current_skip {
                                        if *s > 0 { *s -= 1; }
                                    }
                                }
                            }
                            read_cursor += 1;
                        }
                    }
                    Kind::Deletion | Kind::Skip => genome_cursor += len as i64,
                    Kind::SoftClip => read_cursor += len,
                    _ => {}
                }
            }
        }
    }

    Ok(FerruRecord {
        chrom, pos, strand, cigar: cigar_str,
        mod_types, mod_offsets, mod_probs,
    })
}

fn main() -> Result<()> {
    // --- DYNAMIC ARGUMENT PARSING ---
    let args: Vec<String> = std::env::args().collect();

    // Check if the correct number of arguments was provided
    if args.len() != 3 {
        eprintln!("Usage: {} <input.bam> <output.parquet>", args[0]);
        std::process::exit(1);
    }

    // Assign variables from CLI
    let input_path = &args[1];
    let output_path = &args[2];
    // --------------------------------
    
    println!("Opening BAM file: {}", input_path);
    let mut reader = bam::io::reader::Builder::default().build_from_path(input_path)?;
    let header = reader.read_header()?;

    println!("Reading records into memory...");
    let records: Vec<_> = reader.records().filter_map(|r| r.ok()).collect();
    
    println!("Processing {} reads in parallel...", records.len());
    let processed: Vec<FerruRecord> = records.par_iter()
        .filter_map(|r| process_read(r, &header).ok())
        .collect();

    println!("Building Polars DataFrame...");
    let c1 = Series::new("chrom", processed.iter().map(|r| r.chrom.as_str()).collect::<Vec<_>>());
    let c2 = Series::new("pos", processed.iter().map(|r| r.pos).collect::<Vec<_>>());
    let c3 = Series::new("strand", processed.iter().map(|r| r.strand).collect::<Vec<_>>());
    let c4 = Series::new("cigar", processed.iter().map(|r| r.cigar.as_str()).collect::<Vec<_>>());
    
    let c5 = Series::new("mod_offsets", processed.iter().map(|r| Series::new("", &r.mod_offsets)).collect::<Vec<_>>());
    let c6 = Series::new("mod_probs", processed.iter().map(|r| Series::new("", &r.mod_probs)).collect::<Vec<_>>());
    let c7 = Series::new("mod_type", processed.iter().map(|r| Series::new("", &r.mod_types)).collect::<Vec<_>>());

    let mut df = DataFrame::new(vec![c1, c2, c3, c4, c5, c6, c7])?;

    println!("Writing to Parquet at {}...", output_path);
    let mut file = File::create(output_path)?;
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Zstd(None))
        .finish(&mut df)?;

    println!("Success! Parquet created with {} records.", processed.len());
    Ok(())
}
