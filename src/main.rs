use anyhow::{Context, Result};
use noodles::bam;
use noodles::sam::record::data::field::tag;
use polars::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;

// Threshold: only store modifications with ML > 3 (~1.2% probability)
// This aggressively shrinks file size while keeping almost all signal.
const ML_THRESHOLD: u8 = 3;

struct FerruRecord {
    chrom: String,
    pos: i64,
    strand: i8,
    cigar: String,
    mod_types: Vec<i8>,
    mod_offsets: Vec<u32>,
    mod_probs: Vec<u8>,
    qual: Vec<u8>,
    extra_tags: Vec<u8>,
}

fn encode_mod_type(m: &[u8]) -> i8 {
    match m {
        b"m" => 1, // 5mC
        b"a" => 0, // m6A
        b"h" => 2, // 5hmC
        _ => -1,   // Other
    }
}

fn process_read(record: &bam::Record, header: &noodles::sam::Header) -> Result<FerruRecord> {
    let chrom = record
        .reference_sequence(header)
        .and_then(|s| s.ok())
        .map(|(name, _)| name.to_string())
        .unwrap_or_else(|| "unmapped".to_string());

    let pos = record.alignment_start().map(|p| p.get() as i64).unwrap_or(-1);
    let is_reverse = record.flags().is_reverse_complemented();
    let strand = if is_reverse { 1 } else { 0 };
    let cigar_str = record.cigar().to_string();
    let qual = record.quality_scores().as_ref().to_vec();

    let mut mod_types = Vec::new();
    let mut mod_offsets = Vec::new();
    let mut mod_probs = Vec::new();

    // Access MM and ML tags
    if let (Some(Ok(mm_groups)), Some(Ok(ml_data))) = (
        record.data().get(&tag::BASE_MODIFICATIONS).map(|v| v.as_base_modifications()),
        record.data().get(&tag::BASE_MODIFICATION_PROBABILITIES).map(|v| v.as_uint8_array())
    ) {
        let mut ml_iter = ml_data.iter();
        let sequence = record.sequence();

        for group in mm_groups.iter() {
            let mod_char = group.base() as char;
            let m_id = group.modifications().first().map(|&m| encode_mod_type(&[m])).unwrap_or(-1);

            let mut skip_iter = group.unmodified_occurrence_counts().iter();
            let mut current_skip = skip_iter.next().map(|&s| s as usize);
            
            let mut read_cursor = 0;
            let mut genome_cursor = pos;

            for op in record.cigar().iter() {
                let len = op.len();
                match op.kind() {
                    noodles::sam::record::cigar::op::Kind::Match | 
                    noodles::sam::record::cigar::op::Kind::SequenceMatch | 
                    noodles::sam::record::cigar::op::Kind::SequenceMismatch => {
                        for _ in 0..len {
                            let base = sequence.get(read_cursor).map(|b| b as char).unwrap_or('N');
                            if base == mod_char {
                                if let Some(0) = current_skip {
                                    if let Some(&prob) = ml_iter.next() {
                                        if prob >= ML_THRESHOLD {
                                            mod_types.push(m_id);
                                            // Store as relative offset from read start for efficiency
                                            mod_offsets.push((genome_cursor - pos) as u32);
                                            mod_probs.push(prob);
                                        }
                                    }
                                    current_skip = skip_iter.next().map(|&s| s as usize);
                                } else if let Some(ref mut s) = current_skip {
                                    *s -= 1;
                                }
                            }
                            read_cursor += 1;
                            genome_cursor += 1;
                        }
                    }
                    noodles::sam::record::cigar::op::Kind::Insertion => {
                        for _ in 0..len {
                            let base = sequence.get(read_cursor).map(|b| b as char).unwrap_or('N');
                            if base == mod_char {
                                if let Some(0) = current_skip {
                                    let _ = ml_iter.next(); // Skip ML value for insertions
                                    current_skip = skip_iter.next().map(|&s| s as usize);
                                } else if let Some(ref mut s) = current_skip {
                                    *s -= 1;
                                }
                            }
                            read_cursor += 1;
                        }
                    }
                    noodles::sam::record::cigar::op::Kind::Deletion | 
                    noodles::sam::record::cigar::op::Kind::Skip => {
                        genome_cursor += len as i64;
                    }
                    noodles::sam::record::cigar::op::Kind::SoftClip => {
                        read_cursor += len;
                    }
                    _ => {}
                }
            }
        }
    }

    // Capture extra tags (excluding MM/ML which we columnarized)
    let mut extra_tags = Vec::new();
    for result in record.data().iter() {
        let (tag, value) = result?;
        if tag != tag::BASE_MODIFICATIONS && tag != tag::BASE_MODIFICATION_PROBABILITIES {
            extra_tags.extend_from_slice(tag.as_ref());
            // This is a simplified binary storage; for full lossless, 
            // you'd use the raw BAM internal tag buffer.
        }
    }

    Ok(FerruRecord {
        chrom, pos, strand, cigar: cigar_str,
        mod_types, mod_offsets, mod_probs,
        qual, extra_tags,
    })
}

fn main() -> Result<()> {
    let input_path = "input.bam";
    let output_path = "output.ferrumod.parquet";

    let mut reader = bam::io::reader::Builder::default().build_from_path(input_path)?;
    let header = reader.read_header()?;

    println!("Reading BAM into memory...");
    let records: Vec<bam::Record> = reader.records().map(|r| r.unwrap()).collect();

    println!("Processing {} reads in parallel...", records.len());
    let processed_data: Vec<FerruRecord> = records
        .par_iter()
        .map(|r| process_read(r, &header).unwrap())
        .collect();

    println!("Building optimized DataFrame...");
    
    // Convert to Polars Series with precise typing
    let df = df!(
        "chrom" => processed_data.iter().map(|r| r.chrom.as_str()).collect::<Vec<_>>(),
        "pos" => processed_data.iter().map(|r| r.pos).collect::<Vec<_>>(),
        "strand" => processed_data.iter().map(|r| r.strand).collect::<Vec<_>>(),
        "cigar" => processed_data.iter().map(|r| r.cigar.as_str()).collect::<Vec<_>>(),
        "mod_type" => Series::new("mod_type", processed_data.iter().map(|r| Series::new("", &r.mod_types)).collect::<Vec<_>>()),
        "mod_offsets" => Series::new("mod_offsets", processed_data.iter().map(|r| Series::new("", &r.mod_offsets)).collect::<Vec<_>>()),
        "mod_probs" => Series::new("mod_probs", processed_data.iter().map(|r| Series::new("", &r.mod_probs)).collect::<Vec<_>>()),
        "qual" => processed_data.iter().map(|r| r.qual.as_slice()).collect::<Vec<_>>(),
        "extra_tags" => processed_data.iter().map(|r| r.extra_tags.as_slice()).collect::<Vec<_>>()
    )?;

    println!("Writing to compressed Parquet...");
    let mut file = File::create(output_path)?;
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Zstd(Some(ZstdLevel::try_new(3)?)))
        .with_statistics(true)
        .finish(&mut df.clone())?;

    println!("Done! Output saved to {}", output_path);
    Ok(())
}
