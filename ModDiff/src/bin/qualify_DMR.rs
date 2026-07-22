use anyhow::{bail, Context, Result};
use duckdb::{params, Connection};
use std::collections::HashMap;
use std::env;

const REQUIRED_SAMPLE_FRACTION: f64 = 0.80;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Group {
    Control,
    Case,
}

#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    sites: f64,
    calls: f64,
    modified: f64,
}

#[derive(Debug, Default)]
struct WindowCounts {
    chrom: String,
    start: i64,
    end: i64,
    samples: HashMap<String, Counts>,
}

#[derive(Clone, Debug)]
struct QualifiedWindow {
    chrom: String,
    start: i64,
    end: i64,
    meth_diff: f64,
    case_coverage: f64,
    control_coverage: f64,
    case_passing_pct: f64,
    control_passing_pct: f64,
    case_variance: f64,
    control_variance: f64,
}

#[derive(Debug)]
struct QualifiedDmr {
    chrom: String,
    start: i64,
    end: i64,
    window_count: usize,
    meth_diff: f64,
    case_coverage: f64,
    control_coverage: f64,
    case_passing_pct: f64,
    control_passing_pct: f64,
    case_variance: f64,
    control_variance: f64,
}

#[derive(Clone, Copy)]
struct Thresholds {
    min_meth_diff: f64,
    min_group_coverage: f64,
    min_sample_coverage: f64,
    max_variance: f64,
}

#[derive(Clone, Copy, Debug)]
struct FilterResults {
    meth_diff: bool,
    case_group_coverage: bool,
    control_group_coverage: bool,
    case_sample_coverage: bool,
    control_sample_coverage: bool,
    case_variance: bool,
    control_variance: bool,
}

impl FilterResults {
    fn group_coverage(self) -> bool {
        self.case_group_coverage && self.control_group_coverage
    }

    fn sample_coverage(self) -> bool {
        self.case_sample_coverage && self.control_sample_coverage
    }

    fn variance(self) -> bool {
        self.case_variance && self.control_variance
    }

    fn all(self) -> bool {
        self.meth_diff
            && self.group_coverage()
            && self.sample_coverage()
            && self.variance()
    }
}

#[derive(Debug, Default)]
struct FilterStats {
    total: usize,
    usable: usize,
    meth_diff: usize,
    case_group_coverage: usize,
    control_group_coverage: usize,
    group_coverage: usize,
    case_sample_coverage: usize,
    control_sample_coverage: usize,
    sample_coverage: usize,
    case_variance: usize,
    control_variance: usize,
    variance: usize,
    after_meth_diff: usize,
    after_group_coverage: usize,
    after_sample_coverage: usize,
    after_variance: usize,
}

impl FilterStats {
    fn observe(&mut self, filters: FilterResults) {
        self.usable += 1;
        self.meth_diff += usize::from(filters.meth_diff);
        self.case_group_coverage += usize::from(filters.case_group_coverage);
        self.control_group_coverage += usize::from(filters.control_group_coverage);
        self.group_coverage += usize::from(filters.group_coverage());
        self.case_sample_coverage += usize::from(filters.case_sample_coverage);
        self.control_sample_coverage += usize::from(filters.control_sample_coverage);
        self.sample_coverage += usize::from(filters.sample_coverage());
        self.case_variance += usize::from(filters.case_variance);
        self.control_variance += usize::from(filters.control_variance);
        self.variance += usize::from(filters.variance());

        let after_meth_diff = filters.meth_diff;
        let after_group_coverage = after_meth_diff && filters.group_coverage();
        let after_sample_coverage = after_group_coverage && filters.sample_coverage();
        let after_variance = after_sample_coverage && filters.variance();
        self.after_meth_diff += usize::from(after_meth_diff);
        self.after_group_coverage += usize::from(after_group_coverage);
        self.after_sample_coverage += usize::from(after_sample_coverage);
        self.after_variance += usize::from(after_variance);
    }
}

#[derive(Debug)]
struct QualificationRun {
    qualified: Vec<QualifiedWindow>,
    stats: FilterStats,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 7 {
        eprintln!(
            "Usage: qualify_DMR <mod.db> <metadata.csv> <min_meth_diff> \
             <min_group_coverage> <min_sample_coverage> <max_variance>"
        );
        std::process::exit(1);
    }

    let db_path = &args[1];
    let metadata_path = &args[2];
    let thresholds = Thresholds {
        min_meth_diff: parse_nonnegative(&args[3], "min_meth_diff")?,
        min_group_coverage: parse_nonnegative(&args[4], "min_group_coverage")?,
        min_sample_coverage: parse_nonnegative(&args[5], "min_sample_coverage")?,
        max_variance: parse_nonnegative(&args[6], "max_variance")?,
    };
    if thresholds.min_meth_diff > 1.0 {
        bail!("min_meth_diff must be between 0 and 1");
    }

    let (metadata, case_count, control_count) = read_metadata(metadata_path)?;
    println!(
        "Loaded {} case and {} control samples from metadata.",
        case_count, control_count
    );

    let conn = Connection::open(db_path)
        .with_context(|| format!("failed to open DuckDB database {db_path}"))?;
    let temp_dir = format!("{}.tmp", db_path);
    conn.execute_batch(&format!(
        "PRAGMA threads=8; PRAGMA temp_directory='{}';",
        temp_dir.replace('\'', "''")
    ))?;

    println!("Evaluating windows...");
    let run = qualify_windows(
        &conn,
        &metadata,
        case_count,
        control_count,
        thresholds,
    )?;
    print_filter_stats(&run.stats, thresholds);
    println!("Qualified {} windows.", run.qualified.len());

    let dmrs = collapse_windows(run.qualified);
    save_dmrs(&conn, &dmrs)?;
    println!(
        "Saved {} collapsed regions to 'qualified_DMRs'.",
        dmrs.len()
    );
    Ok(())
}

fn parse_nonnegative(value: &str, name: &str) -> Result<f64> {
    let parsed: f64 = value
        .parse()
        .with_context(|| format!("{name} must be a number"))?;
    if !parsed.is_finite() || parsed < 0.0 {
        bail!("{name} must be a finite, non-negative number");
    }
    Ok(parsed)
}

fn read_metadata(path: &str) -> Result<(HashMap<String, Group>, usize, usize)> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open metadata CSV {path}"))?;
    let mut metadata = HashMap::new();
    let (mut cases, mut controls) = (0, 0);

    for (index, result) in reader.records().enumerate() {
        let record = result.with_context(|| format!("invalid metadata row {}", index + 2))?;
        if record.len() < 2 {
            bail!("metadata row {} needs sample and group columns", index + 2);
        }
        let sample = record[0].trim();
        if sample.is_empty() {
            bail!("metadata row {} has an empty sample name", index + 2);
        }
        let group_value: f64 = record[1]
            .trim()
            .parse()
            .with_context(|| format!("invalid group for sample {sample}"))?;
        if !group_value.is_finite() {
            bail!("group for sample {sample} must be finite");
        }
        let group = if group_value == 0.0 {
            controls += 1;
            Group::Control
        } else {
            cases += 1;
            Group::Case
        };
        if metadata.insert(sample.to_owned(), group).is_some() {
            bail!("sample {sample} appears more than once in metadata");
        }
    }
    if cases == 0 || controls == 0 {
        bail!("metadata must contain at least one case and one control sample");
    }
    Ok((metadata, cases, controls))
}

fn qualify_windows(
    conn: &Connection,
    metadata: &HashMap<String, Group>,
    case_count: usize,
    control_count: usize,
    thresholds: Thresholds,
) -> Result<QualificationRun> {
    let mut statement = conn.prepare(
        r#"SELECT chrom, start, "end", sample_name,
                  CAST(num_sites AS DOUBLE), CAST(num_calls AS DOUBLE),
                  CAST(mod_counts AS DOUBLE)
           FROM windows
           WHERE chrom IS NOT NULL AND start IS NOT NULL AND "end" IS NOT NULL
             AND sample_name IS NOT NULL
           ORDER BY chrom, start, "end", sample_name"#,
    )?;
    let mut rows = statement.query([])?;
    let mut current: Option<WindowCounts> = None;
    let mut qualified = Vec::new();
    let mut stats = FilterStats::default();

    while let Some(row) = rows.next()? {
        let chrom: String = row.get(0)?;
        let start: i64 = row.get(1)?;
        let end: i64 = row.get(2)?;
        let sample: String = row.get(3)?;
        let counts = Counts {
            sites: row.get(4)?,
            calls: row.get(5)?,
            modified: row.get(6)?,
        };

        let is_new = current
            .as_ref()
            .map(|window| window.chrom != chrom || window.start != start || window.end != end)
            .unwrap_or(false);
        if is_new {
            if let Some(window) = current.take() {
                evaluate_window(
                    window,
                    metadata,
                    case_count,
                    control_count,
                    thresholds,
                    &mut qualified,
                    &mut stats,
                );
            }
        }

        let window = current.get_or_insert_with(|| WindowCounts {
            chrom,
            start,
            end,
            samples: HashMap::new(),
        });
        let total = window.samples.entry(sample).or_default();
        total.sites += counts.sites;
        total.calls += counts.calls;
        total.modified += counts.modified;
    }

    if let Some(window) = current {
        evaluate_window(
            window,
            metadata,
            case_count,
            control_count,
            thresholds,
            &mut qualified,
            &mut stats,
        );
    }
    Ok(QualificationRun { qualified, stats })
}

fn evaluate_window(
    window: WindowCounts,
    metadata: &HashMap<String, Group>,
    case_count: usize,
    control_count: usize,
    thresholds: Thresholds,
    qualified: &mut Vec<QualifiedWindow>,
    stats: &mut FilterStats,
) {
    stats.total += 1;
    if let Some((result, filters)) = evaluate_one(
        window,
        metadata,
        case_count,
        control_count,
        thresholds,
    ) {
        stats.observe(filters);
        if filters.all() {
            qualified.push(result);
        }
    }
}

fn evaluate_one(
    window: WindowCounts,
    metadata: &HashMap<String, Group>,
    case_count: usize,
    control_count: usize,
    thresholds: Thresholds,
) -> Option<(QualifiedWindow, FilterResults)> {
    let mut case_total = Counts::default();
    let mut control_total = Counts::default();
    let mut case_methylation = Vec::new();
    let mut control_methylation = Vec::new();
    let (mut case_passing, mut control_passing) = (0usize, 0usize);
    // num_sites describes the window, so count it once when calculating
    // pooled group coverage. MAX is deterministic if sparse samples report
    // fewer covered sites than other samples in the same window.
    let mut window_sites = 0.0_f64;

    for (sample, counts) in &window.samples {
        let Some(group) = metadata.get(sample) else {
            continue;
        };
        if !counts.sites.is_finite()
            || !counts.calls.is_finite()
            || !counts.modified.is_finite()
            || counts.sites <= 0.0
            || counts.calls < 0.0
            || counts.modified < 0.0
            || counts.modified > counts.calls
        {
            continue;
        }
        let sample_coverage = counts.calls / counts.sites;
        let methylation = (counts.calls > 0.0).then_some(counts.modified / counts.calls);
        window_sites = window_sites.max(counts.sites);

        match group {
            Group::Case => {
                add_group_counts(&mut case_total, counts);
                case_passing += usize::from(sample_coverage >= thresholds.min_sample_coverage);
                if let Some(value) = methylation {
                    case_methylation.push(value);
                }
            }
            Group::Control => {
                add_group_counts(&mut control_total, counts);
                control_passing +=
                    usize::from(sample_coverage >= thresholds.min_sample_coverage);
                if let Some(value) = methylation {
                    control_methylation.push(value);
                }
            }
        }
    }

    if window_sites <= 0.0
        || case_total.calls <= 0.0
        || control_total.calls <= 0.0
    {
        return None;
    }

    let case_coverage = case_total.calls / window_sites;
    let control_coverage = control_total.calls / window_sites;
    let case_fraction = case_passing as f64 / case_count as f64;
    let control_fraction = control_passing as f64 / control_count as f64;
    let case_variance = population_variance(&case_methylation);
    let control_variance = population_variance(&control_methylation);
    let meth_diff = case_total.modified / case_total.calls
        - control_total.modified / control_total.calls;

    let filters = FilterResults {
        meth_diff: meth_diff.abs() >= thresholds.min_meth_diff,
        case_group_coverage: case_coverage >= thresholds.min_group_coverage,
        control_group_coverage: control_coverage >= thresholds.min_group_coverage,
        case_sample_coverage: case_fraction >= REQUIRED_SAMPLE_FRACTION,
        control_sample_coverage: control_fraction >= REQUIRED_SAMPLE_FRACTION,
        case_variance: case_variance <= thresholds.max_variance,
        control_variance: control_variance <= thresholds.max_variance,
    };

    Some((
        QualifiedWindow {
            chrom: window.chrom,
            start: window.start,
            end: window.end,
            meth_diff,
            case_coverage,
            control_coverage,
            case_passing_pct: 100.0 * case_fraction,
            control_passing_pct: 100.0 * control_fraction,
            case_variance,
            control_variance,
        },
        filters,
    ))
}

#[cfg(test)]
fn qualify_one(
    window: WindowCounts,
    metadata: &HashMap<String, Group>,
    case_count: usize,
    control_count: usize,
    thresholds: Thresholds,
) -> Option<QualifiedWindow> {
    evaluate_one(window, metadata, case_count, control_count, thresholds)
        .and_then(|(result, filters)| filters.all().then_some(result))
}

fn print_filter_stats(stats: &FilterStats, thresholds: Thresholds) {
    println!("Filter diagnostics (candidate-window counts):");
    println!("  Total windows read: {}", stats.total);
    println!(
        "  Usable data in both groups: {} ({} excluded before threshold checks)",
        stats.usable,
        stats.total - stats.usable
    );
    println!(
        "  {:<68} {:>12} {:>12} {:>12}",
        "Threshold", "Pass alone", "Removed here", "Remaining"
    );
    println!(
        "  {:<68} {:>12} {:>12} {:>12}",
        format!(
            "min_meth_diff: |methylation difference| >= {}",
            thresholds.min_meth_diff
        ),
        stats.meth_diff,
        stats.usable - stats.after_meth_diff,
        stats.after_meth_diff
    );
    println!(
        "  {:<68} {:>12} {:>12} {:>12}",
        format!(
            "min_group_coverage: coverage >= {} in both groups",
            thresholds.min_group_coverage
        ),
        stats.group_coverage,
        stats.after_meth_diff - stats.after_group_coverage,
        stats.after_group_coverage
    );
    println!(
        "    case: {}; control: {}",
        stats.case_group_coverage, stats.control_group_coverage
    );
    println!(
        "  {:<68} {:>12} {:>12} {:>12}",
        format!(
            "min_sample_coverage: coverage >= {} in >= {:.0}% of both groups",
            thresholds.min_sample_coverage,
            100.0 * REQUIRED_SAMPLE_FRACTION
        ),
        stats.sample_coverage,
        stats.after_group_coverage - stats.after_sample_coverage,
        stats.after_sample_coverage
    );
    println!(
        "    case: {}; control: {}",
        stats.case_sample_coverage, stats.control_sample_coverage
    );
    println!(
        "  {:<68} {:>12} {:>12} {:>12}",
        format!(
            "max_variance: inter-sample variance <= {} in both groups",
            thresholds.max_variance
        ),
        stats.variance,
        stats.after_sample_coverage - stats.after_variance,
        stats.after_variance
    );
    println!(
        "    case: {}; control: {}",
        stats.case_variance, stats.control_variance
    );
    println!("  'Pass alone' uses all windows with usable case/control data.");
    println!(
        "  'Removed here' and 'Remaining' apply thresholds cumulatively top to bottom."
    );
}

fn add_group_counts(total: &mut Counts, value: &Counts) {
    total.calls += value.calls;
    total.modified += value.modified;
}

fn population_variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / values.len() as f64
}

fn collapse_windows(mut windows: Vec<QualifiedWindow>) -> Vec<QualifiedDmr> {
    windows.sort_by(|a, b| {
        a.chrom
            .cmp(&b.chrom)
            .then(a.start.cmp(&b.start))
            .then(a.end.cmp(&b.end))
    });
    let mut dmrs = Vec::new();
    let mut run: Vec<QualifiedWindow> = Vec::new();
    let mut run_end: Option<i64> = None;

    for window in windows {
        let contiguous = run
            .first()
            .map(|first| {
                first.chrom == window.chrom
                    && window.start <= run_end.expect("a non-empty run has an end")
            })
            .unwrap_or(true);
        if !contiguous {
            dmrs.push(summarize_run(&run));
            run.clear();
            run_end = None;
        }
        run_end = Some(run_end.map_or(window.end, |end| end.max(window.end)));
        run.push(window);
    }
    if !run.is_empty() {
        dmrs.push(summarize_run(&run));
    }
    dmrs
}

fn summarize_run(run: &[QualifiedWindow]) -> QualifiedDmr {
    let count = run.len() as f64;
    QualifiedDmr {
        chrom: run[0].chrom.clone(),
        start: run.iter().map(|window| window.start).min().unwrap(),
        end: run.iter().map(|window| window.end).max().unwrap(),
        window_count: run.len(),
        meth_diff: run.iter().map(|window| window.meth_diff).sum::<f64>() / count,
        case_coverage: run.iter().map(|window| window.case_coverage).sum::<f64>() / count,
        control_coverage: run.iter().map(|window| window.control_coverage).sum::<f64>() / count,
        case_passing_pct: run.iter().map(|window| window.case_passing_pct).sum::<f64>() / count,
        control_passing_pct: run.iter().map(|window| window.control_passing_pct).sum::<f64>() / count,
        case_variance: run.iter().map(|window| window.case_variance).sum::<f64>() / count,
        control_variance: run.iter().map(|window| window.control_variance).sum::<f64>() / count,
    }
}

fn save_dmrs(conn: &Connection, dmrs: &[QualifiedDmr]) -> Result<()> {
    conn.execute_batch(
        r#"DROP TABLE IF EXISTS qualified_DMRs;
           CREATE TABLE qualified_DMRs (
               chrom VARCHAR,
               start BIGINT,
               "end" BIGINT,
               window_count BIGINT,
               meth_diff DOUBLE,
               case_overall_coverage DOUBLE,
               control_overall_coverage DOUBLE,
               case_samples_passing_pct DOUBLE,
               control_samples_passing_pct DOUBLE,
               case_inter_sample_variance DOUBLE,
               control_inter_sample_variance DOUBLE
           );"#,
    )?;
    let mut insert = conn.prepare(
        r#"INSERT INTO qualified_DMRs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
    )?;
    for dmr in dmrs {
        insert.execute(params![
            &dmr.chrom,
            dmr.start,
            dmr.end,
            dmr.window_count as i64,
            dmr.meth_diff,
            dmr.case_coverage,
            dmr.control_coverage,
            dmr.case_passing_pct,
            dmr.control_passing_pct,
            dmr.case_variance,
            dmr.control_variance,
        ])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thresholds() -> Thresholds {
        Thresholds {
            min_meth_diff: 0.25,
            min_group_coverage: 2.0,
            min_sample_coverage: 2.0,
            max_variance: 0.01,
        }
    }

    #[test]
    fn qualifies_using_group_and_per_sample_coverage() {
        let metadata = HashMap::from([
            ("case1".to_string(), Group::Case),
            ("case2".to_string(), Group::Case),
            ("ctrl1".to_string(), Group::Control),
            ("ctrl2".to_string(), Group::Control),
        ]);
        let window = WindowCounts {
            chrom: "chr1".into(),
            start: 0,
            end: 100,
            samples: HashMap::from([
                (
                    "case1".into(),
                    Counts {
                        sites: 10.0,
                        calls: 30.0,
                        modified: 24.0,
                    },
                ),
                (
                    "case2".into(),
                    Counts {
                        sites: 10.0,
                        calls: 20.0,
                        modified: 16.0,
                    },
                ),
                (
                    "ctrl1".into(),
                    Counts {
                        sites: 10.0,
                        calls: 20.0,
                        modified: 8.0,
                    },
                ),
                (
                    "ctrl2".into(),
                    Counts {
                        sites: 10.0,
                        calls: 20.0,
                        modified: 8.0,
                    },
                ),
            ]),
        };
        let result = qualify_one(window, &metadata, 2, 2, thresholds()).unwrap();
        assert!((result.meth_diff - 0.4).abs() < 1e-12);
        assert_eq!(result.case_coverage, 5.0);
        assert_eq!(result.control_coverage, 4.0);
        assert_eq!(result.case_passing_pct, 100.0);
        assert_eq!(result.case_variance, 0.0);
    }

    #[test]
    fn missing_metadata_sample_fails_the_eighty_percent_rule() {
        let mut metadata = HashMap::new();
        for index in 0..5 {
            metadata.insert(format!("case{index}"), Group::Case);
            metadata.insert(format!("ctrl{index}"), Group::Control);
        }
        let mut samples = HashMap::new();
        for index in 0..4 {
            samples.insert(
                format!("case{index}"),
                Counts {
                    sites: 10.0,
                    calls: 20.0,
                    modified: 18.0,
                },
            );
            samples.insert(
                format!("ctrl{index}"),
                Counts {
                    sites: 10.0,
                    calls: 20.0,
                    modified: 2.0,
                },
            );
        }
        assert!(qualify_one(
            WindowCounts {
                chrom: "chr1".into(),
                start: 0,
                end: 100,
                samples: samples.clone()
            },
            &metadata,
            5,
            5,
            thresholds(),
        ).is_some());
        samples.remove("case3");
        assert!(qualify_one(
            WindowCounts {
                chrom: "chr1".into(),
                start: 0,
                end: 100,
                samples
            },
            &metadata,
            5,
            5,
            thresholds(),
        ).is_none());
    }

    #[test]
    fn filter_stats_track_independent_and_cumulative_counts() {
        let mut stats = FilterStats::default();
        stats.total = 3;
        stats.observe(FilterResults {
            meth_diff: true,
            case_group_coverage: true,
            control_group_coverage: true,
            case_sample_coverage: false,
            control_sample_coverage: true,
            case_variance: true,
            control_variance: true,
        });
        stats.observe(FilterResults {
            meth_diff: false,
            case_group_coverage: true,
            control_group_coverage: true,
            case_sample_coverage: true,
            control_sample_coverage: true,
            case_variance: true,
            control_variance: true,
        });

        assert_eq!(stats.usable, 2);
        assert_eq!(stats.meth_diff, 1);
        assert_eq!(stats.group_coverage, 2);
        assert_eq!(stats.sample_coverage, 1);
        assert_eq!(stats.variance, 2);
        assert_eq!(stats.after_meth_diff, 1);
        assert_eq!(stats.after_group_coverage, 1);
        assert_eq!(stats.after_sample_coverage, 0);
        assert_eq!(stats.after_variance, 0);
    }

    #[test]
    fn collapses_only_touching_or_overlapping_windows_on_same_chromosome() {
        fn window(chrom: &str, start: i64, end: i64) -> QualifiedWindow {
            QualifiedWindow {
                chrom: chrom.into(),
                start,
                end,
                meth_diff: 0.5,
                case_coverage: 3.0,
                control_coverage: 3.0,
                case_passing_pct: 100.0,
                control_passing_pct: 100.0,
                case_variance: 0.0,
                control_variance: 0.0,
            }
        }
        let dmrs = collapse_windows(vec![
            window("chr1", 50, 150),
            window("chr2", 0, 100),
            window("chr1", 0, 100),
            window("chr1", 151, 250),
        ]);
        assert_eq!(dmrs.len(), 3);
        assert_eq!(dmrs[0].start, 0);
        assert_eq!(dmrs[0].end, 150);
        assert_eq!(dmrs[0].window_count, 2);
    }
}
