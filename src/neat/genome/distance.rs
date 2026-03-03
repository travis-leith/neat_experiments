use crate::neat::genome::crossover::{for_each_aligned_by_innovation, Alignment};
use crate::neat::genome::types::{DistanceCoefficients, Genome, Innovation};

pub fn genetic_distance(left: &Genome, right: &Genome, c: DistanceCoefficients) -> f64 {
    let max_left = left.innovations().max().unwrap_or(Innovation(0));
    let max_right = right.innovations().max().unwrap_or(Innovation(0));

    let mut excess = 0usize;
    let mut disjoint = 0usize;
    let mut matching = 0usize;
    let mut weight_diff_sum = 0.0f64;

    for_each_aligned_by_innovation(left, right, |aligned| match aligned {
        Alignment::Both(l, r) => {
            matching += 1;
            weight_diff_sum += (l.weight - r.weight).abs();
        }
        Alignment::Left(l) => {
            if l.innovation > max_right {
                excess += 1;
            } else {
                disjoint += 1;
            }
        }
        Alignment::Right(r) => {
            if r.innovation > max_left {
                excess += 1;
            } else {
                disjoint += 1;
            }
        }
    });

    let n = left.connection_count().max(right.connection_count());
    let n_norm = if n < c.small_genome_threshold {
        1.0
    } else {
        n as f64
    };

    let avg_weight_diff = if matching == 0 {
        0.0
    } else {
        weight_diff_sum / (matching as f64)
    };

    c.excess * (excess as f64) / n_norm
        + c.disjoint * (disjoint as f64) / n_norm
        + c.weight * avg_weight_diff
}
