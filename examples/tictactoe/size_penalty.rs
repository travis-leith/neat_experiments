/// A penalty applied to fitness based on network size (node count).
/// All penalty functions return a value in [0.0, 1.0] that is multiplied
/// with the raw fitness to produce the final fitness.
///
/// A return value of 1.0 means no penalty; 0.0 means total penalty.

/// No penalty — baseline.
pub fn no_penalty(_node_count: usize, _generation: usize) -> f64 {
    1.0
}

/// Constant linear penalty: fitness is scaled by `1 - rate * node_count`.
/// Clamped to [0.0, 1.0].
pub fn linear(rate: f64) -> impl Fn(usize, usize) -> f64 {
    move |node_count, _generation| (1.0 - rate * node_count as f64).clamp(0.0, 1.0)
}

/// Threshold penalty: no penalty below `threshold` nodes, then linear penalty
/// at the given `rate` per node above the threshold.
///
/// `penalty_factor(n) = 1.0 - rate * max(0, n - threshold)`
pub fn threshold(threshold: usize, rate: f64) -> impl Fn(usize, usize) -> f64 {
    move |node_count, _generation| {
        let excess = node_count.saturating_sub(threshold) as f64;
        (1.0 - rate * excess).clamp(0.0, 1.0)
    }
}

/// Seasonal penalty simulating winter/summer cycles.
///
/// In "summer" (cycle trough), the penalty is mild — networks can grow freely.
/// In "winter" (cycle peak), the penalty is harsh — bloated networks are punished.
///
/// The penalty oscillates sinusoidally between `min_rate` and `max_rate` per node,
/// with a full cycle every `period` generations.
///
/// `effective_rate = min_rate + (max_rate - min_rate) * (1 + cos(2π * gen / period)) / 2`
///
/// Note: cos peaks at gen=0, so generation 0 starts in "winter" (max penalty).
/// Shift by half a period if you want to start in summer.
pub fn seasonal(period: usize, min_rate: f64, max_rate: f64) -> impl Fn(usize, usize) -> f64 {
    move |node_count, generation| {
        let phase = 2.0 * std::f64::consts::PI * (generation as f64) / (period as f64);
        let seasonal_factor = (1.0 + phase.cos()) / 2.0; // 1.0 at winter, 0.0 at summer
        let effective_rate = min_rate + (max_rate - min_rate) * seasonal_factor;
        (1.0 - effective_rate * node_count as f64).clamp(0.0, 1.0)
    }
}

/// Seasonal penalty with a threshold: no penalty below `threshold` nodes,
/// then seasonal scaling on the excess.
pub fn seasonal_with_threshold(
    threshold: usize,
    period: usize,
    min_rate: f64,
    max_rate: f64,
) -> impl Fn(usize, usize) -> f64 {
    let seasonal_fn = seasonal(period, min_rate, max_rate);
    move |node_count, generation| {
        if node_count <= threshold {
            1.0
        } else {
            let excess_count = node_count - threshold;
            seasonal_fn(excess_count, generation)
        }
    }
}

/// Exponential decay penalty: `penalty_factor = e^(-rate * node_count)`.
/// Gentle for small networks, increasingly harsh for large ones.
pub fn exponential(rate: f64) -> impl Fn(usize, usize) -> f64 {
    move |node_count, _generation| (-rate * node_count as f64).exp()
}

/// Exponential decay with a threshold: no penalty below `threshold`,
/// then exponential decay on the excess.
pub fn exponential_with_threshold(threshold: usize, rate: f64) -> impl Fn(usize, usize) -> f64 {
    move |node_count, _generation| {
        let excess = node_count.saturating_sub(threshold) as f64;
        (-rate * excess).exp()
    }
}

/// Step function: full fitness below `threshold`, then a flat multiplier above it.
/// Useful as a hard cap rather than a gradient.
pub fn step(threshold: usize, penalty_multiplier: f64) -> impl Fn(usize, usize) -> f64 {
    move |node_count, _generation| {
        if node_count <= threshold {
            1.0
        } else {
            penalty_multiplier.clamp(0.0, 1.0)
        }
    }
}

/// Apply a size penalty to a raw fitness value.
#[inline]
pub fn apply<F: Fn(usize, usize) -> f64>(
    raw_fitness: f64,
    node_count: usize,
    generation: usize,
    penalty_fn: &F,
) -> f64 {
    raw_fitness * penalty_fn(node_count, generation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_penalty_always_returns_one() {
        assert_eq!(no_penalty(100, 500), 1.0);
    }

    #[test]
    fn linear_penalty_scales_down() {
        let pen = linear(0.01);
        assert!((pen(50, 0) - 0.5).abs() < 1e-12);
        assert!((pen(0, 0) - 1.0).abs() < 1e-12);
        assert!((pen(200, 0) - 0.0).abs() < 1e-12); // clamped
    }

    #[test]
    fn threshold_penalty_no_effect_below() {
        let pen = threshold(20, 0.05);
        assert!((pen(10, 0) - 1.0).abs() < 1e-12);
        assert!((pen(20, 0) - 1.0).abs() < 1e-12);
        assert!((pen(30, 0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn seasonal_oscillates_between_extremes() {
        let pen = seasonal(100, 0.0, 0.02);
        // gen 0 = winter (max penalty)
        let winter = pen(50, 0);
        // gen 50 = summer (min penalty = 0.0 rate => factor 1.0)
        let summer = pen(50, 50);
        assert!(winter < summer, "winter={winter}, summer={summer}");
        assert!((summer - 1.0).abs() < 1e-12);
        assert!((winter - 0.0).abs() < 1e-12);
    }

    #[test]
    fn exponential_decays_smoothly() {
        let pen = exponential(0.1);
        let small = pen(5, 0);
        let large = pen(50, 0);
        assert!(small > large);
        assert!((pen(0, 0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn step_is_binary() {
        let pen = step(20, 0.5);
        assert!((pen(15, 0) - 1.0).abs() < 1e-12);
        assert!((pen(25, 0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn apply_multiplies_correctly() {
        let pen = threshold(10, 0.1);
        let result = apply(0.8, 15, 0, &pen);
        // excess = 5, factor = 1.0 - 0.1*5 = 0.5, result = 0.8 * 0.5 = 0.4
        assert!((result - 0.4).abs() < 1e-12);
    }
}
