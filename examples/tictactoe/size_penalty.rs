/// Description of a network's size, passed to penalty functions.
#[derive(Debug, Copy, Clone)]
pub struct NetworkSize {
    pub nodes: usize,
    pub connections: usize,
}

/// A penalty applied to fitness based on network size.
/// All penalty functions return a value in [0.0, 1.0] that is multiplied
/// with the raw fitness to produce the final fitness.
///
/// A return value of 1.0 means no penalty; 0.0 means total penalty.

/// No penalty — baseline.
pub fn no_penalty(_size: NetworkSize, _generation: usize) -> f64 {
    1.0
}

/// Constant linear penalty on node count: fitness is scaled by `1 - rate * nodes`.
/// Clamped to [0.0, 1.0].
pub fn linear_nodes(rate: f64) -> impl Fn(NetworkSize, usize) -> f64 {
    move |size, _generation| (1.0 - rate * size.nodes as f64).clamp(0.0, 1.0)
}

/// Constant linear penalty on connection count.
pub fn linear_connections(rate: f64) -> impl Fn(NetworkSize, usize) -> f64 {
    move |size, _generation| (1.0 - rate * size.connections as f64).clamp(0.0, 1.0)
}

/// Threshold penalty on node count: no penalty below `threshold` nodes,
/// then linear penalty at `rate` per excess node.
pub fn threshold_nodes(threshold: usize, rate: f64) -> impl Fn(NetworkSize, usize) -> f64 {
    move |size, _generation| {
        let excess = size.nodes.saturating_sub(threshold) as f64;
        (1.0 - rate * excess).clamp(0.0, 1.0)
    }
}

/// Threshold penalty on connection count: no penalty below `threshold`,
/// then linear penalty at `rate` per excess connection.
pub fn threshold_connections(threshold: usize, rate: f64) -> impl Fn(NetworkSize, usize) -> f64 {
    move |size, _generation| {
        let excess = size.connections.saturating_sub(threshold) as f64;
        (1.0 - rate * excess).clamp(0.0, 1.0)
    }
}

/// Seasonal penalty simulating winter/summer cycles.
///
/// `metric_fn` extracts the relevant size metric (nodes, connections, or a combination).
///
/// In "summer" (cycle trough), the penalty is mild — networks can grow freely.
/// In "winter" (cycle peak), the penalty is harsh — bloated networks are punished.
///
/// The penalty oscillates sinusoidally between `min_rate` and `max_rate`,
/// with a full cycle every `period` generations.
///
/// Note: cos peaks at gen=0, so generation 0 starts in "winter" (max penalty).
fn seasonal_on<M>(
    metric_fn: M,
    period: usize,
    min_rate: f64,
    max_rate: f64,
) -> impl Fn(NetworkSize, usize) -> f64
where
    M: Fn(NetworkSize) -> f64,
{
    move |size, generation| {
        let phase = 2.0 * std::f64::consts::PI * (generation as f64) / (period as f64);
        let seasonal_factor = (1.0 + phase.cos()) / 2.0;
        let effective_rate = min_rate + (max_rate - min_rate) * seasonal_factor;
        (1.0 - effective_rate * metric_fn(size)).clamp(0.0, 1.0)
    }
}

/// Seasonal penalty on node count.
pub fn seasonal_nodes(
    period: usize,
    min_rate: f64,
    max_rate: f64,
) -> impl Fn(NetworkSize, usize) -> f64 {
    seasonal_on(|s| s.nodes as f64, period, min_rate, max_rate)
}

/// Seasonal penalty on connection count.
pub fn seasonal_connections(
    period: usize,
    min_rate: f64,
    max_rate: f64,
) -> impl Fn(NetworkSize, usize) -> f64 {
    seasonal_on(|s| s.connections as f64, period, min_rate, max_rate)
}

/// Seasonal penalty with a threshold: no penalty below `threshold`,
/// then seasonal scaling on the excess.
fn seasonal_with_threshold_on<M>(
    metric_fn: M,
    threshold: usize,
    period: usize,
    min_rate: f64,
    max_rate: f64,
) -> impl Fn(NetworkSize, usize) -> f64
where
    M: Fn(NetworkSize) -> usize + Copy,
{
    move |size, generation| {
        let metric = metric_fn(size);
        if metric <= threshold {
            1.0
        } else {
            // Build a synthetic NetworkSize with only the excess so seasonal_on
            // sees the excess count through the same metric_fn lens.
            // Instead, compute directly to avoid coupling.
            let excess = metric - threshold;
            let phase = 2.0 * std::f64::consts::PI * (generation as f64) / (period as f64);
            let seasonal_factor = (1.0 + phase.cos()) / 2.0;
            let effective_rate = min_rate + (max_rate - min_rate) * seasonal_factor;
            (1.0 - effective_rate * excess as f64).clamp(0.5, 1.0)
        }
    }
}

/// Seasonal penalty with threshold on node count.
pub fn seasonal_nodes_with_threshold(
    threshold: usize,
    period: usize,
    min_rate: f64,
    max_rate: f64,
) -> impl Fn(NetworkSize, usize) -> f64 {
    seasonal_with_threshold_on(|s| s.nodes, threshold, period, min_rate, max_rate)
}

/// Seasonal penalty with threshold on connection count.
pub fn seasonal_connections_with_threshold(
    threshold: usize,
    period: usize,
    min_rate: f64,
    max_rate: f64,
) -> impl Fn(NetworkSize, usize) -> f64 {
    seasonal_with_threshold_on(|s| s.connections, threshold, period, min_rate, max_rate)
}

/// Exponential decay penalty on a chosen metric.
fn exponential_on<M>(metric_fn: M, rate: f64) -> impl Fn(NetworkSize, usize) -> f64
where
    M: Fn(NetworkSize) -> f64,
{
    move |size, _generation| (-rate * metric_fn(size)).exp()
}

/// Exponential decay on node count.
pub fn exponential_nodes(rate: f64) -> impl Fn(NetworkSize, usize) -> f64 {
    exponential_on(|s| s.nodes as f64, rate)
}

/// Exponential decay on connection count.
pub fn exponential_connections(rate: f64) -> impl Fn(NetworkSize, usize) -> f64 {
    exponential_on(|s| s.connections as f64, rate)
}

/// Exponential decay with threshold on a chosen metric.
fn exponential_with_threshold_on<M>(
    metric_fn: M,
    threshold: usize,
    rate: f64,
) -> impl Fn(NetworkSize, usize) -> f64
where
    M: Fn(NetworkSize) -> usize,
{
    move |size, _generation| {
        let excess = metric_fn(size).saturating_sub(threshold) as f64;
        (-rate * excess).exp()
    }
}

/// Exponential decay with threshold on node count.
pub fn exponential_nodes_with_threshold(
    threshold: usize,
    rate: f64,
) -> impl Fn(NetworkSize, usize) -> f64 {
    exponential_with_threshold_on(|s| s.nodes, threshold, rate)
}

/// Exponential decay with threshold on connection count.
pub fn exponential_connections_with_threshold(
    threshold: usize,
    rate: f64,
) -> impl Fn(NetworkSize, usize) -> f64 {
    exponential_with_threshold_on(|s| s.connections, threshold, rate)
}

/// Step function: full fitness below `threshold`, flat multiplier above.
fn step_on<M>(
    metric_fn: M,
    threshold: usize,
    penalty_multiplier: f64,
) -> impl Fn(NetworkSize, usize) -> f64
where
    M: Fn(NetworkSize) -> usize,
{
    move |size, _generation| {
        if metric_fn(size) <= threshold {
            1.0
        } else {
            penalty_multiplier.clamp(0.0, 1.0)
        }
    }
}

/// Step function on node count.
pub fn step_nodes(threshold: usize, penalty_multiplier: f64) -> impl Fn(NetworkSize, usize) -> f64 {
    step_on(|s| s.nodes, threshold, penalty_multiplier)
}

/// Step function on connection count.
pub fn step_connections(
    threshold: usize,
    penalty_multiplier: f64,
) -> impl Fn(NetworkSize, usize) -> f64 {
    step_on(|s| s.connections, threshold, penalty_multiplier)
}

/// Compose two penalty functions by multiplying their factors.
/// Useful for penalising both nodes and connections independently.
///
/// # Example
/// ```ignore
/// let pen = compose(
///     threshold_nodes(20, 0.02),
///     threshold_connections(30, 0.01),
/// );
/// ```
pub fn compose<A, B>(a: A, b: B) -> impl Fn(NetworkSize, usize) -> f64
where
    A: Fn(NetworkSize, usize) -> f64,
    B: Fn(NetworkSize, usize) -> f64,
{
    move |size, generation| a(size, generation) * b(size, generation)
}

/// Apply a size penalty to a raw fitness value.
#[inline]
pub fn apply<F: Fn(NetworkSize, usize) -> f64>(
    raw_fitness: f64,
    size: NetworkSize,
    generation: usize,
    penalty_fn: &F,
) -> f64 {
    raw_fitness * penalty_fn(size, generation)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn size(nodes: usize, connections: usize) -> NetworkSize {
        NetworkSize { nodes, connections }
    }

    #[test]
    fn no_penalty_always_returns_one() {
        assert_eq!(no_penalty(size(100, 200), 500), 1.0);
    }

    #[test]
    fn linear_nodes_scales_down() {
        let pen = linear_nodes(0.01);
        assert!((pen(size(50, 0), 0) - 0.5).abs() < 1e-12);
        assert!((pen(size(0, 0), 0) - 1.0).abs() < 1e-12);
        assert!((pen(size(200, 0), 0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn linear_connections_scales_down() {
        let pen = linear_connections(0.01);
        assert!((pen(size(0, 50), 0) - 0.5).abs() < 1e-12);
        assert!((pen(size(0, 0), 0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn threshold_nodes_no_effect_below() {
        let pen = threshold_nodes(20, 0.05);
        assert!((pen(size(10, 100), 0) - 1.0).abs() < 1e-12);
        assert!((pen(size(20, 100), 0) - 1.0).abs() < 1e-12);
        assert!((pen(size(30, 100), 0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn threshold_connections_no_effect_below() {
        let pen = threshold_connections(20, 0.05);
        assert!((pen(size(100, 10), 0) - 1.0).abs() < 1e-12);
        assert!((pen(size(100, 20), 0) - 1.0).abs() < 1e-12);
        assert!((pen(size(100, 30), 0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn seasonal_nodes_oscillates() {
        let pen = seasonal_nodes(100, 0.0, 0.02);
        let winter = pen(size(50, 0), 0);
        let summer = pen(size(50, 0), 50);
        assert!(winter < summer, "winter={winter}, summer={summer}");
        assert!((summer - 1.0).abs() < 1e-12);
        assert!((winter - 0.0).abs() < 1e-12);
    }

    #[test]
    fn seasonal_connections_oscillates() {
        let pen = seasonal_connections(100, 0.0, 0.02);
        let winter = pen(size(0, 50), 0);
        let summer = pen(size(0, 50), 50);
        assert!(winter < summer);
        assert!((summer - 1.0).abs() < 1e-12);
    }

    #[test]
    fn exponential_nodes_decays() {
        let pen = exponential_nodes(0.1);
        assert!(pen(size(5, 0), 0) > pen(size(50, 0), 0));
        assert!((pen(size(0, 0), 0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn step_nodes_is_binary() {
        let pen = step_nodes(20, 0.5);
        assert!((pen(size(15, 100), 0) - 1.0).abs() < 1e-12);
        assert!((pen(size(25, 100), 0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn compose_multiplies_factors() {
        let pen = compose(threshold_nodes(10, 0.1), threshold_connections(20, 0.1));
        // nodes: excess 5 → 0.5, connections: excess 5 → 0.5, combined → 0.25
        let result = pen(size(15, 25), 0);
        assert!((result - 0.25).abs() < 1e-12);
    }

    #[test]
    fn compose_one_side_no_penalty() {
        let pen = compose(threshold_nodes(10, 0.1), no_penalty);
        let result = pen(size(15, 100), 0);
        assert!((result - 0.5).abs() < 1e-12);
    }

    #[test]
    fn apply_multiplies_correctly() {
        let pen = threshold_nodes(10, 0.1);
        let result = apply(0.8, size(15, 0), 0, &pen);
        assert!((result - 0.4).abs() < 1e-12);
    }
}
