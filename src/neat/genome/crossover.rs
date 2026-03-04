use super::types::{ConnectionGene, Genome, GenomeError, NodeId, NodeKind, ParentFitness};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Copy, Clone)]
pub enum Alignment<'a> {
    Both(&'a ConnectionGene, &'a ConnectionGene),
    Left(&'a ConnectionGene),
    Right(&'a ConnectionGene),
}

pub fn for_each_aligned_by_innovation<'a, F>(left: &'a Genome, right: &'a Genome, mut f: F)
where
    F: FnMut(Alignment<'a>),
{
    let mut l_it = left.connections_by_innovation.iter().peekable();
    let mut r_it = right.connections_by_innovation.iter().peekable();

    loop {
        match (l_it.peek(), r_it.peek()) {
            (Some((l_innov, l_gene)), Some((r_innov, r_gene))) => {
                if l_innov == r_innov {
                    f(Alignment::Both(l_gene, r_gene));
                    l_it.next();
                    r_it.next();
                } else if l_innov < r_innov {
                    f(Alignment::Left(l_gene));
                    l_it.next();
                } else {
                    f(Alignment::Right(r_gene));
                    r_it.next();
                }
            }
            (Some((_, l_gene)), None) => {
                f(Alignment::Left(l_gene));
                l_it.next();
            }
            (None, Some((_, r_gene))) => {
                f(Alignment::Right(r_gene));
                r_it.next();
            }
            (None, None) => break,
        }
    }
}

pub(crate) trait CrossoverPolicy {
    fn choose_left_matching(&mut self, left: &ConnectionGene, right: &ConnectionGene) -> bool;
    fn choose_left_when_equal_for_unmatched(&mut self) -> bool;
    fn enable_if_either_parent_disabled(&mut self) -> bool;
}

impl Genome {
    fn validate_crossover_compatibility(left: &Genome, right: &Genome) -> Result<(), GenomeError> {
        if left.n_inputs != right.n_inputs || left.n_outputs != right.n_outputs {
            return Err(GenomeError::MismatchedIo {
                left_inputs: left.n_inputs,
                left_outputs: left.n_outputs,
                right_inputs: right.n_inputs,
                right_outputs: right.n_outputs,
            });
        }
        Ok(())
    }

    fn merged_nodes_for_crossover(
        left: &Genome,
        right: &Genome,
    ) -> Result<BTreeMap<NodeId, NodeKind>, GenomeError> {
        let mut merged = left.nodes.clone();

        for (node, kind_right) in &right.nodes {
            match merged.get(node) {
                Some(kind_left) if kind_left != kind_right => {
                    return Err(GenomeError::MismatchedNodeKind {
                        node: *node,
                        left: *kind_left,
                        right: *kind_right,
                    });
                }
                Some(_) => {}
                None => {
                    merged.insert(*node, *kind_right);
                }
            }
        }

        Ok(merged)
    }

    fn inherit_unmatched<P: CrossoverPolicy>(
        fitness: ParentFitness,
        is_left_gene: bool,
        policy: &mut P,
    ) -> bool {
        match fitness {
            ParentFitness::Left => is_left_gene,
            ParentFitness::Right => !is_left_gene,
            ParentFitness::Equal => {
                let choose_left = policy.choose_left_when_equal_for_unmatched();
                (is_left_gene && choose_left) || (!is_left_gene && !choose_left)
            }
        }
    }

    pub(crate) fn crossover_with_policy<P: CrossoverPolicy>(
        left: &Genome,
        right: &Genome,
        fitness: ParentFitness,
        policy: &mut P,
    ) -> Result<Self, GenomeError> {
        Self::validate_crossover_compatibility(left, right)?;
        let nodes = Self::merged_nodes_for_crossover(left, right)?;

        let mut child = Self {
            n_inputs: left.n_inputs,
            n_outputs: left.n_outputs,
            nodes,
            connections_by_innovation: BTreeMap::new(),
            connection_to_innovation: HashMap::new(),
        };

        for_each_aligned_by_innovation(left, right, |aligned| match aligned {
            Alignment::Both(l, r) => {
                let mut inherited = if policy.choose_left_matching(l, r) {
                    l.clone()
                } else {
                    r.clone()
                };

                if l.enabled ^ r.enabled {
                    inherited.enabled = policy.enable_if_either_parent_disabled();
                }

                child.insert_connection_gene(inherited);
            }
            Alignment::Left(l) => {
                if Self::inherit_unmatched(fitness, true, policy) {
                    child.insert_connection_gene(l.clone());
                }
            }
            Alignment::Right(r) => {
                if Self::inherit_unmatched(fitness, false, policy) {
                    child.insert_connection_gene(r.clone());
                }
            }
        });

        Ok(child)
    }
}
