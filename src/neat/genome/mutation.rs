use super::innovation::InnovationTracker;
use super::types::{
    ConnectionGene, ConnectionKey, Genome, GenomeError, Innovation, NodeId, NodeKind,
};

#[derive(Debug, Clone)]
pub enum Mutation {
    AddConnection {
        in_node: NodeId,
        out_node: NodeId,
        weight: f64,
    },
    AddNode {
        split_innovation: Innovation,
    },
    PerturbWeight {
        innovation: Innovation,
        delta: f64,
    },
    DisableConnection {
        innovation: Innovation,
    },
}

impl Genome {
    pub fn with_added_connection(
        &self,
        tracker: &mut InnovationTracker,
        in_node: NodeId,
        out_node: NodeId,
        weight: f64,
    ) -> Result<Self, GenomeError> {
        self.validate_connection_endpoints(in_node, out_node)?;

        let key = ConnectionKey { in_node, out_node };
        if self.connection_to_innovation.contains_key(&key) {
            return Err(GenomeError::DuplicateConnection(key));
        }

        let innovation = tracker.next_connection_innovation();
        let gene = ConnectionGene {
            key: key.clone(),
            innovation,
            weight,
            enabled: true,
        };

        let mut next = self.clone();
        next.insert_connection_gene(gene);
        Ok(next)
    }

    pub fn with_added_node(
        &self,
        tracker: &mut InnovationTracker,
        split_innovation: Innovation,
    ) -> Result<Self, GenomeError> {
        let original = self
            .connections_by_innovation
            .get(&split_innovation)
            .ok_or(GenomeError::UnknownInnovation(split_innovation))?;

        if !original.enabled {
            return Err(GenomeError::ConnectionAlreadyDisabled(split_innovation));
        }

        let mut next = self.clone();

        if let Some(g) = next.connections_by_innovation.get_mut(&split_innovation) {
            g.enabled = false;
        }

        let new_node = tracker.next_hidden_node_id();
        next.nodes.insert(new_node, NodeKind::Hidden);

        let left_key = ConnectionKey {
            in_node: original.key.in_node,
            out_node: new_node,
        };
        let right_key = ConnectionKey {
            in_node: new_node,
            out_node: original.key.out_node,
        };

        let left_gene = ConnectionGene {
            key: left_key.clone(),
            innovation: tracker.next_connection_innovation(),
            weight: 1.0,
            enabled: true,
        };

        let right_gene = ConnectionGene {
            key: right_key.clone(),
            innovation: tracker.next_connection_innovation(),
            weight: original.weight,
            enabled: true,
        };

        next.insert_connection_gene(left_gene);
        next.insert_connection_gene(right_gene);

        Ok(next)
    }

    pub fn with_perturbed_weight(
        &self,
        innovation: Innovation,
        delta: f64,
    ) -> Result<Self, GenomeError> {
        let mut next = self.clone();
        let gene = next
            .connections_by_innovation
            .get_mut(&innovation)
            .ok_or(GenomeError::UnknownInnovation(innovation))?;

        gene.weight += delta;
        Ok(next)
    }

    pub fn with_disabled_connection(&self, innovation: Innovation) -> Result<Self, GenomeError> {
        let mut next = self.clone();
        let gene = next
            .connections_by_innovation
            .get_mut(&innovation)
            .ok_or(GenomeError::UnknownInnovation(innovation))?;

        if !gene.enabled {
            return Err(GenomeError::ConnectionAlreadyDisabled(innovation));
        }

        gene.enabled = false;
        Ok(next)
    }

    pub fn apply_mutation(
        &self,
        tracker: &mut InnovationTracker,
        mutation: &Mutation,
    ) -> Result<Self, GenomeError> {
        match mutation {
            Mutation::AddConnection {
                in_node,
                out_node,
                weight,
            } => self.with_added_connection(tracker, *in_node, *out_node, *weight),
            Mutation::AddNode { split_innovation } => {
                self.with_added_node(tracker, *split_innovation)
            }
            Mutation::PerturbWeight { innovation, delta } => {
                self.with_perturbed_weight(*innovation, *delta)
            }
            Mutation::DisableConnection { innovation } => {
                self.with_disabled_connection(*innovation)
            }
        }
    }

    pub fn apply_mutations(
        &self,
        tracker: &mut InnovationTracker,
        mutations: &[Mutation],
    ) -> Result<Self, GenomeError> {
        mutations
            .iter()
            .try_fold(self.clone(), |acc, m| acc.apply_mutation(tracker, m))
    }
}
