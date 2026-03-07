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
        let mut next = self.clone();
        next.add_connection_in_place(tracker, in_node, out_node, weight)?;
        Ok(next)
    }

    /// Mutate in place — avoids a clone when you already own the genome.
    pub fn add_connection_in_place(
        &mut self,
        tracker: &mut InnovationTracker,
        in_node: NodeId,
        out_node: NodeId,
        weight: f64,
    ) -> Result<(), GenomeError> {
        self.validate_connection_endpoints(in_node, out_node)?;

        let key = ConnectionKey { in_node, out_node };

        if let Some(&innovation) = self.connection_to_innovation.get(&key) {
            let gene = self
                .connections_by_innovation
                .get_mut(&innovation)
                .ok_or(GenomeError::UnknownInnovation(innovation))?;

            if gene.enabled {
                return Err(GenomeError::DuplicateConnection(key));
            }

            gene.enabled = true;
            gene.weight = weight;
            return Ok(());
        }

        let innovation = tracker.next_connection_innovation();
        let gene = ConnectionGene {
            key,
            innovation,
            weight,
            enabled: true,
        };

        self.insert_connection_gene(gene);
        Ok(())
    }

    pub fn with_added_node(
        &self,
        tracker: &mut InnovationTracker,
        split_innovation: Innovation,
    ) -> Result<Self, GenomeError> {
        let mut next = self.clone();
        next.add_node_in_place(tracker, split_innovation)?;
        Ok(next)
    }

    /// Mutate in place — avoids a clone when you already own the genome.
    pub fn add_node_in_place(
        &mut self,
        tracker: &mut InnovationTracker,
        split_innovation: Innovation,
    ) -> Result<(), GenomeError> {
        let original = self
            .connections_by_innovation
            .get(&split_innovation)
            .ok_or(GenomeError::UnknownInnovation(split_innovation))?;

        if !original.enabled {
            return Err(GenomeError::ConnectionAlreadyDisabled(split_innovation));
        }

        let in_node = original.key.in_node;
        let out_node = original.key.out_node;
        let old_weight = original.weight;

        self.connections_by_innovation
            .get_mut(&split_innovation)
            .unwrap()
            .enabled = false;

        let new_node = tracker.next_hidden_node_id();
        self.nodes.insert(new_node, NodeKind::Hidden);

        let left_gene = ConnectionGene {
            key: ConnectionKey {
                in_node,
                out_node: new_node,
            },
            innovation: tracker.next_connection_innovation(),
            weight: 1.0,
            enabled: true,
        };

        let right_gene = ConnectionGene {
            key: ConnectionKey {
                in_node: new_node,
                out_node,
            },
            innovation: tracker.next_connection_innovation(),
            weight: old_weight,
            enabled: true,
        };

        self.insert_connection_gene(left_gene);
        self.insert_connection_gene(right_gene);

        Ok(())
    }

    pub fn with_perturbed_weight(
        &self,
        innovation: Innovation,
        delta: f64,
    ) -> Result<Self, GenomeError> {
        let mut next = self.clone();
        next.perturb_weight_in_place(innovation, delta)?;
        Ok(next)
    }

    pub fn perturb_weight_in_place(
        &mut self,
        innovation: Innovation,
        delta: f64,
    ) -> Result<(), GenomeError> {
        let gene = self
            .connections_by_innovation
            .get_mut(&innovation)
            .ok_or(GenomeError::UnknownInnovation(innovation))?;
        gene.weight += delta;
        Ok(())
    }

    pub fn with_disabled_connection(&self, innovation: Innovation) -> Result<Self, GenomeError> {
        let mut next = self.clone();
        next.disable_connection_in_place(innovation)?;
        Ok(next)
    }

    pub fn disable_connection_in_place(
        &mut self,
        innovation: Innovation,
    ) -> Result<(), GenomeError> {
        let gene = self
            .connections_by_innovation
            .get_mut(&innovation)
            .ok_or(GenomeError::UnknownInnovation(innovation))?;

        if !gene.enabled {
            return Err(GenomeError::ConnectionAlreadyDisabled(innovation));
        }

        gene.enabled = false;
        Ok(())
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

    /// Apply a sequence of mutations, cloning once and then mutating in place.
    pub fn apply_mutations(
        &self,
        tracker: &mut InnovationTracker,
        mutations: &[Mutation],
    ) -> Result<Self, GenomeError> {
        let mut genome = self.clone();
        for m in mutations {
            genome.apply_mutation_in_place(tracker, m)?;
        }
        Ok(genome)
    }

    /// Apply a single mutation in place — no clone.
    pub fn apply_mutation_in_place(
        &mut self,
        tracker: &mut InnovationTracker,
        mutation: &Mutation,
    ) -> Result<(), GenomeError> {
        match mutation {
            Mutation::AddConnection {
                in_node,
                out_node,
                weight,
            } => self.add_connection_in_place(tracker, *in_node, *out_node, *weight),
            Mutation::AddNode { split_innovation } => {
                self.add_node_in_place(tracker, *split_innovation)
            }
            Mutation::PerturbWeight { innovation, delta } => {
                self.perturb_weight_in_place(*innovation, *delta)
            }
            Mutation::DisableConnection { innovation } => {
                self.disable_connection_in_place(*innovation)
            }
        }
    }

    /// Like `apply_mutation` but consumes self to avoid an extra clone.
    pub fn apply_mutation_owned(
        mut self,
        tracker: &mut InnovationTracker,
        mutation: &Mutation,
    ) -> Result<Self, GenomeError> {
        self.apply_mutation_in_place(tracker, mutation)?;
        Ok(self)
    }
}
