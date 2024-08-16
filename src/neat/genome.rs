use std::ops::Index;
use super::phenome::NodeIndex;
pub struct InnovationNumber(pub usize);
impl InnovationNumber {
    fn inc(mut self) -> InnovationNumber {
        self.0 += 1;
        self
    }
}

#[derive(PartialEq, PartialOrd, Clone, Copy)]
pub struct GeneIndex(pub usize);

pub struct Gene {
    pub in_node_id: NodeIndex,
    pub out_node_id: NodeIndex,
    pub weight: f64,
    pub innovation: InnovationNumber,
    pub enabled: bool
}

impl Gene {
    pub fn create(in_node_id: usize, out_node_id: usize, weight: f64, innovation: usize, enabled: bool) -> Gene {
        Gene {
            in_node_id: NodeIndex(in_node_id),
            out_node_id: NodeIndex(out_node_id),
            weight,
            innovation: InnovationNumber(innovation),
            enabled,
        }
    }
}

pub struct Genome(Vec<Gene>);
impl Genome {
    pub fn calculate_max_node_id(&self) -> usize {
        let res = 
            self.0.iter().fold(NodeIndex(0), |acc, conn| {
                if conn.in_node_id > acc {
                    conn.in_node_id
                } else if conn.out_node_id > acc {
                    conn.out_node_id
                } else {
                    acc
                }
            });
        res.0
    }

    pub fn iter(&self) -> std::slice::Iter<Gene> {
        self.0.iter()
    }

    pub fn create(data: Vec<Gene>) -> Genome {
        Genome(data)
    }

    pub fn push(&mut self, gene: Gene) {
        self.0.push(gene);
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn tarjan_scc(&self) -> Vec<Vec<NodeIndex>> {
        use petgraph::graph::DiGraph;
        let edges = self.iter().map(|gene| (gene.in_node_id.0, gene.out_node_id.0));
        let graph: petgraph::graph::DiGraph<(), (), usize> = DiGraph::from_edges(edges);


        let scc_order = petgraph::algo::tarjan_scc(&graph);
        scc_order.iter().rev().map(|scc| {
            scc.iter().map(|node_index| {
                NodeIndex(node_index.index())
            }).collect()
        }).collect()
    }
}

impl Index<GeneIndex> for Genome {
    type Output = Gene;
    fn index(&self, index: GeneIndex) -> &Self::Output {
        &self.0[index.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn genome_sample_1() -> Genome{
        Genome::create(vec![
            Gene::create(0, 3, 0.0, 0, true),
            Gene::create(1, 3, 0.0, 1, true),
            Gene::create(1, 4, 0.0, 2, true),
            Gene::create(2, 4, 0.0, 3, true),
            Gene::create(3, 4, 0.0, 4, true),
        ])
    } 
    
    #[test]
    fn test_genome_max_node_id() {
        let genome = genome_sample_1();
        assert_eq!(genome.calculate_max_node_id(), 5);
    }
}