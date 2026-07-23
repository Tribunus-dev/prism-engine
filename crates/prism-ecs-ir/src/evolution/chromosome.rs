use super::foundation::CandidateGenome;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Chromosome { pub locus: String, pub allele: String }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenomeChromosomes {
    pub chromosomes: Vec<Chromosome>,
    pub representation: AxisDescriptor,
    pub packing: AxisDescriptor,
    pub schedule: AxisDescriptor,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AxisDescriptor(pub u8);
impl AxisDescriptor { pub fn descriptor(&self) -> u8 { self.0 } }

impl From<&CandidateGenome> for GenomeChromosomes {
    fn from(genome: &CandidateGenome) -> Self {
        Self { chromosomes: vec![
            Chromosome { locus: "representation".into(), allele: format!("{:?}", genome.representation) },
            Chromosome { locus: "packing".into(), allele: format!("{:?}", genome.packing) },
            Chromosome { locus: "fusion".into(), allele: format!("{:?}", genome.fusion) },
        ], representation: AxisDescriptor(0), packing: AxisDescriptor(0), schedule: AxisDescriptor(0) }
    }
}
