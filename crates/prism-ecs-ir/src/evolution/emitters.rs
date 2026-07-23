use super::{CandidateGenome, EvolutionaryMemory, EvolutionContextKey}; use rand::Rng; use serde::{Serialize,Deserialize};
#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash,Serialize,Deserialize)] pub enum EmitterKind{Local,Memory,Random,Semantic,Failure}
#[derive(Debug,Clone,Copy,Default,Serialize,Deserialize)] pub struct EmitterStats{pub attempts:u64,pub successes:u64,pub reward_sum:f64}
#[derive(Debug,Clone,Serialize,Deserialize)] pub struct EmitterPolicy{pub stats:std::collections::HashMap<EmitterKind,EmitterStats>,pub exploration:f64}
impl Default for EmitterPolicy{fn default()->Self{let mut s=std::collections::HashMap::new();for k in [EmitterKind::Local,EmitterKind::Memory,EmitterKind::Random,EmitterKind::Semantic,EmitterKind::Failure]{s.insert(k,EmitterStats::default());}Self{stats:s,exploration:0.25}}}
impl EmitterPolicy{pub fn choose(&self,_:&mut impl Rng)->EmitterKind{EmitterKind::Local} pub fn record(&mut self,k:EmitterKind,r:f64){let s=self.stats.entry(k).or_default();s.attempts+=1;if r>0.0{s.successes+=1;}s.reward_sum+=r}}
pub trait Emitter{fn emit(&self,seed:&CandidateGenome,_:&[u8],_memory:&EvolutionaryMemory,_:&mut impl Rng)->Vec<CandidateGenome>;}
pub struct LocalEmitter; pub struct MemoryEmitter; pub struct RandomEmitter; pub struct StrategyEmitter(pub EmitterKind);
impl Emitter for LocalEmitter{fn emit(&self,s:&CandidateGenome,_:&[u8],_:&EvolutionaryMemory,_:&mut impl Rng)->Vec<CandidateGenome>{vec![s.clone()]}}
impl Emitter for MemoryEmitter{fn emit(&self,s:&CandidateGenome,_:&[u8],_:&EvolutionaryMemory,_:&mut impl Rng)->Vec<CandidateGenome>{vec![s.clone()]}}
impl Emitter for RandomEmitter{fn emit(&self,s:&CandidateGenome,_:&[u8],_:&EvolutionaryMemory,_:&mut impl Rng)->Vec<CandidateGenome>{vec![s.clone()]}}
impl Emitter for StrategyEmitter{fn emit(&self,s:&CandidateGenome,_:&[u8],_:&EvolutionaryMemory,_:&mut impl Rng)->Vec<CandidateGenome>{vec![s.clone()]}}
#[derive(Debug,Clone,Serialize,Deserialize)] pub struct EmitterContext(pub EvolutionContextKey);
