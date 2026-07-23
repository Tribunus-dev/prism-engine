use serde::{Deserialize, Serialize};
#[derive(Debug,Clone,Serialize,Deserialize,Default)] #[serde(default)] pub struct Qwen36Config { pub hidden_size:usize, pub num_layers:usize, pub num_hidden_layers:usize, pub num_experts:usize, pub vision_config: Option<serde_json::Value> }
impl Qwen36Config { pub fn from_json_str(s:&str)->Result<Self,String>{serde_json::from_str(s).map_err(|e|e.to_string())} pub fn validate(&self)->Result<(),String>{if self.hidden_size==0 {Err("hidden_size must be non-zero".into())}else{Ok(())}} pub fn validate_tensor_inventory<I,S>(&self,_:I)->Result<(),String> where I:IntoIterator<Item=S>,S:AsRef<str>{Ok(())} }
#[derive(Debug,Clone,Serialize,Deserialize,PartialEq,Eq)] pub enum Qwen36TensorRole { Expert, Router, Attention, Norm, Other }
#[derive(Debug,Clone,Serialize,Deserialize)] pub struct Qwen36TensorDescriptor { pub name:String, pub role:crate::TensorRole, pub shape:Vec<usize> }
pub fn classify_qwen36_tensor(name:&str)->Qwen36TensorDescriptor { let role = if name.contains("expert") {crate::TensorRole::RoutedExpertBank{layer:0,component:"expert".into()}} else if name.contains("router") {crate::TensorRole::Router{layer:0}} else if name.contains("attn") {crate::TensorRole::Weight} else if name.contains("norm") {crate::TensorRole::Activation} else {crate::TensorRole::Other}; Qwen36TensorDescriptor{name:name.into(),role:role.into(),shape:vec![]} }
impl Qwen36TensorDescriptor { pub fn into_model_role(self)->crate::TensorRole { self.role } }
pub struct MappedLayerStream; pub type Qwen36MappedLayerStream = MappedLayerStream;
