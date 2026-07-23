use prism_ecs_core::identity::GraphProvider;
use prism_ecs_core::World;

pub struct OnnxGraphProvider {
    _data: Vec<u8>,
}

impl OnnxGraphProvider {
    pub fn new(data: &[u8]) -> Result<Self, String> {
        Ok(Self {
            _data: data.to_vec(),
        })
    }
}

impl GraphProvider for OnnxGraphProvider {
    fn import_graph(&self, _world: &mut World) -> Result<(), String> {
        Err("ONNX graph import not yet implemented".into())
    }
}
