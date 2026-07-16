pub trait AudioCodec: Send {
    fn encode(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<u32>, String>;
    fn decode(&self, tokens: &[u32]) -> Result<Vec<f32>, String>;
}

pub struct EnCodecDecoder {/* Meta's EnCodec, 24/32/48 kHz */}
pub struct DacDecoder {/* Descript Audio Codec */}
pub struct MdctCodec {/* Stable Audio's MDCT-based codec */}

impl AudioCodec for EnCodecDecoder {
    fn encode(&self, _audio: &[f32], _sample_rate: u32) -> Result<Vec<u32>, String> {
        Ok(vec![])
    }
    fn decode(&self, _tokens: &[u32]) -> Result<Vec<f32>, String> {
        Ok(vec![])
    }
}

impl AudioCodec for DacDecoder {
    fn encode(&self, _audio: &[f32], _sample_rate: u32) -> Result<Vec<u32>, String> {
        Ok(vec![])
    }
    fn decode(&self, _tokens: &[u32]) -> Result<Vec<f32>, String> {
        Ok(vec![])
    }
}

impl AudioCodec for MdctCodec {
    fn encode(&self, _audio: &[f32], _sample_rate: u32) -> Result<Vec<u32>, String> {
        Ok(vec![])
    }
    fn decode(&self, _tokens: &[u32]) -> Result<Vec<f32>, String> {
        Ok(vec![])
    }
}
