pub mod embedding;
pub mod multimodal;

pub mod media {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum MediaKind { Audio, Image, Video }
    #[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum PixelFormat { Rgba8, Bgra8, F32 }
    #[derive(Debug, Clone)] pub struct MediaDescriptor { pub kind: MediaKind, pub format: PixelFormat, pub width: Option<u32>, pub height: Option<u32>, pub sample_rate: Option<u32> }
    #[derive(Debug, Clone)] pub struct NativeVideoBuffer { pub rgba: Vec<u8> }
    impl NativeVideoBuffer { pub fn copy_rgba(&self) -> Result<Vec<u8>, String> { Ok(self.rgba.clone()) } }
    #[derive(Debug, Clone)] pub struct MediaPacket { pub descriptor: MediaDescriptor, pub payload: Vec<u8>, pub model_id: Option<String>, pub native_video: Option<NativeVideoBuffer> }
}

pub mod io {
    use super::media::MediaPacket;
    pub fn audio_packet_features(packet: &MediaPacket) -> Result<Vec<Vec<f32>>, String> {
        if packet.payload.len() % 4 != 0 { return Err("audio payload is not f32 aligned".into()); }
        Ok(vec![packet.payload.chunks_exact(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect()])
    }
}
