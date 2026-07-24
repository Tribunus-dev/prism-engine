pub mod embedding;
pub mod multimodal;

pub mod media {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum MediaKind {
        Audio,
        Image,
        Video,
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum PixelFormat {
        Rgba8,
        Bgra8,
        Gray8,
        F32,
        #[serde(rename = "f32_pcm")]
        F32Pcm,
        #[serde(rename = "s16_pcm")]
        S16Pcm,
        Nv12,
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum MediaSessionMode {
        Realtime,
        Batch,
        Batched,
    }
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(tag = "kind", rename_all = "lowercase")]
    pub enum MediaSource {
        File { path: String },
        SystemCamera { device: Option<String> },
        ConnectedIphoneCamera { device: Option<String> },
        SystemMicrophone { device: Option<String> },
        ConnectedIphoneMicrophone { device: Option<String> },
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum MediaMemory {
        Cpu,
        Unified,
        Gpu,
    }
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MediaDescriptor {
        pub kind: MediaKind,
        pub format: PixelFormat,
        pub width: Option<u32>,
        pub height: Option<u32>,
        pub sample_rate: Option<u32>,
        #[serde(default)]
        pub batch_size: u32,
    }
    impl MediaDescriptor {
        pub fn rgba(width: u32, height: u32, batch_size: u32) -> Self {
            Self {
                kind: MediaKind::Image,
                format: PixelFormat::Rgba8,
                width: Some(width),
                height: Some(height),
                sample_rate: None,
                batch_size,
            }
        }
    }
    #[derive(Debug, Clone)]
    pub struct NativeVideoBuffer {
        pub rgba: Vec<u8>,
    }
    impl NativeVideoBuffer {
        pub fn copy_rgba(&self) -> Result<Vec<u8>, String> {
            Ok(self.rgba.clone())
        }
    }
    #[derive(Debug, Clone)]
    pub struct MediaPacket {
        pub descriptor: MediaDescriptor,
        pub payload: Vec<u8>,
        pub model_id: Option<String>,
        pub native_video: Option<NativeVideoBuffer>,
        pub source: MediaSource,
        pub memory: MediaMemory,
        pub timestamp_ns: u64,
        pub sequence: u64,
        pub payload_bytes: u64,
    }
    #[derive(Debug, Clone, Serialize)]
    pub struct RouteEvidence {
        pub route: String,
        pub memory: MediaMemory,
        pub accelerators: Vec<String>,
        pub zero_copy: bool,
    }
    pub fn resolve_ingress(
        source: &MediaSource,
        descriptor: &MediaDescriptor,
    ) -> Result<RouteEvidence, String> {
        let route = match (source, descriptor.kind) {
            (MediaSource::File { .. }, MediaKind::Image) => "image_file_decode",
            (MediaSource::File { .. }, MediaKind::Audio) => "audio_file_decode",
            (MediaSource::File { .. }, MediaKind::Video) => "video_toolbox_decode",
            (
                MediaSource::SystemCamera { .. } | MediaSource::ConnectedIphoneCamera { .. },
                MediaKind::Image,
            ) => "camera_capture",
            (
                MediaSource::SystemMicrophone { .. }
                | MediaSource::ConnectedIphoneMicrophone { .. },
                MediaKind::Audio,
            ) => "microphone_capture",
            (_, kind) => return Err(format!("source is incompatible with {:?} media", kind)),
        };
        Ok(RouteEvidence {
            route: route.into(),
            memory: MediaMemory::Cpu,
            accelerators: Vec::new(),
            zero_copy: false,
        })
    }
    pub fn resolve_egress(kind: MediaKind, format: PixelFormat) -> Result<RouteEvidence, String> {
        let route = match kind {
            MediaKind::Image => "image_materialize",
            MediaKind::Audio => "audio_materialize",
            MediaKind::Video => "video_toolbox_encode",
        };
        let _ = format;
        Ok(RouteEvidence {
            route: route.into(),
            memory: MediaMemory::Cpu,
            accelerators: Vec::new(),
            zero_copy: false,
        })
    }
}

pub mod io {
    use super::media::{
        MediaDescriptor, MediaKind, MediaMemory, MediaPacket, MediaSource, PixelFormat,
    };
    pub fn audio_packet_features(packet: &MediaPacket) -> Result<Vec<Vec<f32>>, String> {
        if packet.payload.len() % 4 != 0 {
            return Err("audio payload is not f32 aligned".into());
        }
        Ok(vec![packet
            .payload
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()])
    }
    pub struct ImportedImage {
        pub width: u32,
        pub height: u32,
        pub rgba: Vec<u8>,
    }
    impl ImportedImage {
        pub fn into_packet(
            self,
            source: MediaSource,
            batch_size: u32,
            model_id: Option<String>,
        ) -> MediaPacket {
            let bytes = self.rgba.len() as u64;
            MediaPacket {
                descriptor: MediaDescriptor {
                    kind: MediaKind::Image,
                    format: PixelFormat::Rgba8,
                    width: Some(self.width),
                    height: Some(self.height),
                    sample_rate: None,
                    batch_size,
                },
                payload: self.rgba,
                model_id,
                native_video: None,
                source,
                memory: MediaMemory::Cpu,
                timestamp_ns: 0,
                sequence: 0,
                payload_bytes: bytes,
            }
        }
    }
    pub fn import_image_rgba(path: &str) -> Result<ImportedImage, String> {
        let data = std::fs::read(path).map_err(|e| e.to_string())?;
        Ok(ImportedImage {
            width: 1,
            height: (data.len() as u32 / 4).max(1),
            rgba: data,
        })
    }
    pub fn import_video_frames(path: &str) -> Result<Vec<Vec<u8>>, String> {
        Ok(vec![std::fs::read(path).map_err(|e| e.to_string())?])
    }
    pub fn import_audio_packet(
        path: &str,
        source: MediaSource,
        batch_size: u32,
        model_id: Option<String>,
    ) -> Result<MediaPacket, String> {
        let payload = std::fs::read(path).map_err(|e| e.to_string())?;
        Ok(MediaPacket {
            descriptor: MediaDescriptor {
                kind: MediaKind::Audio,
                format: PixelFormat::F32Pcm,
                width: None,
                height: None,
                sample_rate: Some(24_000),
                batch_size,
            },
            payload_bytes: payload.len() as u64,
            payload,
            model_id,
            native_video: None,
            source,
            memory: MediaMemory::Cpu,
            timestamp_ns: 0,
            sequence: 0,
        })
    }
    pub fn export_packet(packet: &MediaPacket, path: &str) -> Result<(), String> {
        std::fs::write(path, &packet.payload).map_err(|e| e.to_string())
    }
    pub fn export_video_frames(
        frames: &[Vec<u8>],
        _width: u32,
        _height: u32,
        _fps: f32,
        path: &str,
        _codec: impl std::fmt::Debug,
    ) -> Result<(), String> {
        let bytes = frames.iter().flatten().copied().collect::<Vec<_>>();
        std::fs::write(path, bytes).map_err(|e| e.to_string())
    }
}

pub mod capture {
    use super::media::{MediaDescriptor, MediaPacket, MediaSessionMode, MediaSource};
    pub fn admit_live_source(source: &MediaSource) -> Result<(), String> {
        let _ = source;
        Ok(())
    }
    pub fn hardware_resize_rgba(
        rgba: &[u8],
        _width: u32,
        _height: u32,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, String> {
        let mut output = vec![0; (width * height * 4) as usize];
        for y in 0..height {
            for x in 0..width {
                let source = (((y as usize * _height as usize / height as usize)
                    * _width as usize)
                    + (x as usize * _width as usize / width as usize))
                    * 4;
                let target = ((y * width + x) * 4) as usize;
                output[target..target + 4].copy_from_slice(
                    rgba.get(source..source + 4)
                        .ok_or("RGBA buffer is too short")?,
                );
            }
        }
        Ok(output)
    }
    pub fn enumerate_apple_capture_devices() -> Result<Vec<String>, String> {
        Ok(Vec::new())
    }
    #[derive(Debug, Clone, Copy)]
    pub struct CapturePermissions {
        pub microphone: bool,
        pub camera: bool,
    }
    pub fn probe_apple_capture_permissions() -> Result<CapturePermissions, String> {
        Ok(CapturePermissions {
            microphone: false,
            camera: false,
        })
    }
    pub struct CaptureCoordinator;
    impl CaptureCoordinator {
        pub fn start_for_model(
            _model: &str,
            _source: MediaSource,
            _descriptor: MediaDescriptor,
            _mode: MediaSessionMode,
        ) -> Result<Self, String> {
            Ok(Self)
        }
        pub fn start_zero_copy_camera_for_model(
            _model: &str,
            _source: MediaSource,
            _descriptor: MediaDescriptor,
            _mode: MediaSessionMode,
        ) -> Result<Self, String> {
            Ok(Self)
        }
        pub fn poll(&mut self) -> Result<Option<MediaPacket>, String> {
            Ok(None)
        }
    }
}
