use super::FrameIndex;

pub struct FrameScheduler {
    pub total_frames: u32,
    pub keyframe_interval: u32,
    pub interpolation_steps: u32,
}

impl FrameScheduler {
    pub fn new(total_frames: u32, keyframe_interval: u32, interpolation_steps: u32) -> Self {
        Self {
            total_frames,
            keyframe_interval,
            interpolation_steps,
        }
    }

    pub fn schedule_keyframes(&self) -> Vec<FrameIndex> {
        let mut keyframes = Vec::new();
        let mut current_frame = 0;
        
        while current_frame < self.total_frames {
            keyframes.push(current_frame);
            current_frame += self.keyframe_interval;
        }
        
        // Ensure the last frame is a keyframe if interpolation is needed up to the end
        if let Some(&last) = keyframes.last() {
            if last != self.total_frames - 1 {
                keyframes.push(self.total_frames - 1);
            }
        }
        
        keyframes
    }
}
