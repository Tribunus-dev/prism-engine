use crate::ecs::canonical::identity::HardwareProfileId;

/// Compatibility suite — target hardware profiles and release gates.
/// Plan Section 16: "Release gate passes on supported Apple Silicon profiles."
pub struct CompatibilitySuite {
    pub profile: HardwareProfileId,
}

impl CompatibilitySuite {
    pub fn for_profile(profile: HardwareProfileId) -> Self {
        Self { profile }
    }

    /// Check if the current profile is supported.
    /// Plan Section 16: Hardware tests run in a dedicated macOS lane.
    pub fn is_supported(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            return true;
        }
        #[cfg(not(target_os = "macos"))]
        {
            // CI-only: not a release target
            return false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compatibility_on_macos() {
        let suite = CompatibilitySuite::for_profile(HardwareProfileId("apple-m1".into()));
        #[cfg(target_os = "macos")]
        assert!(suite.is_supported());
    }
}
