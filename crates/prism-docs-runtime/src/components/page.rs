//! Page components.

use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageRoute(pub String);
impl Component for PageRoute {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageTitle(pub String);
impl Component for PageTitle {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageBlurb(pub String);
impl Component for PageBlurb {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageChapterRefs(pub Vec<String>);
impl Component for PageChapterRefs {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageClaimRefs(pub Vec<String>);
impl Component for PageClaimRefs {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageAdrRefs(pub Vec<String>);
impl Component for PageAdrRefs {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageNext(pub String);
impl Component for PageNext {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PagePrev(pub String);
impl Component for PagePrev {}
