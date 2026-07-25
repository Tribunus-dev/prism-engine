//! World bootstrap — manifest → typed world.
//!
//! This module owns the canonical authority for inserting the
//! manifest's entities into a `prism_ecs_core::World`. It is the
//! only place where direct mutation is allowed (`Bootstrap` policy).
//! After this module returns, the policy flips to
//! `TransactionalOnly` and any further mutation goes through
//! `WorldTxn`.

use std::path::Path;

use prism_docs_content::{
    Adr, AdrStatus, Chapter, Claim, ContentManifest, ExistenceState, Link, LinkKind, Page,
};
use prism_ecs_core::{Entity, EntityKind, MutationPolicy, World};

use crate::components::adr::{
    AdrBodyPath, AdrContext, AdrConsequences, AdrDecision, AdrNumber, AdrSlug,
    AdrStatusComponent, AdrSupersedes, AdrTitle,
};
use crate::components::body::{MarkdownBody, MarkdownSections, MarkdownSource, MarkdownSourcePath};
use crate::components::chapter::{
    ChapterBlurb, ChapterBodyPath, ChapterIntent, ChapterOrder, ChapterReadingMinutes,
    ChapterSlug, ChapterTitle,
};
use crate::components::claim::{
    ClaimClassComponent, ClaimFramedBy, ClaimSourceRefs, ClaimText, ExistenceStateComponent,
    KnowledgeStateComponent,
};
use crate::components::identity::{SiteEntityId, SiteEntityKind};
use crate::components::link::{LinkFrom, LinkKindComponent, LinkTo};
use crate::components::page::{
    PageAdrRefs, PageBlurb, PageChapterRefs, PageClaimRefs, PageNext, PagePrev, PageRoute,
    PageTitle,
};
use crate::error::RuntimeError;
use crate::resources::site_config::SiteConfig;
use crate::resources::visitor_state::VisitorState;

/// Marker for the docs site world. We use a newtype so the type
/// system can distinguish our entities from other `prism-ecs-core`
/// worlds in the same binary.
#[derive(Debug)]
pub struct DocsSiteWorld;

/// Spawns an entity in the docs world with kind `Node`. We use the
/// core's `Node` kind as a generic placeholder; the runtime
/// discriminates with `SiteEntityKind`.
fn spawn_node(world: &mut World) -> Result<Entity, RuntimeError> {
    world
        .spawn(EntityKind::Node, None)
        .map(|s| s.entity.into())
        .map_err(RuntimeError::from)
}

/// Result of bootstrap. The world is left in `TransactionalOnly`
/// policy, ready for the schedule to run.
pub struct BootstrappedWorld {
    pub world: World,
    pub entity_count: usize,
}

impl BootstrappedWorld {
    pub fn entity_count(&self) -> usize {
        self.entity_count
    }
}

/// Build a `prism_ecs_core::World` populated from a content manifest.
///
/// This is the single composition point for translating manifest
/// entries into typed components. Every component attached to a
/// docs entity must be added through this function. Direct
/// `world.add_component` outside bootstrap is a constitutional
/// violation.
pub fn build_static_world(
    manifest: &ContentManifest,
    site_config: SiteConfig,
) -> Result<BootstrappedWorld, RuntimeError> {
    // Bootstrap policy is the default for a new World. We can add
    // entities directly. After this function returns, callers
    // should flip the policy via `seal_for_runtime` (below) before
    // running any systems.
    let mut world = World::new();
    world.set_direct_mutation_allowed(true);

    world.add_resource(site_config);
    world.add_resource(VisitorState::default());

    let mut count = 0;

    for chapter in &manifest.chapters {
        let id = spawn_node(&mut world)?;
        insert_chapter(&mut world, id, chapter)?;
        count += 1;
    }

    for adr in &manifest.adrs {
        let id = spawn_node(&mut world)?;
        insert_adr(&mut world, id, adr)?;
        count += 1;
    }

    for claim in &manifest.claims {
        let id = spawn_node(&mut world)?;
        insert_claim(&mut world, id, claim)?;
        count += 1;
    }

    for page in &manifest.pages {
        let id = spawn_node(&mut world)?;
        insert_page(&mut world, id, page)?;
        count += 1;
    }

    for link in &manifest.links {
        let id = spawn_node(&mut world)?;
        insert_link(&mut world, id, link)?;
        count += 1;
    }

    Ok(BootstrappedWorld {
        world,
        entity_count: count,
    })
}

/// Flip the world's mutation policy to `TransactionalOnly` after
/// bootstrap. Any subsequent direct mutation is rejected.
pub fn seal_for_runtime(world: &mut World) {
    world.set_mutation_policy(MutationPolicy::TransactionalOnly);
}

fn insert_chapter(
    world: &mut World,
    entity: Entity,
    chapter: &Chapter,
) -> Result<(), RuntimeError> {
    world
        .add_component(entity, SiteEntityKind::Chapter)
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, SiteEntityId(chapter.id.clone()))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, ChapterTitle(chapter.title.clone()))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, ChapterSlug(chapter.slug.clone()))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, ChapterOrder(chapter.order))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, ChapterIntent(chapter.intent.clone()))
        .map_err(RuntimeError::from)?;
    if let Some(blurb) = &chapter.blurb {
        world
            .add_component(entity, ChapterBlurb(blurb.clone()))
            .map_err(RuntimeError::from)?;
    }
    if let Some(min) = chapter.reading_minutes {
        world
            .add_component(entity, ChapterReadingMinutes(min))
            .map_err(RuntimeError::from)?;
    }
    if !chapter.source_refs.is_empty() {
        let refs: Vec<String> = chapter
            .source_refs
            .iter()
            .map(|r| r.to_anchor())
            .collect();
        world
            .add_component(entity, ClaimSourceRefs(refs))
            .map_err(RuntimeError::from)?;
    }
    world
        .add_component(
            entity,
            ChapterBodyPath(chapter.body_path.to_string_lossy().to_string()),
        )
        .map_err(RuntimeError::from)?;
    Ok(())
}

fn insert_adr(world: &mut World, entity: Entity, adr: &Adr) -> Result<(), RuntimeError> {
    world
        .add_component(entity, SiteEntityKind::Adr)
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, SiteEntityId(adr.id.clone()))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, AdrTitle(adr.title.clone()))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, AdrSlug(adr.slug.clone()))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, AdrNumber(adr.number))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, map_adr_status(adr.status))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, AdrContext(adr.context.clone()))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, AdrDecision(adr.decision.clone()))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, AdrConsequences(adr.consequences.clone()))
        .map_err(RuntimeError::from)?;
    if let Some(supersedes) = &adr.supersedes {
        world
            .add_component(entity, AdrSupersedes(supersedes.to_string()))
            .map_err(RuntimeError::from)?;
    }
    world
        .add_component(
            entity,
            AdrBodyPath(adr.body_path.to_string_lossy().to_string()),
        )
        .map_err(RuntimeError::from)?;
    Ok(())
}

fn insert_claim(world: &mut World, entity: Entity, claim: &Claim) -> Result<(), RuntimeError> {
    world
        .add_component(entity, SiteEntityKind::Claim)
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, SiteEntityId(claim.id.clone()))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, ClaimText(claim.text.clone()))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, ClaimClassComponent(claim.class.as_str().into()))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, KnowledgeStateComponent(claim.state.as_str().into()))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, ExistenceStateComponent(ExistenceState::Active.as_str().into()))
        .map_err(RuntimeError::from)?;
    if !claim.source_refs.is_empty() {
        let refs: Vec<String> = claim.source_refs.iter().map(|r| r.to_anchor()).collect();
        world
            .add_component(entity, ClaimSourceRefs(refs))
            .map_err(RuntimeError::from)?;
    }
    if let Some(framed_by) = &claim.framed_by {
        world
            .add_component(entity, ClaimFramedBy(framed_by.to_string()))
            .map_err(RuntimeError::from)?;
    }
    Ok(())
}

fn insert_page(world: &mut World, entity: Entity, page: &Page) -> Result<(), RuntimeError> {
    world
        .add_component(entity, SiteEntityKind::Page)
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, SiteEntityId(page.id.clone()))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, PageRoute(page.route.clone()))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, PageTitle(page.title.clone()))
        .map_err(RuntimeError::from)?;
    if let Some(blurb) = &page.blurb {
        world
            .add_component(entity, PageBlurb(blurb.clone()))
            .map_err(RuntimeError::from)?;
    }
    if !page.chapter_refs.is_empty() {
        let refs: Vec<String> = page.chapter_refs.iter().map(|r| r.to_string()).collect();
        world
            .add_component(entity, PageChapterRefs(refs))
            .map_err(RuntimeError::from)?;
    }
    if !page.claim_refs.is_empty() {
        let refs: Vec<String> = page.claim_refs.iter().map(|r| r.to_string()).collect();
        world
            .add_component(entity, PageClaimRefs(refs))
            .map_err(RuntimeError::from)?;
    }
    if !page.adr_refs.is_empty() {
        let refs: Vec<String> = page.adr_refs.iter().map(|r| r.to_string()).collect();
        world
            .add_component(entity, PageAdrRefs(refs))
            .map_err(RuntimeError::from)?;
    }
    if let Some(next) = &page.next {
        world
            .add_component(entity, PageNext(next.to_string()))
            .map_err(RuntimeError::from)?;
    }
    if let Some(prev) = &page.prev {
        world
            .add_component(entity, PagePrev(prev.to_string()))
            .map_err(RuntimeError::from)?;
    }
    Ok(())
}

fn insert_link(world: &mut World, entity: Entity, link: &Link) -> Result<(), RuntimeError> {
    world
        .add_component(entity, SiteEntityKind::Link)
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, LinkFrom(SiteEntityId(link.from.clone())))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, LinkTo(SiteEntityId(link.to.clone())))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, map_link_kind(link.kind))
        .map_err(RuntimeError::from)?;
    Ok(())
}

fn map_adr_status(s: AdrStatus) -> AdrStatusComponent {
    AdrStatusComponent(s)
}

fn map_link_kind(k: LinkKind) -> LinkKindComponent {
    match k {
        LinkKind::Frames => LinkKindComponent::Frames,
        LinkKind::Follows => LinkKindComponent::Follows,
        LinkKind::Depends => LinkKindComponent::Depends,
        LinkKind::Constrained => LinkKindComponent::Constrained,
        LinkKind::Supersedes => LinkKindComponent::Supersedes,
        LinkKind::Composes => LinkKindComponent::Composes,
    }
}

/// Build a body from a markdown source. Reads the file at
/// `body_path` (relative to `content_root`) and inserts the
/// rendered `MarkdownBody` component on the entity.
pub fn attach_body(
    world: &mut World,
    entity: Entity,
    content_root: &Path,
    body_path: &str,
) -> Result<(), RuntimeError> {
    let path = content_root.join(body_path);
    let raw = std::fs::read_to_string(&path).map_err(|e| RuntimeError::Content(
        prism_docs_content::ContentError::Io {
            path: path.clone(),
            source: e,
        },
    ))?;
    let mut doc = prism_docs_content::MarkdownDocument::parse(&path, &raw)
        .map_err(RuntimeError::from)?;
    let html = doc.render().to_string();
    let body = doc.body_markdown.clone();
    let sections = extract_sections(&body);
    world
        .add_component(entity, MarkdownSource(body))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, MarkdownSourcePath(body_path.into()))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, MarkdownBody(html))
        .map_err(RuntimeError::from)?;
    world
        .add_component(entity, MarkdownSections(sections))
        .map_err(RuntimeError::from)?;
    Ok(())
}

fn extract_sections(body: &str) -> Vec<crate::components::body::Section> {
    use pulldown_cmark::{Event, Parser, Tag, TagEnd};
    let mut sections = Vec::new();
    let parser = Parser::new(body);
    let mut current_level: Option<u8> = None;
    let mut current_title = String::new();
    for event in parser {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                current_level = Some(level as u8);
                current_title.clear();
            }
            Event::End(TagEnd::Heading(_)) => {
                if let Some(level) = current_level.take() {
                    let anchor = slugify(&current_title);
                    sections.push(crate::components::body::Section {
                        level,
                        anchor,
                        title: current_title.trim().to_string(),
                    });
                }
            }
            Event::Text(t) | Event::Code(t) => {
                if current_level.is_some() {
                    current_title.push_str(&t);
                }
            }
            _ => {}
        }
    }
    sections
}

fn slugify(s: &str) -> String {
    let mut out = String::new();
    let mut last_dash = false;
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash && !out.is_empty() {
            out.push('-');
            last_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::site_config::SiteConfig;
    use prism_docs_content::ontology::KnowledgeState;
    use prism_docs_content::{
        Chapter, Claim, ClaimClass, ContentManifest, EntityId, Page,
    };

    fn minimal_chapter(id: &str, slug: &str, title: &str, order: u32) -> Chapter {
        Chapter {
            id: EntityId::new(id).unwrap(),
            slug: slug.into(),
            title: title.into(),
            order,
            intent: "an intent".into(),
            blurb: None,
            reading_minutes: None,
            source_refs: vec![],
            body_path: std::path::PathBuf::from("chapters/x.md"),
        }
    }

    fn minimal_claim(id: &str, text: &str) -> Claim {
        Claim {
            id: EntityId::new(id).unwrap(),
            text: text.into(),
            class: ClaimClass::Architectural,
            state: KnowledgeState::Verified,
            source_refs: vec![],
            framed_by: None,
        }
    }

    fn minimal_page(id: &str, route: &str) -> Page {
        Page {
            id: EntityId::new(id).unwrap(),
            route: route.into(),
            title: "Test".into(),
            blurb: None,
            chapter_refs: vec![],
            claim_refs: vec![],
            adr_refs: vec![],
            next: None,
            prev: None,
        }
    }

    #[test]
    fn build_static_world_creates_entities() {
        let mut manifest = ContentManifest::default();
        manifest.chapters.push(minimal_chapter(
            "chapter:home",
            "home",
            "Home",
            1,
        ));
        manifest.claims.push(minimal_claim("claim:x", "x"));
        manifest.pages.push(minimal_page("page:home", "/"));
        let boot =
            build_static_world(&manifest, SiteConfig::default()).unwrap();
        assert_eq!(boot.entity_count(), 3);
    }
}
