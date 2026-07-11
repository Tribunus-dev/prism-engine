-- Migration 001: Core schema for prism-kb knowledge base
-- Creates all tables, indexes, virtual tables, and constraints.

-- Core documents table
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    doc_type    TEXT NOT NULL CHECK(doc_type IN ('adr','architecture','analysis','reference','spec','pattern','decision','note','evidence','concept')),
    content_md  TEXT NOT NULL,
    file_path   TEXT,
    file_hash   TEXT,
    version     INTEGER DEFAULT 1,
    status      TEXT DEFAULT 'draft' CHECK(status IN ('draft','review','accepted','superseded','deprecated')),
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Sections for granular retrieval
CREATE TABLE IF NOT EXISTS sections (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    level       INTEGER NOT NULL CHECK(level BETWEEN 1 AND 6),
    heading     TEXT NOT NULL,
    content     TEXT NOT NULL,
    content_md  TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    word_count  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sections_document ON sections(document_id, seq);

-- Code blocks extracted from sections
CREATE TABLE IF NOT EXISTS code_blocks (
    id          TEXT PRIMARY KEY,
    section_id  TEXT NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    language    TEXT NOT NULL,
    content     TEXT NOT NULL,
    caption     TEXT,
    seq         INTEGER NOT NULL
);

-- Tags for classification
CREATE TABLE IF NOT EXISTS tags (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,
    category    TEXT
);

CREATE TABLE IF NOT EXISTS document_tags (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tag_id      INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (document_id, tag_id)
);

CREATE TABLE IF NOT EXISTS section_tags (
    section_id  TEXT NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    tag_id      INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (section_id, tag_id)
);

-- Arbitrary key-value metadata
CREATE TABLE IF NOT EXISTS metadata (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
    section_id  TEXT REFERENCES sections(id) ON DELETE CASCADE,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    CHECK (document_id IS NOT NULL OR section_id IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_metadata_key_value ON metadata(key, value);

-- Named concepts in the domain model
CREATE TABLE IF NOT EXISTS concepts (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    definition  TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS document_concepts (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    concept_id  TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    PRIMARY KEY (document_id, concept_id)
);

CREATE TABLE IF NOT EXISTS section_concepts (
    section_id  TEXT NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    concept_id  TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    PRIMARY KEY (section_id, concept_id)
);

-- Typed links between documents — the composition layer
CREATE TABLE IF NOT EXISTS links (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    source_doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_section_id  TEXT REFERENCES sections(id) ON DELETE SET NULL,
    target_doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_section_id  TEXT REFERENCES sections(id) ON DELETE SET NULL,
    relationship       TEXT NOT NULL CHECK(relationship IN ('references','refines','implements','supersedes','contradicts','depends_on','derives_from','validates','documents','relates_to')),
    description        TEXT,
    auto_extracted     INTEGER DEFAULT 0,
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_doc_id);
CREATE INDEX IF NOT EXISTS idx_links_target ON links(target_doc_id);
CREATE INDEX IF NOT EXISTS idx_links_relationship ON links(relationship);

-- Retrieval analytics
CREATE TABLE IF NOT EXISTS retrieval_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  TEXT NOT NULL,
    section_id   TEXT,
    agent_id     TEXT,
    query        TEXT,
    retrieved_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Full-text search indexes (standalone — populated via triggers)
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title, content_md,
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
    heading, content,
    tokenize='porter unicode61'
);

-- Initialize FTS with existing data (idempotent)
INSERT OR IGNORE INTO documents_fts(documents_fts) VALUES('rebuild');
INSERT OR IGNORE INTO sections_fts(sections_fts) VALUES('rebuild');
