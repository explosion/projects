-- DDL for parsed Wiki data.

CREATE TABLE entities (
    id TEXT PRIMARY KEY NOT NULL,
    -- This could be normalized. Not worth it at the moment though, since they aren't used.
    claims TEXT
);

-- The FTS5 virtual table implementation doesn't allow for indices, so we rely on ROWID to match entities.
-- This isn't great, but in a controlled setup this allows for stable matching.
-- Same for foreign keys.
CREATE VIRTUAL TABLE entities_texts USING fts5(
    entity_id UNINDEXED,
    name,
    description,
    label
);

CREATE TABLE articles (
    entity_id TEXT PRIMARY KEY NOT NULL,
    id TEXT NOT NULL,
    FOREIGN KEY(entity_id) REFERENCES entities(id)
);
CREATE UNIQUE INDEX idx_articles_id
ON articles (id);

-- Same here: no indices possible, relying on ROWID to match with articles.
CREATE VIRTUAL TABLE articles_texts USING fts5(
    entity_id UNINDEXED,
    title,
    content
);

CREATE TABLE properties_in_entities (
    property_id TEXT NOT NULL,
    from_entity_id TEXT NOT NULL,
    to_entity_id TEXT NOT NULL,
    PRIMARY KEY (property_id, from_entity_id, to_entity_id),
    FOREIGN KEY(from_entity_id) REFERENCES entities(id),
    FOREIGN KEY(to_entity_id) REFERENCES entities(id)
);
CREATE INDEX idx_properties_in_entities
ON properties_in_entities (property_id);

CREATE TABLE aliases_for_entities (
    alias TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    count INTEGER,
    PRIMARY KEY (alias, entity_id),
    FOREIGN KEY(entity_id) REFERENCES entities(id)
);
CREATE INDEX idx_aliases_for_entities_alias
ON aliases_for_entities (alias);
CREATE INDEX idx_aliases_for_entities_entity_id
ON aliases_for_entities (entity_id);