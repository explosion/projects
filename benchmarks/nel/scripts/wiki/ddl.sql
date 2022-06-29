-- DDL for parsed Wiki data.

CREATE TABLE entities (
    id TEXT PRIMARY KEY NOT NULL,
    title TEXT NOT NULL,
    wd_description TEXT,
    wp_description TEXT,
    label TEXT,
    article_title TEXT,
    article_text TEXT,
    -- This could be normalized. Not worth it at the moment though, since they aren't used.
    claims TEXT
);
CREATE UNIQUE INDEX idx_entities_title
ON entities (title);

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
    frequency INTEGER,
    PRIMARY KEY (alias, entity_id),
    FOREIGN KEY(entity_id) REFERENCES entities(id)
);
CREATE INDEX idx_aliases_for_entities_alias
ON aliases_for_entities (alias);
CREATE INDEX idx_aliases_for_entities_entity_id
ON aliases_for_entities (entity_id);