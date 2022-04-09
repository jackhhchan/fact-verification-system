DROP TABLE IF EXISTS TABLE_PLACEHOLDER;
CREATE TABLE TABLE_PLACEHOLDER
(
    title    text,
    idx      number,
    sentence text
);

CREATE INDEX wiki_title ON TABLE_PLACEHOLDER(title);
CREATE INDEX wiki_idx ON TABLE_PLACEHOLDER(idx);