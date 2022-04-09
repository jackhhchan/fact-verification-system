DROP TABLE IF EXISTS TABLE_PLACEHOLDER;
CREATE TABLE TABLE_PLACEHOLDER
(
    title    text,
    idx      number,
    sentence text
);

CREATE INDEX wiki ON TABLE_PLACEHOLDER(title, idx);