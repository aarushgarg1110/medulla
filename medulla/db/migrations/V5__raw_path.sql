-- V5: Add raw_path to wiki_pages for source-type pages.
-- Enables O(1) reverse lookup: raw file → source slug for medulla remove raw/file.
ALTER TABLE wiki_pages ADD COLUMN raw_path TEXT;
