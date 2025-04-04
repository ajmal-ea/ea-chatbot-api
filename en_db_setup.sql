
                -- Enable vector extension if not already enabled
                CREATE EXTENSION IF NOT EXISTS vector;
                
                    CREATE TABLE IF NOT EXISTS "EN_website_data" (
                        id BIGINT GENERATED BY DEFAULT AS IDENTITY NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE NULL DEFAULT now(),
                        url TEXT NOT NULL,
                        service_name TEXT NULL,
                        sitemap_last_update TIMESTAMP WITH TIME ZONE NULL,
                        last_information_update TIMESTAMP WITH TIME ZONE NULL,
                        type TEXT NOT NULL,
                        vector_store_row_ids BIGINT[] NOT NULL DEFAULT '{}'::BIGINT[],
                        last_extracted TIMESTAMP WITH TIME ZONE NULL DEFAULT now(),
                        CONSTRAINT EN_website_data_pkey PRIMARY KEY (url),
                        CONSTRAINT EN_website_data_type_check CHECK (
                            (
                                type = ANY (
                                    ARRAY[
                                        'Other'::TEXT,
                                        'Solution'::TEXT,
                                        'Product'::TEXT,
                                        'Blog'::TEXT,
                                        'Service'::TEXT,
                                        'About'::TEXT,
                                        'Contact'::TEXT
                                    ]
                                )
                            )
                        )
                    ) TABLESPACE pg_default;
                    
                    CREATE TABLE IF NOT EXISTS "EN_documents" (
                        id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                        content TEXT,
                        metadata JSONB,
                        embedding VECTOR(1024),
                        url TEXT
                    );

                    -- Create vector index if available
                    CREATE INDEX IF NOT EXISTS EN_documents_embedding_idx
                    ON "EN_documents"
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                    
                    -- Create match function for semantic search
                    CREATE OR REPLACE FUNCTION match_EN_documents(
                        query_embedding VECTOR(1024),
                        match_count INT DEFAULT 5,
                        filter JSONB DEFAULT '{}'::jsonb,
                        match_threshold REAL DEFAULT 0.5
                    ) RETURNS TABLE (
                        id BIGINT,
                        content TEXT,
                        metadata JSONB,
                        url TEXT,
                        similarity REAL
                    ) 
                    LANGUAGE plpgsql 
                    AS $$
                    BEGIN
                        RETURN QUERY
                        SELECT
                            id,
                            content,
                            metadata,
                            url,
                            1 - (embedding <=> query_embedding) AS similarity
                        FROM
                            "EN_documents"
                        WHERE
                            1 - (embedding <=> query_embedding) > match_threshold
                        ORDER BY
                            embedding <=> query_embedding
                        LIMIT match_count;
                    END;
                    $$;
                    
                    -- Create function for updating vector_store_row_ids
                    CREATE OR REPLACE FUNCTION update_EN_vector_store_row_ids()
                    RETURNS trigger
                    LANGUAGE plpgsql
                    AS $function$
                    BEGIN
                      -- On INSERT: Add new document ID, ensuring uniqueness
                      IF TG_OP = 'INSERT' THEN
                        UPDATE "EN_website_data"
                        SET vector_store_row_ids = ARRAY(
                          SELECT DISTINCT unnest(array_append(vector_store_row_ids, NEW.id))
                        )
                        WHERE url = NEW.url;
                      END IF;

                      -- On DELETE: Remove document ID from array
                      IF TG_OP = 'DELETE' THEN
                        UPDATE "EN_website_data"
                        SET vector_store_row_ids = array_remove(vector_store_row_ids, OLD.id)
                        WHERE url = OLD.url;
                      END IF;

                      -- On UPDATE: If URL changes, move ID from old URL to new URL
                      IF TG_OP = 'UPDATE' AND OLD.url IS DISTINCT FROM NEW.url THEN
                        -- Remove from old URL
                        UPDATE "EN_website_data"
                        SET vector_store_row_ids = array_remove(vector_store_row_ids, OLD.id)
                        WHERE url = OLD.url;

                        -- Add to new URL
                        UPDATE "EN_website_data"
                        SET vector_store_row_ids = ARRAY(
                          SELECT DISTINCT unnest(array_append(vector_store_row_ids, NEW.id))
                        )
                        WHERE url = NEW.url;
                      END IF;

                      RETURN NULL;
                    END;
                    $function$;

                    -- Create trigger for automatically updating vector_store_row_ids
                    DROP TRIGGER IF EXISTS trigger_update_EN_vector_store_row_ids ON "EN_documents";

                    CREATE TRIGGER trigger_update_EN_vector_store_row_ids
                    AFTER DELETE OR INSERT OR UPDATE
                    ON "EN_documents"
                    FOR EACH ROW
                    EXECUTE FUNCTION update_EN_vector_store_row_ids();
                    