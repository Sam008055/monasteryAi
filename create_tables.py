import urllib.request
import json
import os

sql = """
CREATE TABLE IF NOT EXISTS public.monasteries (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    district TEXT NOT NULL,
    altitude_meters INTEGER,
    founded_year INTEGER,
    sect TEXT,
    glb_url TEXT NOT NULL,
    description TEXT,
    key_features JSONB,
    faqs JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.monasteries ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE tablename = 'monasteries' AND policyname = 'Public Read Monasteries'
    ) THEN
        CREATE POLICY "Public Read Monasteries" ON public.monasteries FOR SELECT USING (true);
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE tablename = 'monasteries' AND policyname = 'Service Insert Monasteries'
    ) THEN
        CREATE POLICY "Service Insert Monasteries" ON public.monasteries FOR ALL USING (true);
    END IF;
END
$$;
"""

url = 'https://api.supabase.com/v1/projects/ygdmzmqkztwpmkdozzsp/database/query'
payload = json.dumps({'query': sql}).encode()

supabase_access_token = os.environ.get("SUPABASE_ACCESS_TOKEN", "YOUR_SUPABASE_ACCESS_TOKEN")

req = urllib.request.Request(
    url,
    data=payload,
    headers={
        'Authorization': f'Bearer {supabase_access_token}',
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0'
    }
)

try:
    with urllib.request.urlopen(req) as resp:
        print('Database table creation result:', resp.read().decode())
except urllib.error.HTTPError as e:
    print('HTTP Error:', e.code, e.read().decode())
except Exception as e:
    print('Error:', e)
