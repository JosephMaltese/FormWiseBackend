import { createClient } from '@supabase/supabase-js'

const supabaseURL = process.env.REACT_APP_SUPABASE_URL;
const supabaseKey = process.env.REACT_APP_SUPABASE_PUBLISHABLE_KEY;
// Create a single supabase client for interacting with your database
const supabase = createClient(supabaseURL, supabaseKey);

export default supabase;