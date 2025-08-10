-- Migration: Create payment and subscription tables
-- Description: Initial database schema for payment processing, subscriptions, and usage tracking
-- Created: 2024-01-01

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create user_profiles table for extended user information
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    name TEXT,
    stripe_customer_id TEXT UNIQUE,
    avatar_url TEXT,
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- Create subscriptions table
CREATE TABLE IF NOT EXISTS subscriptions (
    id TEXT PRIMARY KEY, -- Stripe subscription ID
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    customer_id TEXT NOT NULL, -- Stripe customer ID
    status TEXT NOT NULL CHECK (status IN ('active', 'canceled', 'incomplete', 'incomplete_expired', 'past_due', 'trialing', 'unpaid')),
    tier TEXT NOT NULL CHECK (tier IN ('free', 'pro', 'enterprise')) DEFAULT 'free',
    price_id TEXT NOT NULL,
    current_period_start TIMESTAMPTZ NOT NULL,
    current_period_end TIMESTAMPTZ NOT NULL,
    cancel_at_period_end BOOLEAN DEFAULT FALSE,
    canceled_at TIMESTAMPTZ,
    trial_start TIMESTAMPTZ,
    trial_end TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create payment_history table
CREATE TABLE IF NOT EXISTS payment_history (
    id TEXT PRIMARY KEY, -- Stripe invoice/payment intent ID
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    customer_id TEXT NOT NULL,
    subscription_id TEXT REFERENCES subscriptions(id) ON DELETE SET NULL,
    amount INTEGER NOT NULL, -- Amount in cents
    currency TEXT NOT NULL DEFAULT 'usd',
    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed', 'pending', 'canceled', 'refunded')),
    description TEXT NOT NULL,
    invoice_url TEXT,
    receipt_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create usage_records table for tracking API usage
CREATE TABLE IF NOT EXISTS usage_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    subscription_id TEXT REFERENCES subscriptions(id) ON DELETE SET NULL,
    usage_type TEXT NOT NULL CHECK (usage_type IN ('api_call', 'research_request', 'token_usage', 'mcp_server_call')),
    quantity INTEGER NOT NULL DEFAULT 1,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    endpoint TEXT,
    model_used TEXT,
    tokens_consumed INTEGER,
    cost DECIMAL(10,4),
    metadata JSONB DEFAULT '{}'
);

-- Create subscription_tiers table for tier configuration
CREATE TABLE IF NOT EXISTS subscription_tiers (
    tier TEXT PRIMARY KEY CHECK (tier IN ('free', 'pro', 'enterprise')),
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    monthly_price DECIMAL(10,2) NOT NULL,
    stripe_price_id TEXT,
    features JSONB NOT NULL DEFAULT '{}',
    limits JSONB NOT NULL DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create api_keys table for user API access
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    key_hash TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    prefix TEXT NOT NULL,
    permissions TEXT[] DEFAULT '{}',
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create usage_summaries table for aggregated usage data
CREATE TABLE IF NOT EXISTS usage_summaries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    api_calls INTEGER DEFAULT 0,
    research_requests INTEGER DEFAULT 0,
    tokens_used INTEGER DEFAULT 0,
    mcp_server_calls INTEGER DEFAULT 0,
    total_cost DECIMAL(10,4) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, period_start, period_end)
);

-- Create webhook_events table for tracking processed webhooks
CREATE TABLE IF NOT EXISTS webhook_events (
    id TEXT PRIMARY KEY, -- Stripe event ID
    event_type TEXT NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMPTZ,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    raw_data JSONB DEFAULT '{}'
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_user_profiles_stripe_customer ON user_profiles(stripe_customer_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_email ON user_profiles(email);

CREATE INDEX IF NOT EXISTS idx_subscriptions_user_id ON subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_customer_id ON subscriptions(customer_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status);
CREATE INDEX IF NOT EXISTS idx_subscriptions_tier ON subscriptions(tier);

CREATE INDEX IF NOT EXISTS idx_payment_history_user_id ON payment_history(user_id);
CREATE INDEX IF NOT EXISTS idx_payment_history_customer_id ON payment_history(customer_id);
CREATE INDEX IF NOT EXISTS idx_payment_history_subscription_id ON payment_history(subscription_id);
CREATE INDEX IF NOT EXISTS idx_payment_history_status ON payment_history(status);
CREATE INDEX IF NOT EXISTS idx_payment_history_created_at ON payment_history(created_at);

CREATE INDEX IF NOT EXISTS idx_usage_records_user_id ON usage_records(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_records_subscription_id ON usage_records(subscription_id);
CREATE INDEX IF NOT EXISTS idx_usage_records_usage_type ON usage_records(usage_type);
CREATE INDEX IF NOT EXISTS idx_usage_records_timestamp ON usage_records(timestamp);

CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(prefix);
CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON api_keys(is_active);

CREATE INDEX IF NOT EXISTS idx_usage_summaries_user_id ON usage_summaries(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_summaries_period ON usage_summaries(period_start, period_end);

CREATE INDEX IF NOT EXISTS idx_webhook_events_event_type ON webhook_events(event_type);
CREATE INDEX IF NOT EXISTS idx_webhook_events_processed ON webhook_events(processed);
CREATE INDEX IF NOT EXISTS idx_webhook_events_created_at ON webhook_events(created_at);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_user_profiles_updated_at BEFORE UPDATE ON user_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_subscriptions_updated_at BEFORE UPDATE ON subscriptions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_subscription_tiers_updated_at BEFORE UPDATE ON subscription_tiers FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_api_keys_updated_at BEFORE UPDATE ON api_keys FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default subscription tiers
INSERT INTO subscription_tiers (tier, name, description, monthly_price, features, limits) VALUES
('free', 'Free', 'Basic research capabilities with limited usage', 0.00, 
 '{"research_requests": true, "mcp_servers": ["reddit", "youtube"], "api_access": false, "priority_support": false, "advanced_rag": false}',
 '{"concurrent_research_units": 1, "monthly_research_requests": 10, "monthly_api_calls": 0, "monthly_tokens": 50000}'),
('pro', 'Pro', 'Enhanced research with GitHub integration and API access', 29.99,
 '{"research_requests": true, "mcp_servers": ["reddit", "youtube", "github"], "api_access": true, "priority_support": true, "advanced_rag": true}',
 '{"concurrent_research_units": 5, "monthly_research_requests": 500, "monthly_api_calls": 10000, "monthly_tokens": 1000000}'),
('enterprise', 'Enterprise', 'Unlimited research with custom integrations and dedicated support', 99.99,
 '{"research_requests": true, "mcp_servers": ["reddit", "youtube", "github"], "api_access": true, "priority_support": true, "advanced_rag": true, "custom_integrations": true, "dedicated_support": true}',
 '{"concurrent_research_units": 20, "monthly_research_requests": -1, "monthly_api_calls": -1, "monthly_tokens": -1}')
ON CONFLICT (tier) DO NOTHING;

-- Enable Row Level Security (RLS) for user data protection
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE payment_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_summaries ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for user_profiles
CREATE POLICY "Users can view own profile" ON user_profiles FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can update own profile" ON user_profiles FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own profile" ON user_profiles FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Create RLS policies for subscriptions
CREATE POLICY "Users can view own subscriptions" ON subscriptions FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Service role can manage subscriptions" ON subscriptions FOR ALL USING (auth.role() = 'service_role');

-- Create RLS policies for payment_history
CREATE POLICY "Users can view own payment history" ON payment_history FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Service role can manage payment history" ON payment_history FOR ALL USING (auth.role() = 'service_role');

-- Create RLS policies for usage_records
CREATE POLICY "Users can view own usage records" ON usage_records FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Service role can manage usage records" ON usage_records FOR ALL USING (auth.role() = 'service_role');

-- Create RLS policies for api_keys
CREATE POLICY "Users can view own API keys" ON api_keys FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can manage own API keys" ON api_keys FOR ALL USING (auth.uid() = user_id);

-- Create RLS policies for usage_summaries
CREATE POLICY "Users can view own usage summaries" ON usage_summaries FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Service role can manage usage summaries" ON usage_summaries FOR ALL USING (auth.role() = 'service_role');

-- Create function to get user subscription status
CREATE OR REPLACE FUNCTION get_user_subscription_status(user_uuid UUID)
RETURNS TABLE (
    subscription_id TEXT,
    tier TEXT,
    status TEXT,
    current_period_end TIMESTAMPTZ,
    cancel_at_period_end BOOLEAN,
    features JSONB,
    limits JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.id,
        s.tier,
        s.status,
        s.current_period_end,
        s.cancel_at_period_end,
        st.features,
        st.limits
    FROM subscriptions s
    JOIN subscription_tiers st ON s.tier = st.tier
    WHERE s.user_id = user_uuid 
    AND s.status = 'active'
    ORDER BY s.created_at DESC
    LIMIT 1;
    
    -- If no active subscription found, return free tier
    IF NOT FOUND THEN
        RETURN QUERY
        SELECT 
            NULL::TEXT,
            st.tier,
            'active'::TEXT,
            NULL::TIMESTAMPTZ,
            FALSE,
            st.features,
            st.limits
        FROM subscription_tiers st
        WHERE st.tier = 'free';
    END IF;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to check usage limits
CREATE OR REPLACE FUNCTION check_usage_limit(
    user_uuid UUID,
    usage_type_param TEXT,
    period_start_param TIMESTAMPTZ DEFAULT date_trunc('month', NOW()),
    period_end_param TIMESTAMPTZ DEFAULT NOW()
)
RETURNS TABLE (
    allowed BOOLEAN,
    limit_value INTEGER,
    current_usage INTEGER,
    remaining INTEGER
) AS $$
DECLARE
    subscription_limits JSONB;
    limit_key TEXT;
    usage_limit INTEGER;
    current_count INTEGER;
BEGIN
    -- Get subscription limits
    SELECT limits INTO subscription_limits
    FROM get_user_subscription_status(user_uuid)
    LIMIT 1;
    
    -- Construct limit key
    limit_key := 'monthly_' || usage_type_param;
    
    -- Get limit value
    usage_limit := (subscription_limits ->> limit_key)::INTEGER;
    
    -- Get current usage
    SELECT COALESCE(SUM(quantity), 0)::INTEGER INTO current_count
    FROM usage_records
    WHERE user_id = user_uuid
    AND usage_type = usage_type_param
    AND timestamp >= period_start_param
    AND timestamp <= period_end_param;
    
    -- Return results
    RETURN QUERY SELECT 
        CASE 
            WHEN usage_limit = -1 THEN TRUE  -- Unlimited
            WHEN current_count < usage_limit THEN TRUE
            ELSE FALSE
        END,
        usage_limit,
        current_count,
        CASE 
            WHEN usage_limit = -1 THEN -1  -- Unlimited
            ELSE GREATEST(0, usage_limit - current_count)
        END;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
