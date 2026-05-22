// API Response types
export interface AnalysisResult {
  ticker: string;
  current_price: number;
  daily_return: number;
  return_30d: number;
  return_90d: number;
  volume: number;
  
  // Forecast
  outlook: 'Bullish' | 'Bearish' | 'Neutral';
  pred_return: number;
  price_forecast_low: number;
  price_forecast_mid: number;
  price_forecast_high: number;
  model_confidence: number;
  forecast_horizon_days: number;
  model_used: string;
  
  // Risk
  risk_score: number;
  risk_label: string;
  risk_color: string;
  ann_volatility: number;
  var_1d: number;
  cvar_1d: number;
  max_drawdown: number;
  beta: number;
  pred_downside_prob: number;
  var_horizon: number;
  
  // Sentiment
  sentiment_features: SentimentFeatures;
  articles: NewsArticle[];
  sentiment_adjustment: SentimentAdjustment;
  
  // Chart data
  chart_data: ChartData;
  
  timestamp: string;
  error?: string;
}

export interface ChartData {
  dates: string[];
  prices: number[];
  volatility_windows: number[];
  drawdown_windows: number[];
}

export interface SentimentFeatures {
  weighted_sentiment: number;
  positive_news_ratio: number;
  negative_news_ratio: number;
  sentiment_volatility: number;
  news_volume: number;
}

export interface NewsArticle {
  title: string;
  url: string;
  source: string;
  age_hours: number;
  sentiment: {
    label: 'bullish' | 'bearish' | 'neutral';
    score: number;
  };
}

export interface SentimentAdjustment {
  return_adj: number;
  vol_adj: number;
  downside_adj: number;
  uncertainty_mult: number;
}

export interface AnalysisRequest {
  ticker: string;
  forecast_horizon: number;
}

export interface ApiResponse<T> {
  data: T;
  error?: string;
  success: boolean;
}
