import React from 'react';
import { AnalysisResult } from '../types';
import { SectionHeader } from '../components/SectionHeader';
import { MetricCard } from '../components/MetricCard';
import { ArticleCard } from '../components/ArticleCard';
import { formatPercent } from '../utils/formatters';
import { getSentimentColor } from '../utils/colors';

interface SentimentPageProps {
  result: AnalysisResult;
}

const SentimentPage: React.FC<SentimentPageProps> = ({ result }) => {
  const sf = result.sentiment_features;
  const articles = result.articles || [];

  const bullish = articles.filter((a) => a.sentiment.label === 'bullish');
  const bearish = articles.filter((a) => a.sentiment.label === 'bearish');
  const neutral = articles.filter((a) => a.sentiment.label === 'neutral');

  const ws = sf.weighted_sentiment;

  return (
    <div>
      <SectionHeader
        title="News Sentiment"
        subtitle={`${sf.news_volume} articles via NewsAPI + Yahoo RSS`}
      />

      <div className="grid grid-cols-4 gap-4 mb-8">
        <MetricCard
          label="Weighted Sentiment"
          value={ws.toFixed(3)}
          color={ws > 0.1 ? '#3ef5c8' : ws < -0.1 ? '#ff6b6b' : '#ffd166'}
        />
        <MetricCard
          label="Bullish"
          value={formatPercent(sf.positive_news_ratio, 0)}
          color="#3ef5c8"
        />
        <MetricCard
          label="Bearish"
          value={formatPercent(sf.negative_news_ratio, 0)}
          color="#ff6b6b"
        />
        <MetricCard
          label="Sent. Volatility"
          value={sf.sentiment_volatility.toFixed(3)}
          color="#ffd166"
        />
      </div>

      {articles.length === 0 ? (
        <div className="bg-card-bg border border-border-color rounded-lg p-4 text-text-muted">
          No news articles fetched. Check NEWSAPI_KEY in .env
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-8">
          <div>
            <h3
              className="font-mono text-sm font-bold uppercase tracking-wider mb-4"
              style={{ color: '#3ef5c8' }}
            >
              Bullish Articles
            </h3>
            <div>
              {bullish.slice(0, 4).map((a, i) => (
                <ArticleCard key={i} article={a} />
              ))}
            </div>
          </div>

          <div>
            <h3
              className="font-mono text-sm font-bold uppercase tracking-wider mb-4"
              style={{ color: '#ff6b6b' }}
            >
              Bearish Articles
            </h3>
            <div>
              {bearish.slice(0, 4).map((a, i) => (
                <ArticleCard key={i} article={a} />
              ))}
            </div>
          </div>
        </div>
      )}

      {neutral.length > 0 && (
        <div className="mt-8">
          <details className="bg-card-bg border border-border-color rounded-lg p-4">
            <summary className="cursor-pointer font-mono font-bold text-text-muted hover:text-text-primary">
              Neutral Articles ({neutral.length})
            </summary>
            <div className="mt-4">
              {neutral.slice(0, 5).map((a, i) => (
                <ArticleCard key={i} article={a} />
              ))}
            </div>
          </details>
        </div>
      )}
    </div>
  );
};

export default SentimentPage;
