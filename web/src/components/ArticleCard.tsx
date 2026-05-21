import React from 'react';
import { NewsArticle } from '../types';
import { getSentimentColor } from '../utils/colors';
import { formatAge } from '../utils/formatters';

interface ArticleCardProps {
  article: NewsArticle;
}

export const ArticleCard: React.FC<ArticleCardProps> = ({ article }) => {
  const color = getSentimentColor(article.sentiment.label);

  return (
    <a
      href={article.url}
      target="_blank"
      rel="noopener noreferrer"
      className="block bg-card-bg border-l-2 rounded px-4 py-3 mb-3 hover:bg-opacity-80 transition-all"
      style={{ borderLeftColor: color }}
    >
      <h3 className="text-sm font-semibold text-text-primary leading-5 mb-2 line-clamp-2">
        {article.title}
      </h3>
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-mono text-text-muted">
          {article.source} · {formatAge(article.age_hours)}
        </span>
        <span
          className="text-xs font-mono font-bold px-2 py-1 rounded"
          style={{ backgroundColor: `${color}18`, color }}
        >
          {article.sentiment.label.toUpperCase()} {article.sentiment.score:+.2f}
        </span>
      </div>
    </a>
  );
};
