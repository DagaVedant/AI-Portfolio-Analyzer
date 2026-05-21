import React from 'react';
import { AnalysisResult } from '../types';
import { SectionHeader } from '../components/SectionHeader';
import { MetricCard } from '../components/MetricCard';
import { formatPrice, formatPercent } from '../utils/formatters';
import { getReturnColor } from '../utils/colors';

interface OverviewPageProps {
  result: AnalysisResult;
}

const OverviewPage: React.FC<OverviewPageProps> = ({ result }) => {
  const r1d = result.daily_return;
  const r30 = result.return_30d;
  const r90 = result.return_90d;

  return (
    <div>
      <SectionHeader title="Market Snapshot" subtitle={result.ticker} />
      
      <div className="grid grid-cols-4 gap-4 mb-8">
        <MetricCard label="Current Price" value={formatPrice(result.current_price)} />
        <MetricCard
          label="1-Day Return"
          value={formatPercent(r1d)}
          color={getReturnColor(r1d)}
        />
        <MetricCard
          label="30-Day Return"
          value={formatPercent(r30)}
          color={getReturnColor(r30)}
        />
        <MetricCard
          label="90-Day Return"
          value={formatPercent(r90)}
          color={getReturnColor(r90)}
        />
      </div>

      <SectionHeader title="Price Predictions" subtitle={`${result.forecast_horizon_days}-Day Horizon`} />
      
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          label="Bear Case"
          value={formatPrice(result.price_forecast_low)}
          delta={formatPercent((result.price_forecast_low - result.current_price) / result.current_price)}
          color="#ff6b6b"
        />
        <MetricCard
          label="Base Case"
          value={formatPrice(result.price_forecast_mid)}
          delta={formatPercent((result.price_forecast_mid - result.current_price) / result.current_price)}
          color="#ffd166"
        />
        <MetricCard
          label="Bull Case"
          value={formatPrice(result.price_forecast_high)}
          delta={formatPercent((result.price_forecast_high - result.current_price) / result.current_price)}
          color="#3ef5c8"
        />
        <MetricCard
          label="Model Confidence"
          value={`${(result.model_confidence * 100).toFixed(0)}%`}
          color={result.model_confidence > 0.6 ? '#3ef5c8' : '#ffd166'}
        />
      </div>

      <div className="mt-8 bg-card-bg border border-border-color rounded-lg p-4">
        <div className="text-xs font-mono text-text-muted uppercase tracking-wider mb-2">Model Info</div>
        <div className="flex gap-4 font-mono text-sm flex-wrap">
          <span>Model: <span className="text-accent-blue">{result.model_used}</span></span>
          <span>Timestamp: <span className="text-text-muted">{result.timestamp.substring(0, 19)} UTC</span></span>
          <span>Horizon: <span className="text-accent-yellow">{result.forecast_horizon_days} days</span></span>
        </div>
        <div className="text-xs font-mono text-text-muted mt-3 leading-relaxed">
          Educational Disclaimer: All predictions, risk metrics and portfolio suggestions are generated
          by machine-learning models for educational purposes only and do NOT constitute financial advice.
        </div>
      </div>
    </div>
  );
};

export default OverviewPage;
