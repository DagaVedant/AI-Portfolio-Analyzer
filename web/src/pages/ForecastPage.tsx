import React from 'react';
import { AnalysisResult } from '../types';
import { SectionHeader } from '../components/SectionHeader';
import { MetricCard } from '../components/MetricCard';
import { formatPrice, formatPercent } from '../utils/formatters';
import { getOutlookColor, getReturnColor } from '../utils/colors';

interface ForecastPageProps {
  result: AnalysisResult;
}

const ForecastPage: React.FC<ForecastPageProps> = ({ result }) => {
  const ret = result.pred_return;
  
  return (
    <div>
      <div className="mb-8">
        <div className="flex items-center gap-4 mb-4">
          <h2 className="text-lg font-bold">
            Forecast — {result.outlook === 'Bullish' ? 'Bull' : result.outlook === 'Bearish' ? 'Bear' : 'Neutral'}
          </h2>
          <span className="text-sm font-mono text-text-muted">
            Horizon: {result.forecast_horizon_days} trading days
          </span>
        </div>

        <div className="grid grid-cols-4 gap-4 mb-8">
          <MetricCard
            label="Predicted Return"
            value={formatPercent(ret)}
            color={getReturnColor(ret)}
          />
          <MetricCard
            label="Forecast Low"
            value={formatPrice(result.price_forecast_low)}
            color="#ff6b6b"
          />
          <MetricCard
            label="Forecast Mid"
            value={formatPrice(result.price_forecast_mid)}
            color="#ffd166"
          />
          <MetricCard
            label="Forecast High"
            value={formatPrice(result.price_forecast_high)}
            color="#3ef5c8"
          />
        </div>

        <div className="grid grid-cols-3 gap-4">
          <MetricCard
            label="Model Confidence"
            value={`${(result.model_confidence * 100).toFixed(0)}%`}
            color={result.model_confidence > 0.6 ? '#3ef5c8' : '#ffd166'}
          />
          <MetricCard
            label="Outlook"
            value={result.outlook}
            color={getOutlookColor(result.outlook as any)}
          />
          <MetricCard label="Model" value={result.model_used} color="#58a6ff" />
        </div>
      </div>

      {result.sentiment_adjustment && (
        <div>
          <SectionHeader title="Sentiment Adjustments Applied" />
          <div className="grid grid-cols-4 gap-4">
            <MetricCard
              label="Return Adj"
              value={formatPercent(result.sentiment_adjustment.return_adj, 3)}
            />
            <MetricCard
              label="Vol Adj"
              value={`+${(result.sentiment_adjustment.vol_adj * 100).toFixed(1)}%`}
            />
            <MetricCard
              label="Downside Adj"
              value={`+${(result.sentiment_adjustment.downside_adj * 100).toFixed(1)}%`}
            />
            <MetricCard
              label="Uncertainty"
              value={`${result.sentiment_adjustment.uncertainty_mult.toFixed(2)}x`}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default ForecastPage;
