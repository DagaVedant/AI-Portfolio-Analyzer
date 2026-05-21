import React from 'react';
import { AnalysisResult } from '../types';
import { SectionHeader } from '../components/SectionHeader';
import { MetricCard } from '../components/MetricCard';
import { formatPercent } from '../utils/formatters';

interface RiskPageProps {
  result: AnalysisResult;
}

const RiskPage: React.FC<RiskPageProps> = ({ result }) => {
  const risk = result.risk_score;
  
  return (
    <div>
      <SectionHeader title="Risk Analysis" />
      
      <div className="mb-8 bg-card-bg border rounded-lg p-6 text-center" style={{ borderColor: result.risk_color + '33' }}>
        <div className="text-xs font-mono text-text-muted uppercase tracking-wider mb-2">
          Composite Risk Score
        </div>
        <div style={{ color: result.risk_color }} className="text-5xl font-mono font-bold my-4">
          {Math.round(risk)}
        </div>
        <div style={{ color: result.risk_color }} className="text-lg font-bold uppercase tracking-wider">
          {result.risk_label}
        </div>
      </div>

      <div className="grid grid-cols-4 gap-4 mb-8">
        <MetricCard
          label="Annual Volatility"
          value={formatPercent(result.ann_volatility)}
          color={result.ann_volatility < 0.3 ? '#ffd166' : '#ff6b6b'}
        />
        <MetricCard
          label="VaR 1-Day 95%"
          value={formatPercent(Math.abs(result.var_1d))}
          color="#ffd166"
        />
        <MetricCard
          label="CVaR 1-Day 95%"
          value={formatPercent(Math.abs(result.cvar_1d))}
          color="#ff6b6b"
        />
        <MetricCard
          label="Max Drawdown"
          value={formatPercent(Math.abs(result.max_drawdown))}
          color="#ff6b6b"
        />
      </div>

      <div className="grid grid-cols-3 gap-4">
        <MetricCard label="Beta vs SPY" value={result.beta.toFixed(2)} color="#ffd166" />
        <MetricCard
          label="Downside Prob"
          value={formatPercent(result.pred_downside_prob)}
          color={result.pred_downside_prob > 0.5 ? '#ff6b6b' : '#ffd166'}
        />
        <MetricCard
          label={`VaR ${result.forecast_horizon_days}-Day`}
          value={formatPercent(Math.abs(result.var_horizon))}
          color="#ff6b6b"
        />
      </div>
    </div>
  );
};

export default RiskPage;
