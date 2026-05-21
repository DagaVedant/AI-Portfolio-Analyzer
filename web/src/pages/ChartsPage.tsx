import React from 'react';

interface ChartPageProps {
  ticker: string;
}

const ChartsPage: React.FC<ChartPageProps> = ({ ticker }) => {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Technical Charts for {ticker}</h2>
      
      <div className="bg-card-bg border border-border-color rounded-lg p-6 text-center text-text-muted">
        <p>Chart integration coming soon...</p>
        <p className="text-sm mt-2">
          Plotly.js charts (price history, volatility, drawdown) will be rendered here
        </p>
      </div>

      <div className="grid grid-cols-2 gap-6 mt-6">
        <div className="bg-card-bg border border-border-color rounded-lg p-6 text-center text-text-muted h-64">
          Volatility Chart
        </div>
        <div className="bg-card-bg border border-border-color rounded-lg p-6 text-center text-text-muted h-64">
          Drawdown Chart
        </div>
      </div>

      <div className="bg-card-bg border border-border-color rounded-lg p-6 text-center text-text-muted mt-6 h-80">
        Extended Price History (365 days)
      </div>
    </div>
  );
};

export default ChartsPage;
