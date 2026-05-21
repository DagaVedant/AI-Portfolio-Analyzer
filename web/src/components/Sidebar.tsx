import React from 'react';

interface SidebarProps {
  ticker: string;
  onTickerChange: (ticker: string) => void;
  horizon: number;
  onHorizonChange: (horizon: number) => void;
  onAnalyze: () => void;
  onClear: () => void;
  loading: boolean;
}

const POPULAR = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'META', 'AMZN', 'GOOGL', 'AMD', 'SPY', 'QQQ'];

export const Sidebar: React.FC<SidebarProps> = ({
  ticker,
  onTickerChange,
  horizon,
  onHorizonChange,
  onAnalyze,
  onClear,
  loading,
}) => {
  return (
    <div className="w-64 bg-dark-bg border-r border-border-color p-6 sticky top-0 h-screen overflow-y-auto">
      <div className="text-accent-green font-mono text-sm font-bold uppercase tracking-widest mb-6">
        Controls
      </div>

      <div className="mb-6">
        <label className="block text-xs font-mono text-text-muted uppercase tracking-wider mb-2">
          Stock Ticker
        </label>
        <input
          type="text"
          value={ticker}
          onChange={(e) => onTickerChange(e.target.value.toUpperCase())}
          placeholder="AAPL"
          maxLength={5}
          className="w-full bg-card-bg border border-border-color rounded px-3 py-2 text-text-primary font-mono focus:outline-none focus:border-accent-green focus:ring-2 focus:ring-accent-green focus:ring-opacity-20"
        />
      </div>

      <div className="mb-6">
        <label className="block text-xs font-mono text-text-muted uppercase tracking-wider mb-3">
          Quick-pick Tickers
        </label>
        <div className="flex flex-wrap gap-2">
          {POPULAR.map((t) => (
            <button
              key={t}
              onClick={() => onTickerChange(t)}
              className={`px-2 py-1 text-xs font-mono font-bold rounded transition-colors ${
                ticker === t
                  ? 'bg-accent-green text-dark-bg'
                  : 'bg-card-bg border border-border-color text-text-muted hover:text-accent-green'
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      <div className="mb-8">
        <label className="block text-xs font-mono text-text-muted uppercase tracking-wider mb-4">
          Forecast Horizon: {horizon} days
        </label>
        <input
          type="range"
          min={5}
          max={63}
          value={horizon}
          onChange={(e) => onHorizonChange(Number(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-xs font-mono text-text-muted mt-2">
          <span>5</span>
          <span>63</span>
        </div>
      </div>

      <div className="border-t border-border-color pt-6 mb-6">
        <button
          onClick={onAnalyze}
          disabled={loading}
          className="w-full bg-gradient-to-r from-accent-green to-cyan-600 text-dark-bg font-mono font-bold py-2 rounded uppercase text-sm tracking-wide hover:shadow-lg hover:shadow-accent-green/25 disabled:opacity-50 transition-all mb-2"
        >
          {loading ? 'Analysing...' : 'Analyze'}
        </button>
        <button
          onClick={onClear}
          disabled={loading}
          className="w-full bg-card-bg border border-border-color text-text-primary font-mono font-bold py-2 rounded uppercase text-sm tracking-wide hover:text-accent-green hover:border-accent-green transition-colors disabled:opacity-50"
        >
          Clear
        </button>
      </div>

      <div className="border-t border-border-color pt-6 text-xs font-mono text-text-muted leading-relaxed">
        Educational purposes only. No financial advice given or implied. Consult a qualified
        financial advisor before investing.
      </div>
    </div>
  );
};
