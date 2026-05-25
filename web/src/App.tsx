import React, { useState, useEffect } from 'react';
import { useAnalysis } from './hooks/useAnalysis';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { TabNavigation } from './components/TabNavigation';
import { LoadingOverlay } from './components/LoadingOverlay';
import OverviewPage from './pages/OverviewPage';
import ForecastPage from './pages/ForecastPage';
import RiskPage from './pages/RiskPage';
import SentimentPage from './pages/SentimentPage';
import ChartsPage from './pages/ChartsPage';

type TabType = 'overview' | 'forecast' | 'risk' | 'sentiment' | 'charts';

export const App: React.FC = () => {
  const { result, loading, analyze, clear } = useAnalysis();
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [ticker, setTicker] = useState('AAPL');
  const [horizon, setHorizon] = useState(21);

  const handleAnalyze = async () => {
    setActiveTab('overview');
    await analyze(ticker, horizon);
  };

  const handleClear = () => {
    setTicker('AAPL');
    setHorizon(21);
    clear();
  };

  return (
    <div className="min-h-screen bg-dark-bg text-text-primary">
      <LoadingOverlay ticker={ticker} show={loading} />
      
      <div className="flex">
        {/* Sidebar */}
        <Sidebar
          ticker={ticker}
          onTickerChange={setTicker}
          horizon={horizon}
          onHorizonChange={setHorizon}
          onAnalyze={handleAnalyze}
          onClear={handleClear}
          loading={loading}
        />

        {/* Main content */}
        <main className="flex-1">
          <Header />
          
          {result ? (
            <>
              <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />
              <div className="p-8 max-w-7xl mx-auto">
                {activeTab === 'overview' && <OverviewPage result={result} />}
                {activeTab === 'forecast' && <ForecastPage result={result} />}
                {activeTab === 'risk' && <RiskPage result={result} />}
                {activeTab === 'sentiment' && <SentimentPage result={result} />}
                {activeTab === 'charts' && <ChartsPage result={result} />}
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center min-h-screen">
              <div className="text-center">
                <h1 className="text-4xl font-bold mb-3">Enter a ticker and click Analyze</h1>
                <p className="text-text-muted mb-8">
                  ML-powered return forecasts + risk projections + live NewsAPI sentiment
                </p>
                <div className="flex gap-2 flex-wrap justify-center">
                  {['AAPL', 'MSFT', 'NVDA', 'TSLA', 'META', 'SPY', 'QQQ'].map((t) => (
                    <button
                      key={t}
                      onClick={() => setTicker(t)}
                      className="px-4 py-2 bg-card-bg border border-border-color rounded text-accent-green font-mono text-sm hover:border-accent-green transition-colors"
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
