import React from 'react';

interface LoadingOverlayProps {
  ticker: string;
  show: boolean;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({ ticker, show }) => {
  if (!show) return null;

  return (
    <div className="fixed inset-0 bg-dark-bg bg-opacity-96 z-50 flex flex-col items-center justify-center">
      <div className="w-24 h-24 border-4 border-border-color border-t-accent-green rounded-full animate-spin-slow mb-6" />
      <div className="text-center">
        <div className="font-mono text-lg font-bold text-accent-green tracking-wider uppercase">
          Analysing {ticker}
        </div>
        <div className="font-mono text-sm text-text-muted mt-2">
          NewsAPI · FinBERT · LSTM model
        </div>
      </div>
    </div>
  );
};
