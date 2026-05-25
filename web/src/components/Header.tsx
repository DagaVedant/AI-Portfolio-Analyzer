import React from 'react';

export const Header: React.FC = () => {
  return (
    <div className="bg-gradient-to-r from-slate-900 to-slate-800 border-b border-border-color px-8 py-6 mb-6 rounded-lg mx-8 mt-8">
      <div className="flex items-baseline gap-3 mb-2">
        <h1 className="text-3xl font-bold text-text-primary">
          AI Portfolio <span className="text-accent-green">Analyzer</span>
        </h1>
      </div>
      <p className="text-text-muted text-sm font-mono tracking-wide">
        ML-driven forecasts + News sentiment via NewsAPI + Risk projection
      </p>
    </div>
  );
};
