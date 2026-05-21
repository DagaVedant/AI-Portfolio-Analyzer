import React from 'react';

interface MetricCardProps {
  label: string;
  value: string;
  delta?: string;
  color?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  delta,
  color = '#dde4f0',
}) => {
  return (
    <div className="bg-card-bg border border-border-color rounded-lg p-4 text-center hover:border-accent-green transition-colors">
      <div className="text-text-muted text-xs font-mono font-bold uppercase tracking-widest mb-2">
        {label}
      </div>
      <div style={{ color }} className="text-xl font-mono font-bold">
        {value}
      </div>
      {delta && (
        <div
          style={{ color }}
          className="text-xs font-mono mt-1"
        >
          {delta}
        </div>
      )}
    </div>
  );
};
