import React from 'react';

type TabType = 'overview' | 'forecast' | 'risk' | 'sentiment' | 'charts';

interface TabNavigationProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
}

const TABS: { id: TabType; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'forecast', label: 'Forecast' },
  { id: 'risk', label: 'Risk' },
  { id: 'sentiment', label: 'Sentiment' },
  { id: 'charts', label: 'Charts' },
];

export const TabNavigation: React.FC<TabNavigationProps> = ({ activeTab, onTabChange }) => {
  return (
    <div className="px-8 mb-8">
      <div className="flex gap-1 bg-card-bg border border-border-color rounded-lg p-1 w-fit">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`px-4 py-2 rounded font-mono text-sm font-bold uppercase tracking-wide transition-all ${
              activeTab === tab.id
                ? 'bg-slate-700 text-text-primary border-b-2 border-accent-green'
                : 'text-text-muted hover:text-text-primary'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
    </div>
  );
};
