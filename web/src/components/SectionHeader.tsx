import React from 'react';

interface SectionHeaderProps {
  title: string;
  subtitle?: string;
}

export const SectionHeader: React.FC<SectionHeaderProps> = ({
  title,
  subtitle,
}) => {
  return (
    <div className="mb-6 pb-3 border-b border-border-color">
      <div className="flex items-baseline gap-3">
        <h2 className="text-lg font-bold text-text-primary tracking-tight">
          {title}
        </h2>
        {subtitle && (
          <span className="text-xs font-mono text-text-muted">
            {subtitle}
          </span>
        )}
      </div>
    </div>
  );
};
