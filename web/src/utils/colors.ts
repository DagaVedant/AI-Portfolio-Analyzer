export const COLORS = {
  darkBg: '#080c12',
  cardBg: '#0f1824',
  accentGreen: '#3ef5c8',
  accentRed: '#ff6b6b',
  accentYellow: '#ffd166',
  accentBlue: '#58a6ff',
  accentPurple: '#bc8cff',
  textPrimary: '#dde4f0',
  textMuted: '#4e6080',
  borderColor: '#1e2d45',
  gridColor: '#21262d',
};

export const getSentimentColor = (sentiment: 'bullish' | 'bearish' | 'neutral'): string => {
  switch (sentiment) {
    case 'bullish':
      return COLORS.accentGreen;
    case 'bearish':
      return COLORS.accentRed;
    case 'neutral':
      return COLORS.accentYellow;
  }
};

export const getOutlookColor = (outlook: 'Bullish' | 'Bearish' | 'Neutral'): string => {
  switch (outlook) {
    case 'Bullish':
      return COLORS.accentGreen;
    case 'Bearish':
      return COLORS.accentRed;
    case 'Neutral':
      return COLORS.accentYellow;
  }
};

export const getReturnColor = (value: number): string => {
  return value >= 0 ? COLORS.accentGreen : COLORS.accentRed;
};
