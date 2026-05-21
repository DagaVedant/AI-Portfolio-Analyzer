import { useState, useCallback } from 'react';
import { AnalysisResult } from '../types';
import { apiService } from '../services/api';

interface UseAnalysisReturn {
  result: AnalysisResult | null;
  loading: boolean;
  error: string | null;
  analyze: (ticker: string, horizon: number) => Promise<void>;
  clear: () => void;
}

export const useAnalysis = (): UseAnalysisReturn => {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyze = useCallback(async (ticker: string, horizon: number) => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiService.analyze(ticker, horizon);
      setResult(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Analysis failed';
      setError(message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const clear = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, loading, error, analyze, clear };
};
