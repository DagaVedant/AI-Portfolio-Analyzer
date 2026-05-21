import axios, { AxiosInstance } from 'axios';
import { AnalysisResult, AnalysisRequest, ApiResponse } from '../types';

const API_BASE_URL = '/api';

const client: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes for analysis
});

export const apiService = {
  async analyze(ticker: string, horizon: number): Promise<AnalysisResult> {
    const response = await client.post<ApiResponse<AnalysisResult>>('/analyze', {
      ticker,
      forecast_horizon: horizon,
    });

    if (!response.data.success) {
      throw new Error(response.data.error || 'Analysis failed');
    }

    return response.data.data;
  },

  async getPopularTickers(): Promise<string[]> {
    const response = await client.get<{ tickers: string[] }>('/tickers');
    return response.data.tickers;
  },

  async health(): Promise<any> {
    return await client.get('/health');
  },
};
