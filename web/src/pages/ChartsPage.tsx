import React from 'react';
import Plot from 'react-plotly.js';
import { AnalysisResult } from '../types';

interface ChartPageProps {
  result: AnalysisResult;
}

const ChartsPage: React.FC<ChartPageProps> = ({ result }) => {
  const { ticker, chart_data } = result;

  if (!chart_data || chart_data.dates.length === 0) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-6">Technical Charts for {ticker}</h2>
        <div className="bg-card-bg border border-border-color rounded-lg p-6 text-center text-text-muted">
          <p>No chart data available</p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Technical Charts for {ticker}</h2>

      {/* Price History Chart */}
      <div className="bg-card-bg border border-border-color rounded-lg p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">Extended Price History (365 days)</h3>
        <Plot
          data={[
            {
              x: chart_data.dates,
              y: chart_data.prices,
              type: 'scatter',
              mode: 'lines',
              name: 'Price',
              line: { color: '#00d4aa', width: 2 },
              hovertemplate: '$%{y:.2f}<extra></extra>',
            },
          ]}
          layout={{
            title: '',
            showlegend: false,
            paper_bgcolor: '#1a1a2e',
            plot_bgcolor: '#16213e',
            font: { color: '#e0e0e0' },
            margin: { l: 60, r: 20, t: 20, b: 40 },
            xaxis: {
              title: 'Date',
              gridcolor: '#333',
              showgrid: true,
            },
            yaxis: {
              title: 'Price ($)',
              gridcolor: '#333',
              showgrid: true,
            },
            hovermode: 'x unified',
          }}
          style={{ width: '100%', height: '400px' }}
          config={{ responsive: true, displayModeBar: false }}
        />
      </div>

      {/* Volatility and Drawdown Charts */}
      <div className="grid grid-cols-2 gap-6">
        {/* Volatility Chart */}
        <div className="bg-card-bg border border-border-color rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">30-Day Rolling Volatility</h3>
          <Plot
            data={[
              {
                x: chart_data.dates,
                y: chart_data.volatility_windows,
                type: 'scatter',
                mode: 'lines',
                name: 'Volatility',
                line: { color: '#ffd166', width: 2 },
                fill: 'tozeroy',
                fillcolor: 'rgba(255, 209, 102, 0.2)',
                hovertemplate: '%{y:.2f}%<extra></extra>',
              },
            ]}
            layout={{
              title: '',
              showlegend: false,
              paper_bgcolor: '#1a1a2e',
              plot_bgcolor: '#16213e',
              font: { color: '#e0e0e0' },
              margin: { l: 50, r: 20, t: 20, b: 40 },
              xaxis: {
                title: 'Date',
                gridcolor: '#333',
                showgrid: true,
              },
              yaxis: {
                title: 'Volatility (%)',
                gridcolor: '#333',
                showgrid: true,
              },
              hovermode: 'x unified',
            }}
            style={{ width: '100%', height: '300px' }}
            config={{ responsive: true, displayModeBar: false }}
          />
        </div>

        {/* Drawdown Chart */}
        <div className="bg-card-bg border border-border-color rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Drawdown from Peak</h3>
          <Plot
            data={[
              {
                x: chart_data.dates,
                y: chart_data.drawdown_windows,
                type: 'scatter',
                mode: 'lines',
                name: 'Drawdown',
                line: { color: '#ff6b6b', width: 2 },
                fill: 'tozeroy',
                fillcolor: 'rgba(255, 107, 107, 0.2)',
                hovertemplate: '%{y:.2f}%<extra></extra>',
              },
            ]}
            layout={{
              title: '',
              showlegend: false,
              paper_bgcolor: '#1a1a2e',
              plot_bgcolor: '#16213e',
              font: { color: '#e0e0e0' },
              margin: { l: 50, r: 20, t: 20, b: 40 },
              xaxis: {
                title: 'Date',
                gridcolor: '#333',
                showgrid: true,
              },
              yaxis: {
                title: 'Drawdown (%)',
                gridcolor: '#333',
                showgrid: true,
              },
              hovermode: 'x unified',
            }}
            style={{ width: '100%', height: '300px' }}
            config={{ responsive: true, displayModeBar: false }}
          />
        </div>
      </div>
    </div>
  );
};

export default ChartsPage;
