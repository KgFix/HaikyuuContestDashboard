import { useState, useEffect } from 'react';
import { api } from '../api';
import { PerformanceChart } from './PerformanceChart';
import type { ViewType, ChartDataPoint } from '../types';
import './Dashboard.css';

export const Dashboard = () => {
  const [viewType, setViewType] = useState<ViewType>('club');
  const [clubs, setClubs] = useState<string[]>([]);
  const [users, setUsers] = useState<string[]>([]);
  const [selectedEntity, setSelectedEntity] = useState<string>('');
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [activityData, setActivityData] = useState<ChartDataPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load clubs and users on mount
  useEffect(() => {
    const loadData = async () => {
      try {
        const [clubsData, usersData] = await Promise.all([
          api.getClubs(),
          api.getUsers()
        ]);
        setClubs(clubsData);
        setUsers(usersData);
        
        // Select first entity by default
        if (viewType === 'club' && clubsData.length > 0) {
          setSelectedEntity(clubsData[0]);
        } else if (viewType === 'player' && usersData.length > 0) {
          setSelectedEntity(usersData[0]);
        }
      } catch (err) {
        setError('Failed to load initial data. Make sure the API server is running.');
        console.error(err);
      }
    };
    loadData();
  }, []);

  // Load performance data when entity or view type changes
  useEffect(() => {
    if (!selectedEntity) return;

    const loadPerformanceData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        if (viewType === 'club') {
          // Load club performance history
          const history = await api.getClubHistory(selectedEntity);
          const formattedData: ChartDataPoint[] = history.map(item => ({
            date: item.date,
            value: item.maxTotalPower,
            label: 'Max Total Power'
          }));
          setChartData(formattedData);

          // Load club activity history
          const activity = await api.getClubActivity(selectedEntity);
          const activityFormatted: ChartDataPoint[] = activity.map(item => ({
            date: item.date,
            value: item.totalUsers,
            label: 'Active Users'
          }));
          setActivityData(activityFormatted);
        } else {
          // Load user performance history
          const history = await api.getUserHistory(selectedEntity);
          const formattedData: ChartDataPoint[] = history.map(item => ({
            date: item.date,
            value: item.bestHighestToday,
            label: 'Best Score'
          }));
          setChartData(formattedData);
          setActivityData([]);
        }
      } catch (err) {
        setError(`Failed to load ${viewType} data`);
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    loadPerformanceData();
  }, [selectedEntity, viewType]);

  // Handle view type change
  const handleViewTypeChange = (newViewType: ViewType) => {
    setViewType(newViewType);
    // Select first entity of new type
    if (newViewType === 'club' && clubs.length > 0) {
      setSelectedEntity(clubs[0]);
    } else if (newViewType === 'player' && users.length > 0) {
      setSelectedEntity(users[0]);
    }
  };

  const entityList = viewType === 'club' ? clubs : users;

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>Haikyuu Contest Dashboard</h1>
        <p>Track club and player performance over time</p>
      </header>

      <div className="controls">
        <div className="view-toggle">
          <button
            className={`toggle-btn ${viewType === 'club' ? 'active' : ''}`}
            onClick={() => handleViewTypeChange('club')}
          >
            Club View
          </button>
          <button
            className={`toggle-btn ${viewType === 'player' ? 'active' : ''}`}
            onClick={() => handleViewTypeChange('player')}
          >
            Player View
          </button>
        </div>

        <div className="entity-selector">
          <label htmlFor="entity-select">
            Select {viewType === 'club' ? 'Club' : 'Player'}:
          </label>
          <select
            id="entity-select"
            value={selectedEntity}
            onChange={(e) => setSelectedEntity(e.target.value)}
            disabled={loading}
          >
            <option value="">-- Select --</option>
            {entityList.map(entity => (
              <option key={entity} value={entity}>
                {entity}
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && (
        <div className="error-message">
          <p>⚠️ {error}</p>
        </div>
      )}

      {loading && (
        <div className="loading">
          <p>Loading data...</p>
        </div>
      )}

      {!loading && !error && selectedEntity && (
        <div className="charts">
          {viewType === 'club' ? (
            <>
              <PerformanceChart
                data={chartData}
                title={`${selectedEntity} - Total Power Over Time`}
                dataKey="value"
                yAxisLabel="Total Power"
                color="#8884d8"
              />
              <PerformanceChart
                data={activityData}
                title={`${selectedEntity} - Daily Active Users`}
                dataKey="value"
                yAxisLabel="Active Users"
                color="#82ca9d"
              />
            </>
          ) : (
            <PerformanceChart
              data={chartData}
              title={`${selectedEntity} - Daily Best Score`}
              dataKey="value"
              yAxisLabel="Score"
              color="#8884d8"
            />
          )}
        </div>
      )}

      {!loading && !error && !selectedEntity && (
        <div className="no-selection">
          <p>Select a {viewType} to view performance data</p>
        </div>
      )}
    </div>
  );
};
