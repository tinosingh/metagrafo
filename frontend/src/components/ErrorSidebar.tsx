import { useState, useEffect } from 'react';
import './ErrorSidebar.css';

type LogLevel = 'error' | 'warning' | 'debug' | 'info';

interface LogEntry {
  id: string;
  timestamp: string;
  level: LogLevel;
  message: string;
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:9001';

function ErrorSidebar() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filter, setFilter] = useState<LogLevel | 'all'>('all');
  const [isCollapsed, setIsCollapsed] = useState(false);
  
  useEffect(() => {
    // Connect to WebSocket for backend logs
    const wsUrl = API_URL.replace('http', 'ws') + '/ws/logs';
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connected for logs');
    };
    
    ws.onmessage = (event) => {
      try {
        const logData = JSON.parse(event.data);
        const newLog: LogEntry = {
          id: crypto.randomUUID(),
          timestamp: new Date().toISOString(),
          level: logData.level || 'info',
          message: logData.message,
        };
        
        setLogs(prev => [newLog, ...prev]);
      } catch (error) {
        console.error('Error parsing log message:', error);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
      console.log('WebSocket closed for logs');
    };
    
    return () => {
      ws.close();
    };
  }, []);
  
  // Capture frontend errors
  useEffect(() => {
    const originalConsoleError = console.error;
    const originalConsoleWarn = console.warn;
    const originalConsoleDebug = console.debug;
    
    console.error = (...args) => {
      originalConsoleError.apply(console, args);
      const message = args.map(arg => 
        typeof arg === 'string' ? arg : JSON.stringify(arg)
      ).join(' ');
      setLogs(prev => [{
        id: crypto.randomUUID(),
        timestamp: new Date().toISOString(),
        level: 'error',
        message: `[Frontend] ${message}`,
      }, ...prev]);
    };
    
    console.warn = (...args) => {
      originalConsoleWarn.apply(console, args);
      const message = args.map(arg => 
        typeof arg === 'string' ? arg : JSON.stringify(arg)
      ).join(' ');
      setLogs(prev => [{
        id: crypto.randomUUID(),
        timestamp: new Date().toISOString(),
        level: 'warning',
        message: `[Frontend] ${message}`,
      }, ...prev]);
    };
    
    console.debug = (...args) => {
      originalConsoleDebug.apply(console, args);
      const message = args.map(arg => 
        typeof arg === 'string' ? arg : JSON.stringify(arg)
      ).join(' ');
      setLogs(prev => [{
        id: crypto.randomUUID(),
        timestamp: new Date().toISOString(),
        level: 'debug',
        message: `[Frontend] ${message}`,
      }, ...prev]);
    };
    
    return () => {
      console.error = originalConsoleError;
      console.warn = originalConsoleWarn;
      console.debug = originalConsoleDebug;
    };
  }, []);
  
  const filteredLogs = filter === 'all' 
    ? logs 
    : logs.filter(log => log.level === filter);
  
  const getLevelClass = (level: LogLevel) => {
    switch (level) {
      case 'error': return 'error';
      case 'warning': return 'warning';
      case 'debug': return 'debug';
      default: return 'info';
    }
  };
  
  return (
    <div className={`error-sidebar ${isCollapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header">
        <h2>Logs & Errors</h2>
        <div className="log-filters">
          <label>
            <input 
              type="radio" 
              name="logLevel" 
              value="all" 
              checked={filter === 'all'} 
              onChange={() => setFilter('all')} 
            /> All
          </label>
          <label>
            <input 
              type="radio" 
              name="logLevel" 
              value="error" 
              checked={filter === 'error'} 
              onChange={() => setFilter('error')} 
            /> Errors
          </label>
          <label>
            <input 
              type="radio" 
              name="logLevel" 
              value="warning" 
              checked={filter === 'warning'} 
              onChange={() => setFilter('warning')} 
            /> Warnings
          </label>
          <label>
            <input 
              type="radio" 
              name="logLevel" 
              value="debug" 
              checked={filter === 'debug'} 
              onChange={() => setFilter('debug')} 
            /> Debug
          </label>
        </div>
        <button 
          className="toggle-button" 
          onClick={() => setIsCollapsed(!isCollapsed)}
        >
          {isCollapsed ? '▲' : '▼'}
        </button>
      </div>
      
      {!isCollapsed && (
        <div className="log-list">
          {filteredLogs.length === 0 ? (
            <div className="no-logs">No logs to display</div>
          ) : (
            filteredLogs.map(log => (
              <div key={log.id} className={`log-entry ${getLevelClass(log.level)}`}>
                <div className="log-timestamp">
                  {new Date(log.timestamp).toLocaleTimeString()}
                </div>
                <div className="log-message">
                  {log.message}
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

export default ErrorSidebar;
