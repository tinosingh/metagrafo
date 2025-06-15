import { useState, useEffect, useRef } from 'react';
import './App.css';
import ErrorSidebar from './components/ErrorSidebar';
import { RealTimeTranscription } from './components/RealTimeTranscription';

type ExportFormat = 'txt' | 'docx';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:9001';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [transcription, setTranscription] = useState('');
  const [summary, setSummary] = useState('');
  const [autoSummarize, setAutoSummarize] = useState(false);
  const [models, setModels] = useState<{ [key: string]: string }>({});
  const [selectedModel, setSelectedModel] = useState('mlx-medium'); 

  const [wsConnected, setWsConnected] = useState(false);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const messageBuffer = useRef<Array<string>>([]);
  const MAX_BUFFER_SIZE = 100;

  // Use a ref to store the client ID so it persists across renders
  const clientIdRef = useRef(crypto.randomUUID()); 

  // Fetch available models
  useEffect(() => {
    fetch(`${API_URL}/models`)
      .then(response => response.json())
      .then(data => {
        setModels(data.models);
        if (Object.keys(data.models).length > 0) {
          setSelectedModel(Object.keys(data.models)[0]); 
        }
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    // Use the persistent client ID for the WebSocket
    const clientId = clientIdRef.current; 
    // CORRECTED: Build the WebSocket URL with client_id as a path parameter
    const wsUrl = `${API_URL.replace('http', 'ws')}/ws/${clientId}`; 
    
    let socket: WebSocket; 

    const connectWebSocket = () => {
      socket = new WebSocket(wsUrl); 

      socket.onopen = () => {
        messageBuffer.current.forEach(msg => socket.send(msg));
        messageBuffer.current = [];
        
        console.log(`WebSocket connected with ID: ${clientId}`); 
        setWsConnected(true);
        reconnectAttempts.current = 0;
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'ping') {
            return; // Skip ping messages
          }
          // Handle other message types
        } catch (e) {
          console.warn('WebSocket message error:', e);
        }
      };

      const originalSend = socket.send.bind(socket);
      socket.send = (data: string) => {
        if (socket.readyState !== WebSocket.OPEN) {
          if (messageBuffer.current.length < MAX_BUFFER_SIZE) {
            messageBuffer.current.push(data);
          }
          return;
        }
        originalSend(data);
      };

      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        setWsConnected(false);
      };

      socket.onclose = () => {
        console.log(`WebSocket closed for ID: ${clientId}`); 
        setWsConnected(false);
        
        if (reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current += 1;
          const delay = Math.min(1000 * reconnectAttempts.current, 5000);
          setTimeout(connectWebSocket, delay);
        }
      };

      return socket;
    };

    socket = connectWebSocket(); 
    
    return () => {
      socket.close(); 
    };
  }, []); 

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleTranscribe = async () => {
    if (!file || isLoading) return;
    
    setIsLoading(true);
    setProgress(0);
    setTranscription('');
    setSummary('');
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('model_key', selectedModel); 
      formData.append('client_id', clientIdRef.current); // Use the SAME client_id for the HTTP request
      
      const response = await fetch(`${API_URL}/transcribe`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(`Transcription failed: ${errorData.detail || response.statusText}`);
      }
      
      const data = await response.json();
      setTranscription(data.transcription);
      
      if (autoSummarize) {
        setProgress(50); 
        const summaryResponse = await fetch(`${API_URL}/summarize`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: data.transcription }),
        });
        
        if (summaryResponse.ok) {
          const summaryData = await summaryResponse.json();
          setSummary(summaryData.summary);
        } else {
          const errorData = await summaryResponse.json().catch(() => ({ detail: 'Unknown summarization error' }));
          console.error('Summarization failed:', errorData.detail);
        }
      }
      
      setProgress(100);
    } catch (error) {
      console.error('Error during transcription:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleExport = async (format: ExportFormat) => {
    if (!transcription) return;
    
    try {
      const response = await fetch(`${API_URL}/export`, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          text: transcription,
          summary: autoSummarize ? summary : undefined,
          format 
        }),
      });
      
      if (!response.ok) {
        throw new Error('Export failed');
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `transcription.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export error:', error);
    }
  };

  return (
    <div className="app-container">
      <ErrorSidebar /> 
      
      <div className="main-content">
        <header className="app-header">
          <h1>Whisper Transcription Studio</h1>
        </header>
        
        <div className="top-bar">
          <div className="file-upload">
            <input 
              type="file" 
              accept="audio/*" 
              onChange={handleFileChange} 
              disabled={isLoading}
              id="audio-upload"
            />
            <label htmlFor="audio-upload" className="upload-button">
              {file ? 'File Selected' : 'Choose File'}
            </label>
            <span className="file-name">{file?.name || 'No file selected'}</span>
          </div>
          
          <div className="model-select">
            <label>Model:</label>
            <select 
              value={selectedModel} 
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={isLoading}
            >
              {Object.entries(models).map(([key, label]) => (
                <option key={key} value={key}>{label}</option>
              ))}
            </select>
          </div>
          
          <div className="auto-summarize">
            <label>
              <input 
                type="checkbox" 
                checked={autoSummarize} 
                onChange={() => setAutoSummarize(!autoSummarize)} 
                disabled={isLoading}
              />
              Auto-Summarize
            </label>
          </div>
          
          <button 
            className="transcribe-button" 
            onClick={handleTranscribe} 
            disabled={!file || isLoading}
          >
            {isLoading ? (
              <>
                <span className="spinner"></span>
                Stop Transcription
              </>
            ) : 'Transcribe Audio'}
          </button>
          
          {transcription && (
            <div className="export-actions">
              <button onClick={() => handleExport('txt')}>
                TXT
              </button>
              <button onClick={() => handleExport('docx')}>
                DOCX
              </button>
            </div>
          )}
        </div>
        
        <div className="connection-status">
          WebSocket: {wsConnected ? 
            <span style={{color: 'green'}}>Connected</span> : 
            <span style={{color: 'red'}}>Disconnected</span>}
        </div>
        
        {isLoading && (
          <div className="progress-container">
            <div className="progress-bar" style={{ width: `${progress}%` }}></div>
            <div className="progress-text">Progress: {progress.toFixed(0)}%</div> 
          </div>
        )}
        
        <div className="results-container">
          {transcription && (
            <div className="transcription-result">
              <h2>Transcription:</h2>
              <p className="transcription-text">{transcription}</p>
            </div>
          )}
          
          {summary && (
            <div className="summary-result">
              <h2>Summary:</h2>
              <p className="summary-text">{summary}</p>
            </div>
          )}
        </div>
        
        <RealTimeTranscription 
          onTranscriptUpdate={(text) => console.log('New text:', text)}
        />
      </div>
    </div>
  );
}

export default App;
