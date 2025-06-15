import React, { useState, useEffect, useRef, useCallback } from 'react';
import './RealTimeTranscription.css';

// Add type declarations for Web Audio API
declare global {
  interface Window {
    webkitAudioContext: typeof AudioContext;
  }
}

const SAMPLE_RATE = 16000;
const CHUNK_SIZE = 1024 * 8; // 8KB chunks
const RECONNECT_DELAY = 1000; // 1 second
const MAX_RECONNECT_ATTEMPTS = 5;

interface RealTimeTranscriptionProps {
  onTranscriptUpdate?: (text: string, isFinal?: boolean) => void;
  language?: string;
  onError?: (error: string) => void;
}

type ConnectionState = 'disconnected' | 'connecting' | 'connected';

export const RealTimeTranscription: React.FC<RealTimeTranscriptionProps> = ({
  onTranscriptUpdate = () => {},
  language = 'en',
  onError = () => {},
}) => {
  // Component state
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [error, setError] = useState<string | null>(null);
  
  // Refs for WebSocket and audio processing
  const ws = useRef<WebSocket | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const mediaStream = useRef<MediaStream | null>(null);
  const processor = useRef<ScriptProcessorNode | null>(null);
  const source = useRef<MediaStreamAudioSourceNode | null>(null);
  
  // Reconnection state
  const reconnectAttempts = useRef<number>(0);
  const reconnectTimer = useRef<number | null>(null);
  const isMounted = useRef(false);
  
  // MLX-Whisper config parameters
  const [mlxConfig, setMlxConfig] = useState({
    gpu: true,
    quantized: false,
    flashAttention: true,
    chunkSize: 30, // seconds
  });

  // Declare messageQueue at component level
  const messageQueue = useRef<ArrayBuffer[]>([]);

  // Get WebSocket URL based on environment
  const getWebSocketUrl = useCallback(() => {
    if (import.meta.env.VITE_WS_URL) return import.meta.env.VITE_WS_URL;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/api/stream`;
  }, []);

  // Safe WebSocket send utility
  const safeSend = useCallback((data: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(data));
    }
  }, []);

  // Process queued audio chunks
  const processQueue = useCallback(() => {
    if (!ws.current || messageQueue.current.length === 0) return;
    while (messageQueue.current.length > 0 && ws.current.readyState === WebSocket.OPEN) {
      const chunk = messageQueue.current.shift();
      if (chunk) ws.current.send(chunk);
    }
  }, []);

  // Send configuration to MLX-Whisper backend
  const sendConfig = useCallback(() => {
    safeSend({
      type: 'config',
      model: 'mlx-whisper',
      language,
      sample_rate: SAMPLE_RATE,
      ...mlxConfig
    });
  }, [language, mlxConfig, safeSend]);

  // WebSocket initialization with MLX support
  const initWebSocket = useCallback(() => {
    const wsUrl = `${getWebSocketUrl()}?model=mlx-whisper`;
    ws.current = new WebSocket(wsUrl);
    
    // Heartbeat for connection monitoring
    const heartbeatInterval = setInterval(() => {
      safeSend({ type: 'ping' });
    }, 30000);

    ws.current.onopen = () => {
      console.debug('[WebSocket] MLX-Whisper connection established');
      setConnectionState('connected');
      sendConfig();
    };

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'gpu_warning') {
          console.warn('MLX GPU Warning:', data.message);
          setMlxConfig(prev => ({ ...prev, gpu: false }));
        }
      } catch (error) {
        console.error('[MLX] Message processing error:', error);
      }
    };

    return () => clearInterval(heartbeatInterval);
  }, [getWebSocketUrl, safeSend, sendConfig]);

  // Initialize WebSocket connection with proper cleanup
  const connectWebSocket = useCallback(() => {
    if (!isMounted.current || reconnectAttempts.current >= MAX_RECONNECT_ATTEMPTS) {
      return;
    }

    initWebSocket();

    if (ws.current) {
      ws.current.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
        setTimeout(connectWebSocket, RECONNECT_DELAY * (reconnectAttempts.current + 1));
        reconnectAttempts.current += 1;
      };

      ws.current.onclose = () => {
        console.debug('[WebSocket] Connection closed');
        if (isRecording && reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
          setTimeout(connectWebSocket, RECONNECT_DELAY * (reconnectAttempts.current + 1));
          reconnectAttempts.current += 1;
        }
      };
    }
  }, [initWebSocket, isRecording]);

  // Implement processQueue usage
  useEffect(() => {
    const interval = setInterval(() => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        processQueue();
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [processQueue]);

  // Stop recording and clean up audio resources
  const stopRecording = useCallback(() => {
    mediaStream.current?.getTracks().forEach(track => track.stop());
    audioContext.current?.close();
    setIsRecording(false);
  }, []);

  // Clean up WebSocket connection and audio resources on unmount
  useEffect(() => {
    isMounted.current = true;
    
    const cleanup = () => {
      isMounted.current = false;
      
      // Clean up WebSocket
      if (ws.current) {
        ws.current.onopen = null;
        ws.current.onclose = null;
        ws.current.onerror = null;
        ws.current.onmessage = null;
        
        if (ws.current.readyState === WebSocket.OPEN) {
          ws.current.close(1000, 'Component unmounting');
        }
        ws.current = null;
      }
      
      // Clear timers
      if (reconnectTimer.current !== null) {
        window.clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
      
      // Clean up audio resources
      stopRecording();
    };
    
    return cleanup;
  }, [stopRecording]);
  
  // Handle recording state changes
  useEffect(() => {
    if (isRecording) {
      startRecording();
    } else {
      stopRecording();
    }
    
    return () => {
      stopRecording();
    };
  }, [isRecording, stopRecording]);
  
  // Start audio recording
  const startRecording = async () => {
    try {
      // Initialize audio context if not already done
      if (!audioContext.current) {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        audioContext.current = new AudioContext({ sampleRate: SAMPLE_RATE });
      }
      
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStream.current = stream;
      
      // Create audio source from microphone
      source.current = audioContext.current.createMediaStreamSource(stream);
      
      // Create script processor for audio processing
      processor.current = audioContext.current.createScriptProcessor(CHUNK_SIZE, 1, 1);
      
      // Process audio chunks
      processor.current.onaudioprocess = (e) => {
        if (ws.current?.readyState === WebSocket.OPEN) {
          // Convert float32 to int16
          const inputData = e.inputBuffer.getChannelData(0);
          const pcmData = new Int16Array(inputData.length);
          
          for (let i = 0; i < inputData.length; i++) {
            // Convert from float32 to int16
            const s = Math.max(-1, Math.min(1, inputData[i]));
            pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }
          
          // Send raw PCM data
          ws.current.send(pcmData.buffer);
        }
      };
      
      // Connect audio nodes
      source.current.connect(processor.current);
      processor.current.connect(audioContext.current.destination);
      
      // Connect to WebSocket if not already connected
      if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
        connectWebSocket();
      }
      
    } catch (err) {
      console.error('Error starting recording:', err);
      setError('Failed to access microphone or start recording');
      onError('Failed to access microphone or start recording');
      setIsRecording(false);
    }
  };
  
  // Clear the transcript
  const clearTranscript = useCallback(() => {
    setTranscript('');
    onTranscriptUpdate('', true);
  }, [onTranscriptUpdate]);

  // Error boundary fallback UI
  if (error) {
    return (
      <div className="error">
        <p>{error}</p>
        <button onClick={() => setError(null)}>Dismiss</button>
      </div>
    );
  }

  // Render UI
  return (
    <div className="transcription-container">
      <div className={`status-indicator ${connectionState}`} aria-live="polite">
        {connectionState === 'connecting' && (
          <span className="spinner">🔄 Connecting...</span>
        )}
        {connectionState === 'connected' && (
          <span className="connected">✅ Connected</span>
        )}
        {connectionState === 'disconnected' && (
          <span className="disconnected">❌ Disconnected</span>
        )}
      </div>
      
      <div className="controls">
        <button 
          className={`record-button ${isRecording ? 'recording' : ''}`}
          onClick={isRecording ? stopRecording : startRecording}
          disabled={connectionState !== 'connected'}
          aria-label={isRecording ? 'Stop recording' : 'Start recording'}
        >
          {isRecording ? (
            <>
              <span className="pulse-icon">🔴</span> Stop Recording
            </>
          ) : (
            <>
              <span className="mic-icon">🎤</span> Start Recording
            </>
          )}
        </button>
        
        <button
          className="clear-button"
          onClick={clearTranscript}
          disabled={!transcript}
          aria-label="Clear transcription"
        >
          🗑 Clear
        </button>
      </div>
      
      {error && (
        <div className="error-message" role="alert">
          <p>Error: {error}</p>
          <button 
            className="retry-button" 
            onClick={connectWebSocket}
            aria-label="Retry connection"
          >
            Retry
          </button>
        </div>
      )}
      
      <div className="transcript" aria-live="polite">
        {transcript || (
          <div className="placeholder">
            {connectionState === 'connected' 
              ? 'Start recording to see transcription...'
              : 'Connect to start transcribing'}
          </div>
        )}
      </div>
    </div>
  );
};
