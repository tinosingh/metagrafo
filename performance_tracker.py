"""
Handles performance metrics and statistics tracking.
"""
from typing import Dict, Any

class PerformanceTracker:
    """Tracks transcription performance metrics."""
    
    def __init__(self):
        self.stats: Dict[str, Any] = {
            'total_chunks': 0,
            'total_audio_sec': 0.0,
            'total_processing_sec': 0.0,
            'last_latency': 0.0
        }
    
    def update_chunk_stats(self, audio_duration: float, processing_time: float):
        """Update statistics after processing an audio chunk."""
        self.stats['total_chunks'] += 1
        self.stats['total_audio_sec'] += audio_duration
        self.stats['total_processing_sec'] += processing_time
        self.stats['last_latency'] = processing_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Return current performance statistics."""
        return self.stats
    
    def get_average_latency(self) -> float:
        """Calculate and return average processing latency."""
        if self.stats['total_chunks'] == 0:
            return 0.0
        return self.stats['total_processing_sec'] / self.stats['total_chunks']
