"""WebSocket manager for real-time progress updates."""

from __future__ import annotations

# Standard library imports
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

# Third-party imports
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections and progress updates.
    Includes robust ping/pong for connection health and cleanup of stale connections.
    """

    def __init__(self):
        # Stores active WebSocket connections.
        # Each entry is { 'websocket': WebSocket, 'last_pong': datetime, 'ping_task': asyncio.Task }
        self.active_connections: Dict[str, Dict[str, any]] = {}
        self.ping_interval = 30  # seconds: How often to send pings
        self.pong_timeout = 15  # seconds: How long to wait for a pong response
        self.cleanup_interval = 60  # seconds: How often to check for stale connections
        self.logger = logging.getLogger(__name__)
        self._cleanup_task: Optional[asyncio.Task] = None  # Task for periodic cleanup
        self.lock = asyncio.Lock()

    @property
    def active_connections_count(self) -> int:
        """Returns the number of active WebSocket connections."""
        return len(self.active_connections)

    async def connect(self, websocket: WebSocket, client_id: str):
        """
        Registers a new WebSocket connection and starts its ping task.

        Args:
            websocket (WebSocket): The WebSocket object for the new connection.
            client_id (str): A unique identifier for the client.
        """
        async with self.lock:
            await websocket.accept()
            self.logger.info(f"Client {client_id} attempting connection.")

        # Store connection and start a ping task for this client
        ping_task = asyncio.create_task(self._send_pings(client_id))
        self.active_connections[client_id] = {
            "websocket": websocket,
            "last_pong": datetime.now(),  # Record connection time as last pong
            "ping_task": ping_task,  # Store the task to be able to cancel it
        }
        self.logger.info(
            f"Client {client_id} connected. Total connections: {self.active_connections_count}"
        )

        # Start the global cleanup task if it's not already running
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self.start_ping_checker())
            self.logger.info(
                "Started background task for cleaning up stale WebSocket connections."
            )

    async def disconnect(self, client_id: str):
        """
        Removes a WebSocket connection and cancels its associated ping task.

        Args:
            client_id (str): The unique identifier of the client to disconnect.
        """
        async with self.lock:
            if client_id in self.active_connections:
                # Get and cancel the ping task associated with this client
                ping_task = self.active_connections[client_id].get("ping_task")
                if ping_task and not ping_task.done():
                    ping_task.cancel()
                    self.logger.debug(f"Cancelled ping task for client {client_id}.")

                try:
                    await self.active_connections[client_id]["websocket"].close()
                except Exception as e:
                    self.logger.error(f"Error closing connection {client_id}: {e}")
                del self.active_connections[client_id]
                self.logger.info(
                    f"Client {client_id} disconnected. Total connections: {self.active_connections_count}"
                )
            else:
                self.logger.warning(
                    f"Attempted to disconnect non-existent client: {client_id}"
                )

    async def send_message(self, client_id: str, message: str):
        """
        Asynchronously sends a string message to a specific client.
        Includes error handling for closed connections.

        Args:
            client_id (str): The unique identifier of the target client.
            message (str): The string message to send.
        """
        async with self.lock:
            if client_id not in self.active_connections:
                self.logger.warning(
                    f"Cannot send message to non-existent client {client_id}."
                )
                return

            conn = self.active_connections[client_id]
            try:
                await conn["websocket"].send_text(message)
                self.logger.debug(
                    f"Message sent to {client_id}: {message[:50]}..."
                )  # Log first 50 chars
            except (WebSocketDisconnect, RuntimeError) as e:
                self.logger.error(
                    f"Failed to send message to {client_id}: {e}. Disconnecting client.",
                    exc_info=True,
                )
                await self.disconnect(client_id)  # Disconnect client on send error
            except Exception as e:
                self.logger.error(
                    f"Unexpected error sending message to {client_id}: {e}. Disconnecting client.",
                    exc_info=True,
                )
                await self.disconnect(client_id)

    async def send_progress(self, client_id: str, progress: float, status: str):
        """
        Asynchronously sends a JSON message with progress and status information.
        Includes error handling for closed connections.

        Args:
            client_id (str): The unique identifier of the target client.
            progress (float): The current progress percentage (0.0 to 100.0).
            status (str): A descriptive status message.
        """
        async with self.lock:
            if client_id not in self.active_connections:
                self.logger.warning(
                    f"Cannot send progress to non-existent client {client_id}."
                )
                return

            conn = self.active_connections[client_id]
            message = {"progress": progress, "status": status, "client_id": client_id}
            try:
                await conn["websocket"].send_json(message)
                self.logger.debug(
                    f"Progress sent to {client_id}: {progress}% - {status}"
                )
            except (WebSocketDisconnect, RuntimeError) as e:
                self.logger.error(
                    f"Failed to send progress to {client_id}: {e}. Disconnecting client.",
                    exc_info=True,
                )
                await self.disconnect(client_id)  # Disconnect client on send error
            except Exception as e:
                self.logger.error(
                    f"Unexpected error sending progress to {client_id}: {e}. Disconnecting client.",
                    exc_info=True,
                )
                await self.disconnect(client_id)

    async def send_error(self, client_id: str, error: str):
        """
        Asynchronously sends a JSON error message to a specific client.
        Includes error handling for closed connections.

        Args:
            client_id (str): The unique identifier of the target client.
            error (str): The error message to send.
        """
        async with self.lock:
            if client_id not in self.active_connections:
                self.logger.warning(
                    f"Cannot send error to non-existent client {client_id}."
                )
                return

            conn = self.active_connections[client_id]
            message = {"type": "error", "message": error}
            try:
                await conn["websocket"].send_json(message)
                self.logger.debug(f"Error sent to {client_id}: {error}")
            except (WebSocketDisconnect, RuntimeError) as e:
                self.logger.error(
                    f"Failed to send error to {client_id}: {e}. Disconnecting client.",
                    exc_info=True,
                )
                await self.disconnect(client_id)  # Disconnect client on send error
            except Exception as e:
                self.logger.error(
                    f"Unexpected error sending error to {client_id}: {e}. Disconnecting client.",
                    exc_info=True,
                )
                await self.disconnect(client_id)

    async def _send_pings(self, client_id: str):
        """
        Private task to periodically send "ping" messages to a specific client
        and monitor "pong" responses.

        Args:
            client_id (str): The unique identifier of the client.
        """
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                if client_id not in self.active_connections:
                    self.logger.debug(
                        f"Ping task for {client_id} stopping as client disconnected."
                    )
                    break  # Client disconnected, stop this task

                conn = self.active_connections[client_id]

                # Check if pong was received in time since last ping (or connection)
                if datetime.now() - conn["last_pong"] > timedelta(
                    seconds=self.ping_interval + self.pong_timeout
                ):
                    self.logger.warning(
                        f"Client {client_id} did not respond to ping in time. Disconnecting."
                    )
                    await self.disconnect(client_id)
                    break  # Disconnect and stop this task

                # Send ping
                await self._send_ping()

            except asyncio.CancelledError:
                self.logger.info(f"Ping task for {client_id} cancelled.")
                break  # Task was cancelled, exit loop
            except (WebSocketDisconnect, RuntimeError) as e:
                self.logger.error(
                    f"Ping task detected connection error for {client_id}: {e}. Disconnecting.",
                    exc_info=True,
                )
                await self.disconnect(client_id)
                break  # Disconnect and stop this task
            except Exception as e:
                self.logger.error(
                    f"Unexpected error in ping task for {client_id}: {e}. Disconnecting.",
                    exc_info=True,
                )
                await self.disconnect(client_id)
                break

    async def _send_ping(self):
        try:
            for client_id, conn in list(self.active_connections.items()):
                try:
                    # Send ping as proper JSON message
                    await conn["websocket"].send_json(
                        {"type": "ping", "timestamp": time.time()}
                    )
                except Exception as e:
                    self.logger.warning(f"Ping failed for {client_id}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error sending ping: {e}")

    async def handle_message(self, client_id: str, message: str):
        """
        Handles incoming messages from a client.
        This is where you'd implement specific logic for different message types.

        Args:
            client_id (str): The unique identifier of the client.
            message (str): The message received from the client.
        """
        conn = self.active_connections.get(client_id)
        if not conn:
            self.logger.warning(
                f"Received message for non-existent client {client_id}: {message}"
            )
            return

        if message == "pong":
            # Update last_pong time when a pong message is received
            conn["last_pong"] = datetime.now()
            self.logger.debug(
                f"Received pong from client {client_id}. Last pong updated."
            )
        # Add more message handling logic here if needed for other client-to-server messages
        else:
            self.logger.info(
                f"Received unhandled message from client {client_id}: {message}"
            )
            # Example: Echo back unhandled messages, or process them further
            await self.send_message(client_id, f"Server received: {message}")

    async def start_ping_checker(self):
        """
        Starts a periodic background task to clean up stale WebSocket connections.
        This runs independently and ensures inactive clients are eventually disconnected.
        """
        while True:
            await asyncio.sleep(self.cleanup_interval)
            self.logger.info(
                f"Running stale connection cleanup. Active connections: {self.active_connections_count}"
            )
            # Create a list of clients to disconnect to avoid modifying dict during iteration
            stale_clients = []
            for client_id, conn in list(
                self.active_connections.items()
            ):  # Use list() to iterate over a copy
                if datetime.now() - conn["last_pong"] > timedelta(
                    seconds=self.ping_interval * 2 + self.pong_timeout
                ):
                    self.logger.warning(
                        f"Client {client_id} found stale by checker. Last pong: {conn['last_pong']}"
                    )
                    stale_clients.append(client_id)

            for client_id in stale_clients:
                await self.disconnect(client_id)

            if not self.active_connections and self._cleanup_task:
                self.logger.info("No active connections left, stopping cleanup task.")
                self._cleanup_task.cancel()
                self._cleanup_task = None
                break


def create_websocket_manager() -> WebSocketManager:
    """Factory function to create WebSocketManager instance."""
    return WebSocketManager()


# Global singleton instance of WebSocketManager
manager = create_websocket_manager()


class TqdmProgressWrapper:
    """Wraps tqdm to send progress updates via WebSocket."""

    def __init__(
        self,
        client_id: str,
        loop: asyncio.AbstractEventLoop,
        total: Optional[int] = None,
        *args,
        **kwargs,
    ):
        self.client_id = client_id
        self.loop = loop
        self.total = total
        self.current = 0
        self.logger = logging.getLogger(__name__)
        # Store original tqdm kwargs, although not directly used by TqdmProgressWrapper's update logic
        # self._tqdm_kwargs = kwargs

    def update(self, n: int = 1):
        """Updates the progress by n steps and sends it via WebSocket."""
        self.current += n
        if self.total:
            progress = min(100.0, (self.current / self.total) * 100.0)
        else:
            progress = (
                0.0  # Or handle cases where total is unknown (e.g., set to 0 or -1)
            )

        # Ensure progress is always between 0 and 100
        progress = max(0.0, min(100.0, progress))

        if self.loop:
            # Use run_coroutine_threadsafe to send progress updates from potentially a different thread
            # (e.g., transcription running in a ThreadPoolExecutor) back to the main asyncio loop.
            asyncio.run_coroutine_threadsafe(
                manager.send_progress(self.client_id, progress, "transcribing"),
                self.loop,
            )
        else:
            self.logger.warning(
                f"No asyncio loop available for TqdmProgressWrapper for client {self.client_id}. Progress not sent."
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """On exit, send a final 100% progress message thread-safely."""
        if exc_type is None and self.total:
            # Ensure final progress is 100% on successful completion
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    manager.send_progress(self.client_id, 100.0, "Completed"), self.loop
                )
            self.logger.info(
                f"TqdmProgressWrapper for client {self.client_id} exited normally."
            )
        elif exc_type is not None:
            self.logger.error(
                f"TqdmProgressWrapper for client {self.client_id} exited with exception: {exc_val}"
            )
            # Optionally send an error status via WebSocket if the context manager exits due to an exception
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    manager.send_progress(
                        self.client_id, self.current, f"Error: {exc_val}"
                    ),
                    self.loop,
                )
        elif self.total is None:
            self.logger.debug(
                f"TqdmProgressWrapper for client {self.client_id} exited without total value. No final 100% progress sent."
            )
