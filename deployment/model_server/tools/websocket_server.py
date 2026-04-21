"""WebSocket 服务器模块"""
import asyncio
import logging
import traceback
import functools
from typing import Callable, Any, Optional
import numpy as np
import websockets.asyncio.server
import websockets.frames
from msgpack_numpy import Packer, unpackb


class SimpleWebsocketServer:
    """Simple WebSocket server that accepts a callable function as handler
    
    This server doesn't depend on any local files, just needs a callable object to run
    """

    def __init__(
        self,
        handler_func: Callable[[Any], Any],
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Initialize server
        
        Args:
            handler_func: Handler function that receives input data and returns result
            host: Server host address
            port: Server port
        """
        self._handler_func = handler_func
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._packer = Packer()
        
        # Configure logging
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    def serve_forever(self) -> None:
        """Start server and run forever"""
        asyncio.run(self.run())

    async def run(self):
        """Run server asynchronously"""
        async with websockets.asyncio.server.serve(
            self._connection_handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            logging.info(f"WebSocket server started on {self._host}:{self._port}")
            await server.serve_forever()

    async def _connection_handler(self, websocket: websockets.asyncio.server.ServerConnection):
        """Handle WebSocket connection"""
        logging.info(f"New connection from {websocket.remote_address}")

        # Send initial metadata frame to comply with client handshake expectation
        try:
            await websocket.send(self._packer.pack(self._metadata))
        except Exception as e:
            logging.error(f"Failed to send initial metadata: {e}")
            try:
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Failed to send metadata"
                )
            except:
                pass
            return

        while True:
            try:
                # Receive data
                raw_data = await websocket.recv()
                logging.debug(f"Received data: {raw_data}")
                
                try:
                    input_data = unpackb(raw_data)
                except Exception as e:
                    error_msg = f"msgpack parse error: {e}"
                    logging.error(error_msg)
                    await websocket.send(self._packer.pack({"error": error_msg}))
                    continue
                
                # Call handler function
                try:
                    result = self._handler_func(input_data)
                    logging.debug(f"Processing result: {result}")
                except Exception as e:
                    error_msg = f"Handler function execution error: {e}"
                    # Log full traceback to pinpoint the exact failing line
                    logging.exception(error_msg)
                    result = {"error": error_msg, "traceback": traceback.format_exc()}
                
                # Send result
                await websocket.send(self._packer.pack(result))
                
            except websockets.ConnectionClosed:
                logging.info(f"Connection {websocket.remote_address} closed")
                break
            except Exception as e:
                error_msg = f"Connection handling error: {e}"
                logging.error(error_msg)
                try:
                    await websocket.send(self._packer.pack({
                        "error": error_msg,
                        "traceback": traceback.format_exc()
                    }))
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error"
                    )
                except:
                    pass
                break
