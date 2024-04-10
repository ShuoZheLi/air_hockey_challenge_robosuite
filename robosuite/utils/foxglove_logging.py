import asyncio
import copy
import multiprocessing
import json
import time
from multiprocessing import Process, Manager
from foxglove_websocket.types import ChannelId, ServiceId, ClientChannelId, ClientChannel
from foxglove_websocket import run_cancellable
from foxglove_websocket.server import FoxgloveServer, FoxgloveServerListener


class Listener(FoxgloveServerListener):
    async def on_subscribe(self, server: FoxgloveServer, channel_id: ChannelId):
        print("First client subscribed to", channel_id)

    async def on_unsubscribe(self, server: FoxgloveServer, channel_id: ChannelId):
        print("Last client unsubscribed from", channel_id)

    async def on_client_advertise(
            self, server: FoxgloveServer, channel: ClientChannel
    ):
        print("Client advertise:", json.dumps(channel))

    async def on_client_unadvertise(
            self, server: FoxgloveServer, channel_id: ClientChannelId
    ):
        print("Client unadvertise:", channel_id)

    async def on_client_message(
            self, server: FoxgloveServer, channel_id: ClientChannelId, payload: bytes
    ):
        msg = json.loads(payload)
        print(f"Client message on channel {channel_id}: {msg}")

    async def on_service_request(
            self,
            server: FoxgloveServer,
            service_id: ServiceId,
            call_id: str,
            encoding: str,
            payload: bytes,
    ) -> bytes:
        if encoding != "json":
            return json.dumps(
                {"success": False, "error": f"Invalid encoding {encoding}"}
            ).encode()

        request = json.loads(payload)
        if "data" not in request:
            return json.dumps(
                {"success": False, "error": f"Missing key 'data'"}
            ).encode()

        print(f"Service request on service {service_id}: {request}")
        return json.dumps(
            {"success": True, "message": f"Received boolean: {request['data']}"}
        ).encode()


class Logger:
    def __init__(self):
        self.process = None
        self.manager = Manager()
        self.shared_dict = self.manager.dict()
        self.lock = self.manager.Lock()  # Create a Lock
        self.has_started = False

    def __deepcopy__(self, memo):
        # Create a new Logger object without copying the Manager object
        new_logger = Logger()
        new_logger.process = copy.deepcopy(self.process, memo)
        # Do not copy the Manager object
        return new_logger

    def start(self):
        if self.has_started:
            return
        self.process = Process(target=self.run_server, daemon=True)
        self.process.start()

        self.has_started = True

    def stop(self):
        if self.process is not None:
            self.process.join()

    def run_server(self):
        run_cancellable(self.send_message_loop())

    async def send_message_loop(self):
        async with FoxgloveServer(
                "0.0.0.0",
                8765,
                "example server",
                capabilities=["clientPublish", "services"],
                supported_encodings=["json"],
        ) as server:
            server.set_listener(Listener())
            chan_id = await server.add_channel(
                {
                    "topic": "example_msg",
                    "encoding": "json",
                }
            )
            await server.add_service(
                {
                    "name": "example_set_bool",
                    "requestSchema": json.dumps(
                        {
                            "type": "object",
                            "properties": {
                                "data": {"type": "boolean"},
                            },
                        }
                    ),
                    "responseSchema": json.dumps(
                        {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},
                                "message": {"type": "string"},
                            },
                        }
                    ),
                    "type": "example_set_bool",
                }
            )

            while True:
                await asyncio.sleep(0.01)
                with self.lock:  # Acquire the lock before reading from the shared_dict
                    msg = dict(self.shared_dict)
                    self.shared_dict.clear()

                    if len(msg):
                        await server.send_message(
                            chan_id,
                            time.time_ns(),
                            json.dumps(msg).encode("utf8"),
                        )

    def send_message(self, json_dump):
        if not self.has_started:
            self.start()

        with self.lock:
            self.shared_dict.update(json_dump)
