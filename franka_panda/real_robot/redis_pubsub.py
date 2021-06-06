import pickle
import time
import redis


class RedisPubSub(object):
    def __init__(self, redis_port=6379, pickle_encoding=None):
        self.redis_address = 'localhost'
        self.redis_port = redis_port
        self.redis_request_channel = "redis-request-channel"
        self.redis_response_channel = "redis-response-channel"
        self.connection_ramped = False
        self.pickle_encoding = pickle_encoding

    def _connect(self):
        """
        Connect to redis and subscribe to the pubsub channel
        """
        self.redis_connection = redis.Redis(self.redis_address, self.redis_port)
        self.pubsub = self.redis_connection.pubsub(ignore_subscribe_messages=True)
        self.pubsub.subscribe(self.redis_response_channel)

    def fetch_obj(self, timeout=6000):
        if not self.connection_ramped:
            self._connect()
            self.connection_ramped = True

        timeout_ends = time.time() + timeout
        while time.time() < timeout_ends:
            message = self.pubsub.get_message()

            if message and message["type"] == "message":
                if self.pickle_encoding:
                    obj = pickle.loads(message["data"], encoding=self.pickle_encoding)
                else:
                    obj = pickle.loads(message["data"])
                return obj

            time.sleep(0.01)
        raise ValueError("No obj was published within a timeout of {} seconds. ")

    def publish_obj(self, obj, channel):
        if not self.connection_ramped:
            self._connect()
            self.connection_ramped = True

        self.redis_connection.publish(channel, pickle.dumps(obj, protocol=2))

    def fetch_response(self, timeout=6000):
        try:
            response = self.fetch_obj(timeout=timeout)
        except ValueError:
            raise ValueError("No response was published within a timeout of {} seconds. ".format(timeout))

        return response

    def publish_request(self, request):
        self.publish_obj(request, self.redis_request_channel)

    def run_method(self, method_name, **kwargs):
        self.publish_request((method_name, kwargs))
        response = self.fetch_response()
        if not response[0]:
            raise RuntimeError(response[1])
        return response[1]





