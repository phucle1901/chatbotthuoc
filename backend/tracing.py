from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import os

# Initialize tracer provider with service name and Jaeger exporter
SERVICE = os.getenv("OTEL_SERVICE_NAME", "chatbot-service")
JAEGER_HOST = os.getenv("JAEGER_AGENT_HOST", "localhost")
JAEGER_PORT = int(os.getenv("JAEGER_AGENT_PORT", "6831"))

trace.set_tracer_provider(
    TracerProvider(resource=Resource.create({SERVICE_NAME: SERVICE}))
)

jaeger_exporter = JaegerExporter(
    agent_host_name=JAEGER_HOST,
    agent_port=JAEGER_PORT,
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Expose a module-level tracer to import from other modules
tracer = trace.get_tracer(SERVICE)
